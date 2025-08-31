import os
import tempfile
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
from datetime import datetime
import torch
import ctranslate2
import faster_whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pyannote.audio import Pipeline
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import numpy as np
import textwrap
import logging
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
torch.set_default_device("cpu")  # Force CPU usage

try:
    import tensorboardx as tensorboard  # Use tensorboardx as a drop-in replacement
except ImportError:
    print("Warning: tensorboardx not found, some logging features may be disabled.")

# Configure logging
logging.basicConfig(filename='summarization.log', level=logging.DEBUG)

# Set environment variables for offline model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_HOME"] = os.path.join(BASE_DIR, "models", "pyannote")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "models", "huggingface")

# Use environment variable for INPUT_MP3 if set, otherwise fallback to default
INPUT_MP3 = os.environ.get("INPUT_MP3", os.path.join(BASE_DIR, "002145_a-conversation-with-a-neighbor-53032.mp3"))
MODEL_PATH = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", BASE_DIR)
SAMPLERATE = 16000

def normalize_audio(audio_segment):
    """Normalize audio amplitude to improve ASR and diarization."""
    print("Normalizing audio...")
    normalized = audio_segment.normalize(headroom=1.0)
    normalized = normalized.apply_gain(-normalized.dBFS).low_pass_filter(3000).high_pass_filter(100)
    return normalized

def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to 16kHz mono WAV with normalization and progress bar."""
    print("Converting MP3 to WAV...")
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Input MP3 file not found: {mp3_path}")
    AudioSegment.converter = os.path.join(BASE_DIR, "bin", "ffmpeg")
    AudioSegment.ffmpeg = os.path.join(BASE_DIR, "bin", "ffmpeg")
    AudioSegment.ffprobe = os.path.join(BASE_DIR, "bin", "ffprobe")
    
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio_duration = len(audio) / 1000
        audio = audio.set_channels(1).set_frame_rate(SAMPLERATE)
        audio = normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with tqdm(total=100, desc="Converting audio", unit="%") as pbar:
                audio.export(f.name, format="wav")
                pbar.update(100)
            print("MP3 to WAV conversion completed.")
            return f.name, audio_duration
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        logging.error(f"Error converting MP3 to WAV: {str(e)}")
        raise

def run_diarization(wav_path, num_speakers, audio_duration):
    """Run speaker diarization with improved accuracy using local models."""
    print("Running diarization...")
    logging.debug("Starting diarization with wav_path: %s, num_speakers: %d, audio_duration: %.2f", wav_path, num_speakers, audio_duration)
    try:
        diarization_model_path = os.path.join(MODEL_PATH, "pyannote", "models--pyannote--speaker-diarization-3.1", "snapshots", "84fd25912480287da0247647c3d2b4853cb3ee5d")
        if not os.path.exists(diarization_model_path):
            raise FileNotFoundError(f"Pyannote model directory not found at {diarization_model_path}. Please run download_models.py to download models.")
        
        # Load pipeline from local config
        config_path = os.path.join(diarization_model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}. Ensure pyannote/speaker-diarization-3.1 is downloaded correctly.")
        
        pipeline = Pipeline.from_pretrained(
            config_path,
            cache_dir=diarization_model_path,
            use_auth_token=False  # Disable online authentication
        )
        pipeline.to(torch.device("cpu"))
        print("Diarization pipeline loaded successfully.")
        logging.debug("Diarization pipeline loaded successfully from %s", config_path)
        
        diarization = pipeline(
            wav_path,
            num_speakers=num_speakers,
            min_speakers=num_speakers,
            max_speakers=num_speakers
        )
        
        diar_segments = []
        with tqdm(total=int(audio_duration), desc="Processing diarization", unit="s") as pbar:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diar_segments.append((turn.start, turn.end, speaker))
                pbar.update(int(turn.end - turn.start))
        
        merged_segments = []
        if diar_segments:
            start, end, speaker = diar_segments[0]
            for next_start, next_end, next_speaker in diar_segments[1:]:
                if speaker == next_speaker and (next_start - end) < 0.5:
                    end = next_end
                else:
                    merged_segments.append((start, end, speaker))
                    start, end, speaker = next_start, next_end, next_speaker
            merged_segments.append((start, end, speaker))
        
        print(f"Diarization complete. Found {len(merged_segments)} segments.")
        logging.debug("Diarization completed with %d segments", len(merged_segments))
        return merged_segments
    except Exception as e:
        print(f"Error running diarization: {e}")
        logging.error("Error running diarization: %s", str(e))
        raise

def run_asr(wav_path, diar_segments):
    """Run ASR with Whisper small model and optimized parameters."""
    print("Running ASR...")
    logging.debug("Starting ASR with wav_path: %s, diar_segments: %d", wav_path, len(diar_segments))
    try:
        small_path = os.path.join(MODEL_PATH, "small", "models--Systran--faster-whisper-small", "snapshots", "536b0662742c02347bc0e980a01041f333bce120")
        if not os.path.exists(small_path):
            raise FileNotFoundError(f"Whisper small model directory not found at {small_path}. Please ensure the folder contains the model.bin file.")
        if not os.path.exists(os.path.join(small_path, "model.bin")):
            raise FileNotFoundError(f"model.bin not found at {small_path}. Please verify the model download.")
        print(f"Attempting to load WhisperModel with ctranslate2 version: {ctranslate2.__version__}, faster_whisper version: {faster_whisper.__version__}")
        model = WhisperModel(small_path, device="cpu")
        
        rate, data = wavfile.read(wav_path)
        asr_results = []
        
        with tqdm(total=len(diar_segments), desc="Transcribing segments", unit="segment") as pbar:
            for start, end, _ in diar_segments:
                start_sample = int(start * rate)
                end_sample = int(end * rate)
                if end_sample > len(data):
                    end_sample = len(data)
                segment_data = data[start_sample:end_sample]
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wavfile.write(f.name, rate, segment_data)
                    temp_segment_path = f.name
                
                segments, _ = model.transcribe(
                    temp_segment_path,
                    beam_size=15,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        threshold=0.5
                    ),
                    temperature=0.0,
                    best_of=5,
                    patience=1.0
                )
                text = " ".join(seg.text for seg in segments).strip() or "No speech detected"
                if text != "No speech detected":
                    text = text[0].upper() + text[1:]
                    if not text.endswith((".", "!", "?")):
                        text += "."
                asr_results.append((start, end, text))
                os.remove(temp_segment_path)
                pbar.update(1)
        
        print(f"ASR complete. Transcribed {len(asr_results)} segments.")
        logging.debug("ASR completed with %d segments", len(asr_results))
        return asr_results
    except Exception as e:
        print(f"Error running ASR: {e}")
        logging.error("Error running ASR: %s", str(e))
        raise

def save_outputs(diar_segments, asr_results):
    """Save diarization and ASR outputs to files."""
    print("Saving diarization and ASR outputs...")
    logging.debug("Saving outputs to diarization_output.txt and asr_output.txt")
    diar_path = os.path.join(OUTPUT_DIR, "diarization_output.txt")
    asr_path = os.path.join(OUTPUT_DIR, "asr_output.txt")
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(diar_path, "w", encoding="utf-8") as f:
            for start, end, speaker in diar_segments:
                f.write(f"[{start:.1f} - {end:.1f}] {speaker}\n")
        
        with open(asr_path, "w", encoding="utf-8") as f:
            for start, end, text in asr_results:
                f.write(f"[{start:.1f} - {end:.1f}] {text}\n")
        print(f"Saved diarization to {diar_path}")
        print(f"Saved ASR output to {asr_path}")
        logging.debug("Successfully saved diarization to %s and ASR to %s", diar_path, asr_path)
    except Exception as e:
        print(f"Error saving outputs: {e}")
        logging.error("Error saving outputs: %s", str(e))
        raise

def generate_minutes():
    """Generate professional, general-purpose meeting minutes."""
    print("===== Generating Meeting Minutes =====")
    logging.debug("Starting generate_minutes() function.")
    
    # Set the relative path to the bart-samsum model
    bart_path = os.path.join(MODEL_PATH, "bart-samsum", "models--facebook--bart-large-cnn", "snapshots", "37f520fa929c961707657b28798b30c003dd100b")
    logging.debug("Attempting to load BART tokenizer and model from path: %s", bart_path)

    try:
        print("Loading tokenizer and model...")
        if not os.path.exists(bart_path):
            logging.error("BART model directory not found at %s", bart_path)
            raise FileNotFoundError(f"BART model not found at {bart_path}. Please run download_models.py to download it.")
        
        tokenizer = AutoTokenizer.from_pretrained(bart_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(bart_path, local_files_only=True)
        print("Tokenizer and model loaded successfully.")
        logging.debug("BART tokenizer and model loaded successfully.")
        
        model = model.to(torch.device("cpu"))
    except Exception as e:
        print(f"Error loading tokenizer or model: {e}")
        logging.error("Error loading tokenizer or model: %s", str(e))
        return False

    diar_path = os.path.join(OUTPUT_DIR, "diarization_output.txt")
    asr_path = os.path.join(OUTPUT_DIR, "asr_output.txt")
    
    try:
        logging.debug("Reading diarization file: %s", diar_path)
        if not os.path.exists(diar_path):
            logging.error("Diarization file not found: %s", diar_path)
            raise FileNotFoundError(f"Diarization file not found: {diar_path}")
        with open(diar_path, "r", encoding="utf-8") as f:
            diar_lines = [line.strip() for line in f if line.strip()]
        
        logging.debug("Reading ASR file: %s", asr_path)
        if not os.path.exists(asr_path):
            logging.error("ASR file not found: %s", asr_path)
            raise FileNotFoundError(f"ASR file not found: {asr_path}")
        with open(asr_path, "r", encoding="utf-8") as f:
            asr_lines = [line.strip() for line in f if line.strip()]
            
        if len(diar_lines) != len(asr_lines):
            print(f"Error: Mismatch in line counts (Diarization: {len(diar_lines)}, ASR: {len(asr_lines)})")
            logging.error("Mismatch in line counts - Diarization: %d, ASR: %d", len(diar_lines), len(asr_lines))
            return False
            
        print("Diarization and ASR files loaded successfully.")
        logging.debug("Diarization and ASR files loaded successfully.")
    except Exception as e:
        print(f"Error loading diarization or ASR files: {e}")
        logging.error("Error loading diarization or ASR files: %s", str(e))
        return False

    structured_transcript = []
    speaker_set = set()
    
    for i in range(len(diar_lines)):
        try:
            speaker_id = diar_lines[i].split("]")[1].strip()
            speaker_name = speaker_id.replace("_", " ").capitalize()
            utterance = asr_lines[i].split("]")[1].strip()
            if utterance.lower() == "no speech detected":
                continue
            structured_transcript.append(f"{speaker_name}: {utterance}")
            speaker_set.add(speaker_name)
        except IndexError:
            print(f"Error parsing line {i+1}: Invalid format in diarization or ASR.")
            logging.error("Error parsing line %d: Invalid format in diarization or ASR.", i+1)
            return False

    print("\n===== Structured Transcript =====")
    for line in structured_transcript:
        print(line)
    print("=================================\n")
    logging.debug("Structured transcript generated with %d entries.", len(structured_transcript))

    attendees = ", ".join(sorted(speaker_set))
    if not structured_transcript:
        print("Error: No meaningful transcript generated. Cannot produce meeting minutes.")
        logging.error("No meaningful transcript generated.")
        return False

    chunks = []
    current_chunk = ""
    max_input_tokens = 900

    for line in structured_transcript:
        test_chunk = current_chunk + line + "\n"
        token_count = len(tokenizer(test_chunk).input_ids)
        if token_count > max_input_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Created {len(chunks)} chunk(s) for summarization.")
    logging.debug("Created %d chunks for summarization.", len(chunks))
    if not chunks:
        print("Error: No chunks created for summarization.")
        logging.error("No chunks created for summarization.")
        return False

    all_discussion_points = []
    action_items = []
    key_decisions = []
    concerns_raised = []
    detailed_content = []
    questions_raised = []
    
    with tqdm(total=len(chunks), desc="Generating summaries", unit="chunk") as pbar:
        for i, chunk in enumerate(chunks):
            print(f"\nGenerating summary for chunk {i+1}...")
            logging.debug("Generating summary for chunk %d.", i+1)
            
            prompt = f"{chunk}\n\nProvide a detailed summary capturing all key discussion points, decisions, action items, questions, and concerns raised. Include specific contributions from each speaker and highlight any agreements, disagreements, or unresolved issues:"
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(torch.device("cpu"))
                input_token_count = len(inputs["input_ids"][0])
                print(f"Chunk {i+1} input token count: {input_token_count}")
                logging.debug("Chunk %d input token count: %d", i+1, input_token_count)
                
                generation_config = GenerationConfig(
                    max_length=400,
                    min_length=100,
                    num_beams=8,
                    length_penalty=1.5,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    bos_token_id=0,
                    eos_token_id=2,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=0
                )
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=generation_config)
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                logging.debug("Summary generated for chunk %d: %s", i+1, summary)
                
                if summary.lower().startswith("provide a detailed summary"):
                    summary = summary[summary.lower().find(":")+1:].strip()
                
                word_count = len(summary.split())
                print(f"Chunk {i+1} summary word count: {word_count}")
                logging.debug("Chunk %d summary word count: %d", i+1, word_count)
                
                if not summary or word_count < 30:
                    lines = chunk.split('\n')
                    key_content = []
                    for line in lines:
                        if ':' in line and len(line.split(':')[1].strip()) > 15:
                            speaker = line.split(':')[0].strip()
                            content = line.split(':', 1)[1].strip()
                            key_content.append(f"{speaker} discussed: {content}")
                    summary = '. '.join(key_content[:8]) + '.' if key_content else "No significant discussion captured for this segment."
                    logging.debug("Fallback summary for chunk %d: %s", i+1, summary)
                
                summary_sentences = [s.strip() + "." for s in summary.split(".") if s.strip() and len(s.strip()) > 10]
                
                lines = chunk.split('\n')
                for line in lines:
                    if ':' in line:
                        speaker = line.split(':')[0].strip()
                        content = line.split(':', 1)[1].strip()
                        if len(content) > 20:
                            formatted_point = f"{speaker} stated: {content}"
                            detailed_content.append(formatted_point)
                        if '?' in content:
                            questions_raised.append(f"{speaker} asked: {content}")
                
                for sentence in summary_sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in ["will ", "assigned to", "task", "follow up", "responsible for", "take action"]):
                        action_items.append(sentence)
                    elif any(keyword in sentence_lower for keyword in ["decided", "agreed", "concluded", "resolved", "approved", "finalized"]):
                        key_decisions.append(sentence)
                    elif any(keyword in sentence_lower for keyword in ["question", "concern", "issue", "problem", "challenge", "uncertainty", "doubt"]):
                        concerns_raised.append(sentence)
                    else:
                        all_discussion_points.append(sentence)
                
            except Exception as e:
                print(f"Error generating summary for chunk {i+1}: {e}")
                logging.error("Error generating summary for chunk %d: %s", i+1, str(e))
                lines = chunk.split('\n')
                for line in lines[:15]:
                    if ':' in line and len(line.split(':')[1].strip()) > 15:
                        all_discussion_points.append(f"Discussion point: {line.split(':', 1)[1].strip()}")
            
            pbar.update(1)

    meeting_minutes = f"""Meeting Minutes
===============

• Meeting Title: General Meeting
• Date: {datetime.today().strftime('%B %d, %Y')}
• Time: {datetime.today().strftime('%I:%M %p IST')}
• Location: Not specified (Virtual/In-Person)
• Attendees: {attendees}
• Minute-Taker: Automated Assistant

Purpose:
--------
• To discuss topics raised during the meeting and address relevant issues, decisions, and action items.

Agenda Items:
-------------
• Review of key discussion points and contributions from attendees.
• Identification of decisions, action items, and concerns.
• Planning for follow-up actions and next steps.

Detailed Discussion Points:
--------------------------"""
    
    all_content = all_discussion_points + detailed_content
    seen = set()
    unique_content = []
    for item in all_content:
        if item not in seen:
            seen.add(item)
            unique_content.append(item)
    
    if unique_content:
        for point in unique_content:
            wrapped_point = textwrap.fill(point, width=80, subsequent_indent="  ")
            meeting_minutes += f"\n• {wrapped_point}"
    else:
        meeting_minutes += "\n• No specific discussion points were captured."

    meeting_minutes += "\n\nKey Questions Raised:\n----------------------"
    if questions_raised:
        for question in questions_raised[:15]:
            wrapped_question = textwrap.fill(question, width=80, subsequent_indent="  ")
            meeting_minutes += f"\n• {wrapped_question}"
    else:
        meeting_minutes += "\n• No specific questions were identified in the transcript."

    meeting_minutes += "\n\nKey Decisions Made:\n--------------------"
    if key_decisions:
        for decision in key_decisions:
            wrapped_decision = textwrap.fill(decision, width=80, subsequent_indent="  ")
            meeting_minutes += f"\n• {wrapped_decision}"
    else:
        meeting_minutes += "\n• No formal decisions were recorded during the meeting."

    meeting_minutes += "\n\nConcerns and Uncertainties Raised:\n----------------------------------"
    if concerns_raised:
        for concern in concerns_raised:
            wrapped_concern = textwrap.fill(concern, width=80, subsequent_indent="  ")
            meeting_minutes += f"\n• {wrapped_concern}"
    else:
        meeting_minutes += "\n• No specific concerns or uncertainties were identified."

    meeting_minutes += "\n\nAction Items:\n--------------"
    if action_items:
        for item in action_items:
            wrapped_item = textwrap.fill(item, width=80, subsequent_indent="  ")
            meeting_minutes += f"\n• {wrapped_item}"
    else:
        meeting_minutes += "\n• No specific action items were assigned."

    meeting_minutes += """

Next Steps:
-----------
• Schedule the next meeting to continue discussions and review progress.
• Follow up on assigned action items before the next meeting.
• Address any outstanding concerns or questions raised during this session.

Additional Notes:
----------------
• The meeting covered a range of topics, with active participation from all attendees.
• Further details or clarifications may be required for unresolved issues.

===============
End of Minutes
"""

    output_path = os.path.join(OUTPUT_DIR, "meeting_minutes.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(meeting_minutes)
        print(f"\n✅ Meeting minutes saved to: {output_path}")
        logging.debug("Meeting minutes saved to: %s", output_path)
    except Exception as e:
        print(f"Error saving meeting minutes: {e}")
        logging.error("Error saving meeting minutes: %s", str(e))
        return False

    print("\n===== Meeting Minutes =====")
    print(meeting_minutes)
    print("==========================\n")
    logging.debug("generate_minutes() completed successfully.")
    
    return True

def main():
    """Main function to orchestrate the pipeline."""
    print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    logging.debug("Starting execution at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    while True:
        try:
            num_speakers = int(input("Enter the number of participants (e.g., 3): "))
            if num_speakers > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    wav_path = None
    try:
        wav_path, audio_duration = convert_mp3_to_wav(INPUT_MP3)
        diar_segments = run_diarization(wav_path, num_speakers, audio_duration)
        asr_results = run_asr(wav_path, diar_segments)
        save_outputs(diar_segments, asr_results)
        
        if generate_minutes():
            print("✅ Meeting minutes generated successfully!")
        else:
            print("❌ Failed to generate meeting minutes. Please review the errors above.")
    
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"Cleaned up temporary WAV file: {wav_path}")
            except Exception as e:
                print(f"Error cleaning up temporary WAV file: {e}")
                logging.error("Error cleaning up temporary WAV file: %s", str(e))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--subprocess":
        # Run pipeline directly with provided num_speakers
        num_speakers = int(sys.argv[2])
        INPUT_MP3 = os.environ.get("INPUT_MP3", os.path.join(BASE_DIR, "002145_a-conversation-with-a-neighbor-53032.mp3"))
        OUTPUT_DIR = os.environ.get("OUTPUT_DIR", BASE_DIR)
        wav_path = None
        try:
            wav_path, audio_duration = convert_mp3_to_wav(INPUT_MP3)
            diar_segments = run_diarization(wav_path, num_speakers, audio_duration)
            asr_results = run_asr(wav_path, diar_segments)
            save_outputs(diar_segments, asr_results)
            generate_minutes()
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
    else:
        main()