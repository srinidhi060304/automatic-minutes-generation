import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QFileDialog, QSpinBox, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime
import logging
logging.getLogger('PyQt5').setLevel(logging.WARNING)

class ScriptRunnerThread(QThread):
    output_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, script_path, num_speakers, output_dir, mp3_path):
        super().__init__()
        self.script_path = script_path
        self.num_speakers = num_speakers
        self.output_dir = output_dir
        self.mp3_path = mp3_path
        self.minutes_path = os.path.join(output_dir, "meeting_minutes.txt")

    def run(self):
        try:
            env = os.environ.copy()
            env["OUTPUT_DIR"] = self.output_dir
            env["INPUT_MP3"] = self.mp3_path
            process = subprocess.Popen(
                [sys.executable, self.script_path, "--subprocess", str(self.num_speakers)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env
            )
            stdout, stderr = process.communicate(input=str(self.num_speakers) + "\n")
            stages = [
                ("Converting MP3 to WAV...", 10, "MP3 Conversion"),
                ("Running diarization...", 40, "Diarization"),
                ("Running ASR...", 90, "Transcription"),
                ("Generating Meeting Minutes", 100, "Minute Generation")
            ]
            current_progress = 0
            skip_minutes = False
            for line in stdout.splitlines():
                if "===== Meeting Minutes =====" in line:
                    skip_minutes = True
                    continue
                if "==========================" in line and skip_minutes:
                    skip_minutes = False
                    continue
                if skip_minutes:
                    continue
                self.output_signal.emit(line)
                for keyword, progress, stage in stages:
                    if keyword in line:
                        if progress > current_progress:
                            current_progress = progress
                            self.progress_signal.emit(current_progress, stage)
                            break
            if stderr:
                self.error_signal.emit(stderr)
            if os.path.exists(self.minutes_path):
                with open(self.minutes_path, "r", encoding="utf-8") as f:
                    minutes_content = f.read()
                    self.finished_signal.emit(minutes_content)
            else:
                self.error_signal.emit("Meeting minutes file not found. Check console output for errors.")
        except Exception as e:
            self.error_signal.emit(f"Error running script: {str(e)}")

class MeetingMinutesUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Minutes Generator")
        self.setGeometry(100, 100, 800, 600)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_path = os.path.join(self.base_dir, "test2.py")
        self.output_dir = self.base_dir
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        file_layout = QHBoxLayout()
        self.mp3_path_input = QLineEdit()
        self.mp3_path_input.setPlaceholderText("Select MP3 file...")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_mp3)
        file_layout.addWidget(QLabel("MP3 File:"))
        file_layout.addWidget(self.mp3_path_input)
        file_layout.addWidget(browse_button)
        main_layout.addLayout(file_layout)

        speakers_layout = QHBoxLayout()
        self.speakers_input = QSpinBox()
        self.speakers_input.setRange(1, 10)
        self.speakers_input.setValue(3)
        speakers_layout.addWidget(QLabel("Number of Speakers:"))
        speakers_layout.addWidget(self.speakers_input)
        speakers_layout.addStretch()
        main_layout.addLayout(speakers_layout)

        output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.output_dir)
        self.output_dir_input.setReadOnly(True)
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        output_dir_layout.addWidget(self.output_dir_input)
        main_layout.addLayout(output_dir_layout)

        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Generate Minutes")
        self.run_button.clicked.connect(self.run_script)
        clear_button = QPushButton("Clear Output")
        clear_button.clicked.connect(self.clear_output)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(clear_button)
        main_layout.addLayout(button_layout)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Output:"))
        main_layout.addWidget(self.output_text)

    def browse_mp3(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MP3 File", "", "MP3 Files (*.mp3)")
        if file_path:
            self.mp3_path_input.setText(file_path.replace("/", "\\"))

    def run_script(self):
        """Run test2.py with user inputs."""
        mp3_path = self.mp3_path_input.text()
        num_speakers = self.speakers_input.value()
        if not mp3_path or not os.path.exists(mp3_path):
            QMessageBox.warning(self, "Error", "Please select a valid MP3 file.")
            return
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Error", "Processing is already in progress.")
            return
        self.output_text.clear()
        self.output_text.append(f"Starting process at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.output_text.append(f"MP3 File: {mp3_path}")
        self.output_text.append(f"Number of Speakers: {num_speakers}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Status: Initializing")
        self.run_button.setEnabled(False)
        # Clear existing output files
        for file in ["meeting_minutes.txt", "diarization_output.txt", "asr_output.txt"]:
            path = os.path.join(self.output_dir, file)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    self.output_text.append(f"Cleared existing file: {path}")
                except Exception as e:
                    self.output_text.append(f"Error clearing file: {str(e)}")
        # Run script in thread
        self.thread = ScriptRunnerThread(self.script_path, num_speakers, self.output_dir, mp3_path)
        self.thread.output_signal.connect(self.append_output)
        self.thread.error_signal.connect(self.append_error)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.finish_processing)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.start()

    def append_output(self, text):
        self.output_text.append(text)

    def append_error(self, text):
        self.output_text.append(f"Error: {text}")

    def update_progress(self, value, stage):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"Status: {stage}")

    def finish_processing(self, minutes_content):
        self.output_text.append("\nMeeting Minutes:")
        self.output_text.append(minutes_content)
        self.progress_label.setText("Status: Completed")

    def cleanup_thread(self):
        self.run_button.setEnabled(True)
        self.thread = None

    def clear_output(self):
        self.output_text.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Status: Idle")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeetingMinutesUI()
    window.show()
    sys.exit(app.exec_())