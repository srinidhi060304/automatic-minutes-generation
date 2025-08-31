import zipfile
import os
import sys

def check_wheel_cpu_gpu(wheel_path):
    print(f"Analyzing {wheel_path}...")
    try:
        with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
            # List all files in the .whl
            file_list = zip_ref.namelist()
            
            # Check for CUDA-related files or libraries
            cuda_indicators = ['cuda', 'cudnn', 'cublas', 'curand', 'cusparse', 'libcuda']
            gpu_found = False
            for file_name in file_list:
                if any(indicator in file_name.lower() for indicator in cuda_indicators):
                    print(f"GPU indicator found: {file_name}")
                    gpu_found = True
            
            # Check for CPU-only hints (less definitive)
            cpu_only = not gpu_found and all('dist-info' in f or 'py' in f or 'data' in f for f in file_list)
            
            if gpu_found:
                print("This .whl likely includes GPU support (CUDA-related files detected).")
            elif cpu_only:
                print("This .whl appears to be CPU-only (no GPU indicators found).")
            else:
                print("Unable to definitively determine CPU/GPU status. Manual verification recommended.")
            
            # Extract metadata (if available)
            for file_name in file_list:
                if file_name.endswith('METADATA') or file_name.endswith('RECORD'):
                    with zip_ref.open(file_name) as metadata_file:
                        metadata = metadata_file.read().decode('utf-8', errors='ignore')
                        print(f"Metadata snippet: {metadata[:200]}...")  # Print first 200 chars
                        if any(indicator in metadata.lower() for indicator in cuda_indicators):
                            print("Metadata suggests GPU support.")
                            gpu_found = True
            if not gpu_found:
                print("No GPU-related metadata found.")
                
    except zipfile.BadZipFile:
        print("Invalid .whl file.")
    except Exception as e:
        print(f"Error analyzing .whl: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_wheel.py path_to_wheel.whl")
    else:
        wheel_path = sys.argv[1]
        if os.path.exists(wheel_path):
            check_wheel_cpu_gpu(wheel_path)
        else:
            print("File not found.")