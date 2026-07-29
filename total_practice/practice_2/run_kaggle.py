import os
import subprocess

def run():
    print("Cloning repo to /tmp...")
    os.chdir("/tmp")
    subprocess.run("git clone -b Viet29072026 https://github.com/TiCoder-coder/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING.git", shell=True, check=True)
    
    os.chdir("/tmp/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/total_practice/practice_2")
    
    print("Installing requirements...")
    subprocess.run("pip install -r requirements.txt", shell=True, check=True)
    
    print("Running pipeline...")
    subprocess.run("python -m processing_own_phase.main", shell=True, check=True)
    
    print("Done! Outputs are saved directly to /kaggle/working/")

if __name__ == "__main__":
    run()
