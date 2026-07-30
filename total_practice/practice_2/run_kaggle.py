import os
import subprocess
import shutil
import glob

def run():
    print("Preparing code environment on Kaggle...")
    os.chdir("/tmp")
    if os.path.exists("practice_2"):
        shutil.rmtree("practice_2")
    os.makedirs("/tmp/practice_2", exist_ok=True)
    
    # Copy all files from /kaggle/input recursively to /tmp/practice_2
    if os.path.exists("/kaggle/input"):
        print("Copying files from /kaggle/input to /tmp/practice_2...")
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                src_file = os.path.join(root, f)
                dest_file = os.path.join("/tmp/practice_2", f)
                try:
                    shutil.copy2(src_file, dest_file)
                except Exception as e:
                    print(f"Copy warning for {src_file}: {e}")

    os.chdir("/tmp/practice_2")
    
    # Unzip all .zip files
    zip_files = glob.glob("*.zip")
    print(f"Found zip files: {zip_files}")
    for zf in zip_files:
        print(f"Unzipping {zf}...")
        subprocess.run(f"unzip -o -q {zf} -d /tmp/practice_2/", shell=True)
        
    # Restructure configs package if files are flattened
    config_files = ["core_config.py", "experiment_config.py", "training_config.py"]
    if not os.path.exists("configs") or not os.path.exists("configs/core_config.py"):
        os.makedirs("configs", exist_ok=True)
        for cf in config_files:
            if os.path.exists(cf):
                shutil.move(cf, os.path.join("configs", cf))
        if not os.path.exists("configs/__init__.py"):
            with open("configs/__init__.py", "w") as f:
                f.write("from configs.core_config import *\nfrom configs.experiment_config import *\nfrom configs.training_config import *\n")

    # Restructure processing_own_phase package if files are flattened
    phase_files = ["data.py", "evaluate.py", "experiment.py", "logger.py", "main.py", "model.py", "save_load.py", "train.py", "utils.py", "visualize.py"]
    if not os.path.exists("processing_own_phase") or not os.path.exists("processing_own_phase/main.py"):
        os.makedirs("processing_own_phase", exist_ok=True)
        for pf in phase_files:
            if os.path.exists(pf):
                shutil.move(pf, os.path.join("processing_own_phase", pf))
        if not os.path.exists("processing_own_phase/__init__.py"):
            with open("processing_own_phase/__init__.py", "w") as f:
                f.write("# Init processing_own_phase\n")

    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/cifar-10-python.tar.gz") and not os.path.exists("data/cifar-10-batches-py"):
        print("Extracting local CIFAR-10 archive...")
        subprocess.run("tar -xzf data/cifar-10-python.tar.gz -C data/", shell=True)

    print("Listing files in /tmp/practice_2 after setup:")
    subprocess.run("ls -la /tmp/practice_2", shell=True)
    if os.path.exists("processing_own_phase"):
        subprocess.run("ls -la /tmp/practice_2/processing_own_phase", shell=True)
        
    print("Installing requirements...")
    if os.path.exists("requirements.txt"):
        subprocess.run("pip install -r requirements.txt", shell=True)
    
    print("Running pipeline...")
    subprocess.run("python -m processing_own_phase.main", shell=True, check=True)
    
    print("Pipeline complete! Preserving outputs to /kaggle/working/...")
    os.makedirs("/kaggle/working/outputs", exist_ok=True)
    os.makedirs("/kaggle/working/reports", exist_ok=True)
    os.makedirs("/kaggle/working/runs", exist_ok=True)
    subprocess.run("cp -r /tmp/practice_2/outputs/* /kaggle/working/outputs/ 2>/dev/null || true", shell=True)
    subprocess.run("cp -r /tmp/practice_2/reports/* /kaggle/working/reports/ 2>/dev/null || true", shell=True)
    subprocess.run("cp -r /tmp/practice_2/runs/* /kaggle/working/runs/ 2>/dev/null || true", shell=True)
    print("All outputs saved to /kaggle/working/")

if __name__ == "__main__":
    run()

