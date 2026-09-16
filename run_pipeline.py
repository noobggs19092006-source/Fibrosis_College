import os
import sys
import yaml
import logging
import subprocess
import json

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    log_file = config["pipeline"]["log_file"]
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    scripts = ["01_split_and_extract_3d.py", "02_preprocessing.py", "03_train_models.py", "04_validations.py"]
    
    print(f"🚀 Starting ASK1 Publication Pipeline. Logging to {log_file}...")
    logging.info("=== Starting Pipeline Execution ===")
    
    for script in scripts:
        print(f"Running {script}...")
        logging.info(f"Executing: {script}")
        try:
            result = subprocess.run([sys.executable, script], check=True, 
                                    text=True, encoding='utf-8')
            logging.info(f"[{script}] SUCCESS:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            err_msg = f"❌ FATAL ERROR in {script}. Pipeline halted.\nCheck {log_file} for full stack trace."
            print(err_msg)
            logging.error(f"Failed {script} with exit code {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            sys.exit(1)
            
    print("\n🎉 Pipeline completed all 4 phases successfully!")
    
    if os.path.exists("validation_report.json"):
        with open("validation_report.json", "r") as f:
            report = json.load(f)
        print("\n=== Validation Report Summary ===")
        for entry in report:
            print(f"[{entry['split']}] {entry['model']} - {entry['metric']}: {entry['value']:.4f}")

if __name__ == "__main__":
    main()
