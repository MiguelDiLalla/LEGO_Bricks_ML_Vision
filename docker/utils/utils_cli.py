import argparse
import subprocess
import logging
import os

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="CLI for running pipeline utilities inside Docker")
    
    parser.add_argument("--convert-labelme", nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Convert LabelMe JSON annotations to YOLO format")
    parser.add_argument("--detect-bricks", metavar="IMAGE", help="Run brick detection on an image")
    parser.add_argument("--detect-studs", metavar="IMAGE", help="Run stud detection on an image")
    parser.add_argument("--classify-studs", metavar="IMAGE", help="Classify LEGO brick based on detected studs")
    
    args = parser.parse_args()
    
    command = ["python", "pipeline_utils_docker.py"]
    
    if args.convert_labelme:
        command.extend(["--convert-labelme", args.convert_labelme[0], args.convert_labelme[1]])
    elif args.detect_bricks:
        command.extend(["--detect-bricks", args.detect_bricks])
    elif args.detect_studs:
        command.extend(["--detect-studs", args.detect_studs])
    elif args.classify_studs:
        command.extend(["--classify-studs", args.classify_studs])
    
    logging.info(f"[INFO] Executing command: {' '.join(command)}")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
