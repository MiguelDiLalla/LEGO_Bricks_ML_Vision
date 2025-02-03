import argparse
import subprocess
import logging
import os

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="CLI for training YOLO model inside Docker")
    
    # Training parameters
    parser.add_argument("--custom-images", type=str, help="Path to custom training images")
    parser.add_argument("--custom-labels", type=str, help="Path to custom YOLO labels")
    parser.add_argument("--model-path", type=str, help="Path to a custom YOLO model")
    parser.add_argument("--kaggle-model", type=str, choices=["bricks", "studs"], help="Choose a Kaggle pretrained model")
    
    args = parser.parse_args()
    
    # Construct command to execute pipeline_train_docker.py
    command = ["python", "pipeline_train_docker.py"]
    
    if args.custom_images:
        command.extend(["--custom-images", args.custom_images])
    if args.custom_labels:
        command.extend(["--custom-labels", args.custom_labels])
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.kaggle_model:
        command.extend(["--kaggle-model", args.kaggle_model])
    
    logging.info(f"[INFO] Executing command: {' '.join(command)}")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()