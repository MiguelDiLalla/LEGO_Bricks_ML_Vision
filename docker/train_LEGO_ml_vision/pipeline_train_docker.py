import os
import shutil
import json
import yaml
import logging
import cv2
import optuna
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from tqdm import tqdm
import albumentations as A
import argparse


# === Global Configurations ===
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/data/output")
TRAINING_DIR = os.getenv("TRAINING_DIR", "/mnt/training")
DATASET_PATH = os.getenv("DATASET_PATH", "/app/data/datasets")
EPOCHS = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_cuda_device():
    """
    Detects available CUDA devices and selects the best one.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.info(f"[INFO] CUDA available! {torch.cuda.device_count()} GPUs detected.")
        for i in range(torch.cuda.device_count()):
            logging.info(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("[INFO] No GPU detected, running on CPU.")
    return device

def setup_dataset(custom_images=None, custom_labels=None):
    """
    Ensures the dataset is available inside the container. Uses custom dataset if provided.
    """
    if custom_images and custom_labels:
        logging.info("[INFO] Using custom dataset provided by the user.")
        dataset_yaml = os.path.join(OUTPUT_DIR, "custom_dataset.yaml")
        create_custom_dataset_yaml(custom_images, custom_labels, dataset_yaml)
    else:
        logging.info("[INFO] Checking default dataset availability...")
        datasets = {
            "bricks": "migueldilalla/spiled-lego-bricks",
            "studs": "migueldilalla/labeledstuds-lego-bricks"
        }
        for dataset_name, kaggle_path in datasets.items():
            dataset_dir = os.path.join(DATASET_PATH, dataset_name)
            if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
                logging.info(f"[INFO] Downloading dataset: {dataset_name} from Kaggle...")
                subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_path, "-p", DATASET_PATH, "--unzip"], check=True)
            else:
                logging.info(f"[INFO] Dataset {dataset_name} is already available.")

def setup_pretrained_models(model_path=None, kaggle_model=None):
    """
    Fetches the YOLO model from Kaggle or uses a specified model path.
    """
    if model_path:
        logging.info(f"[INFO] Using custom model provided: {model_path}")
        return model_path
    
    models = {
        "bricks": "migueldilalla/lego_bricks_machinevisonyolofinetune",
        "studs": "migueldilalla/lego_brickstuds_machinevisionyolofinetune"
    }
    os.makedirs(TRAINING_DIR, exist_ok=True)
    
    if kaggle_model and kaggle_model in models:
        model_path = os.path.join(TRAINING_DIR, f"{kaggle_model}.pt")
        if not os.path.exists(model_path):
            logging.info(f"[INFO] Downloading {kaggle_model} model from Kaggle...")
            subprocess.run(["kaggle", "models", "download", "-m", models[kaggle_model], "-p", TRAINING_DIR, "--unzip"], check=True)
        return model_path
    
    return "yolov8n.pt"  # Default model

def train_model(model_path, dataset_yaml):
    """
    Trains the YOLO model.
    """
    logging.info("[INFO] Starting model training...")
    
    if not os.path.exists(dataset_yaml):
        logging.error("[ERROR] dataset.yaml not found. Run dataset setup first.")
        return
    
    model = YOLO(model_path)
    output_dir = os.path.join(TRAINING_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    device = detect_cuda_device()
    
    try:
        model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=640,
            lr0=0.001,
            momentum=0.9,
            project=output_dir,
            name="train",
            device=device
        )
        logging.info(f"[INFO] Training completed. Results saved in {output_dir}.")
    except Exception as e:
        logging.error(f"[ERROR] Training error: {e}")

def create_custom_dataset_yaml(images_path, labels_path, yaml_path):
    """
    Creates a dataset.yaml file for custom datasets.
    """
    dataset_config = {
        "train": images_path,
        "val": images_path,
        "nc": 1,
        "names": ["custom"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    logging.info(f"[INFO] Custom dataset.yaml created at {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model with optional custom datasets and models.")
    parser.add_argument("--custom-images", type=str, help="Path to custom training images")
    parser.add_argument("--custom-labels", type=str, help="Path to custom YOLO labels")
    parser.add_argument("--model-path", type=str, help="Path to a custom YOLO model")
    parser.add_argument("--kaggle-model", type=str, choices=["bricks", "studs"], help="Choose a Kaggle pretrained model")
    args = parser.parse_args()
    
    setup_dataset(args.custom_images, args.custom_labels)
    model_path = setup_pretrained_models(args.model_path, args.kaggle_model)
    dataset_yaml = os.path.join(OUTPUT_DIR, "custom_dataset.yaml") if args.custom_images else os.path.join(OUTPUT_DIR, "dataset", "dataset.yaml")
    train_model(model_path, dataset_yaml)
    logging.info("[INFO] Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

