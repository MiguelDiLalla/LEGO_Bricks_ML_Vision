import os
import logging
import shutil
import sys
import argparse
import torch
import zipfile
import requests
from datetime import datetime
from ultralytics import YOLO

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print logs to console
        logging.FileHandler("/app/data/pipeline_log.txt", mode="w")  # Save logs inside container
    ]
)

GITHUB_REPO = "https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision/raw/main/presentation/Datasets_Compress"
DATA_DIR = "/app/data/datasets"

def detect_hardware():
    """
    Detects available hardware for training.
    
    Returns:
        str: Device identifier ('0', '0,1', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = ",".join(str(i) for i in range(num_gpus))
        logging.info(f"Detected {num_gpus} GPU(s): {device}")
        return device
    
    elif torch.backends.mps.is_available():
        logging.info("Detected Apple MPS device.")
        return "mps"
    
    logging.warning("No GPU or MPS device detected. Falling back to CPU.")
    return "cpu"

def download_dataset(mode):
    """
    Downloads and extracts the selected dataset from GitHub.
    
    Args:
        mode (str): 'bricks' or 'studs', determining which dataset to fetch.
    
    Returns:
        tuple: (images_path, labels_path) for further processing.
    """
    dataset_filename = "LegoBricks_Dataset.zip" if mode == "bricks" else "BrickStuds_Dataset.zip"
    dataset_url = f"{GITHUB_REPO}/{dataset_filename}"
    dataset_path = os.path.join(DATA_DIR, dataset_filename)
    extract_path = os.path.join(DATA_DIR, mode)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info(f"Downloading {dataset_filename} from GitHub...")
    
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open(dataset_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logging.info(f"Dataset downloaded: {dataset_path}")
    except requests.RequestException as e:
        logging.error(f"Dataset download failed: {e}")
        sys.exit(1)
    
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    logging.info(f"Dataset extracted to: {extract_path}")
    
    # Detect image and label folders
    images_path, labels_path = None, None
    for root, dirs, files in os.walk(extract_path):
        if any(f.endswith('.jpg') for f in files):
            images_path = root
        elif any(f.endswith('.txt') for f in files):
            labels_path = root
    
    if not images_path or not labels_path:
        logging.error("Failed to locate images or labels in extracted dataset.")
        sys.exit(1)
    
    logging.info(f"Dataset ready: Images -> {images_path}, Labels -> {labels_path}")
    return images_path, labels_path

def create_dataset_structure():
    """
    Creates the necessary folder structure for dataset preprocessing.
    
    Returns:
        str: Path to the dataset preprocessing directory.
    """
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    subfolders = [
        "dataset/images/train", "dataset/images/val", "dataset/images/test",
        "dataset/labels/train", "dataset/labels/val", "dataset/labels/test"
    ]
    
    for subfolder in subfolders:
        os.makedirs(os.path.join(PREPROCESS_DIR, subfolder), exist_ok=True)
    
    logging.info(f"Dataset structure created at: {PREPROCESS_DIR}")
    return PREPROCESS_DIR

def validate_dataset(mode):
    """
    Validates dataset integrity by checking image-label parity and file integrity.
    
    Args:
        mode (str): 'bricks' or 'studs', defining dataset location.
    """
    dataset_path = os.path.join(DATA_DIR, mode)
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        logging.error(f"Missing required dataset folders in {dataset_path}.")
        sys.exit(1)
    
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".txt")])
    
    if len(image_files) != len(label_files):
        logging.error("Image-label count mismatch. Ensure every image has a corresponding label.")
        sys.exit(1)
    
    for img, lbl in zip(image_files, label_files):
        if os.path.splitext(img)[0] != os.path.splitext(lbl)[0]:
            logging.error(f"Mismatched pair: {img} and {lbl}")
            sys.exit(1)
    
    logging.info(f"Dataset validation successful for mode: {mode}")

def copy_user_dataset(images_path, labels_path, mode):
    """
    Copies user-provided dataset into container's internal dataset folder.
    
    Args:
        images_path (str): Path to user-provided image folder.
        labels_path (str): Path to user-provided label folder.
        mode (str): 'bricks' or 'studs' to define destination folder.
    """
    target_path = os.path.join(DATA_DIR, mode)
    os.makedirs(target_path, exist_ok=True)
    
    img_target = os.path.join(target_path, "images")
    lbl_target = os.path.join(target_path, "labels")
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(lbl_target, exist_ok=True)
    
    for file in os.listdir(images_path):
        shutil.copy(os.path.join(images_path, file), img_target)
    for file in os.listdir(labels_path):
        shutil.copy(os.path.join(labels_path, file), lbl_target)
    
    logging.info(f"Dataset copied to: {target_path}")



def parse_args():
    """Parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="LEGO ML Training Pipeline")
    parser.add_argument("--images", type=str, default=None, help="Path to user-provided image folder")
    parser.add_argument("--labels", type=str, default=None, help="Path to user-provided label folder")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs"], required=True, help="Training mode: 'bricks' or 'studs'")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    return parser.parse_args()

def main():
    """
    Main execution pipeline for dataset preparation and model training.
    """
    args = parse_args()
    logging.info("=== LEGO ML Pipeline Starting ===")
    
    # 1️⃣ Detect hardware
    device = detect_hardware()
    if device == "cpu":
        logging.error("No GPU detected. Training cannot proceed.")
        sys.exit(1)
    logging.info(f"Using device: {device}")
    
    # 2️⃣ Dataset Handling
    if args.images and args.labels:
        logging.info("Using user-provided dataset.")
        copy_user_dataset(args.images, args.labels, args.mode)
    else:
        logging.info("Downloading dataset from GitHub repository...")
        download_dataset(args.mode)
    
    # Validate dataset integrity
    validate_dataset(args.mode)
    
    # 3️⃣ Preprocessing & Augmentations
    dataset_dir = create_dataset_structure()
    split_dataset(os.path.join(DATA_DIR, args.mode, "images"),
                  os.path.join(DATA_DIR, args.mode, "labels"), dataset_dir)
    augment_data(os.path.join(dataset_dir, "dataset/images/train"),
                 os.path.join(dataset_dir, "dataset/labels/train"),
                 dataset_dir)
    
    # 4️⃣ Training Model
    model_output_dir = os.path.join("/app/data/training", datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(model_output_dir, exist_ok=True)
    logging.info("Starting YOLO training...")
    train_model(os.path.join(dataset_dir, "dataset/dataset.yaml"), model_output_dir, device,
                epochs=args.epochs, batch_size=args.batch_size)
    
    # 5️⃣ Post-Processing
    zip_training_results(model_output_dir)
    logging.info("✅ Pipeline Execution Completed Successfully! ✅")
    
if __name__ == "__main__":
    main()
