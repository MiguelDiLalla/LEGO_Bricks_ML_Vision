import os
import logging
import shutil
import zipfile
import json
import argparse
import random
import subprocess
from datetime import datetime
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from ultralytics import YOLO

#all imports


def setup_logging(log_name="train_session"):
    """
    Configures logging for the training script with immediate flushing.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Immediate console logging
            logging.FileHandler(log_file, mode="a")
        ]
    )

    # Ensure immediate flushing
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)
        handler.flush = lambda: None  # Enforce immediate write

    logging.info(f"Logging initialized: {log_file}")

# Auto-detect repository root

def get_repo_root():
    """
    Auto-detects the repository root directory.
    Returns:
        str: Root directory of the repository.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/" and not os.path.exists(os.path.join(current_dir, ".git")):
        current_dir = os.path.dirname(current_dir)
    return current_dir if os.path.exists(os.path.join(current_dir, ".git")) else os.getcwd()

# Detect hardware availability

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
        return device  # Returning the actual device list like "0,1"
    
    elif torch.backends.mps.is_available():
        logging.info("Detected Apple MPS device.")
        return "mps"
    
    logging.warning("No GPU or MPS device detected. Falling back to CPU.")
    return "cpu"

def setup_execution_structure():
    """
    Ensures all necessary cache directories are created before execution.
    """
    repo_root = get_repo_root()
    required_dirs = [
        "cache/datasets",
        "cache/models",
        "cache/logs",
        "cache/results/TrainingSessions",
    ]
    for directory in required_dirs:
        full_path = os.path.join(repo_root, directory)
        os.makedirs(full_path, exist_ok=True)
    logging.info("✅ Execution structure initialized.")

# Dataset download function

def unzip_dataset(mode, force_extract=False):
    """
    Extracts the dataset from the repository's compressed files.

    Args:
        mode (str): 'bricks' or 'studs'.
        force_extract (bool): If True, forces re-extraction even if dataset exists.
    """
    repo_root = get_repo_root()
    dataset_compressed_dir = os.path.join(repo_root, "presentation/Datasets_Compress")
    dataset_dir = os.path.join(repo_root, "cache/datasets")
    
    dataset_filename = "LegoBricks_Dataset.zip" if mode == "bricks" else "BrickStuds_Dataset.zip"
    dataset_path = os.path.join(dataset_compressed_dir, dataset_filename)
    extract_path = os.path.join(dataset_dir, mode)

    os.makedirs(dataset_dir, exist_ok=True)

    if not force_extract and os.path.exists(extract_path):
        logging.info(f"Dataset already extracted at {extract_path}. Skipping extraction.")
        return extract_path

    logging.info(f"Extracting {dataset_filename}...")
    shutil.unpack_archive(dataset_path, extract_path)
    logging.info(f"Dataset extracted to: {extract_path}")

    return extract_path

# Dataset validation function

def validate_dataset(mode):
    """
    Validates dataset integrity by checking image-label parity and file integrity.

    Args:
        mode (str): 'bricks' or 'studs', defining dataset location.
    """
    repo_root = get_repo_root()
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        logging.error(f"Missing required dataset folders in {dataset_path}.")
        raise FileNotFoundError(f"Missing dataset directories: {images_path} or {labels_path}")

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".txt")])

    if len(image_files) != len(label_files):
        logging.error("Mismatch between number of images and labels.")
        raise ValueError("Image-label count mismatch. Ensure every image has a corresponding label.")

    for img, lbl in zip(image_files, label_files):
        if os.path.splitext(img)[0] != os.path.splitext(lbl)[0]:
            logging.error(f"Mismatched pair: {img} and {lbl}")
            raise ValueError(f"Mismatched dataset files: {img} and {lbl}")

    logging.info(f"✅ Dataset validation successful for mode: {mode}")

# create dataset folder tree structure
def create_dataset_structure(mode):
    """
    Creates necessary dataset directories but does not handle splitting.
    """
    repo_root = get_repo_root()
    output_dir = os.path.join(repo_root, "cache/datasets", f"{mode}_yolo")

    yolo_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test"
    ]
    
    for yolo_dir in yolo_dirs:
        os.makedirs(os.path.join(output_dir, yolo_dir), exist_ok=True)

    logging.info(f"✅ Dataset structure created at {output_dir}")
    return output_dir

    # Helper function to move files
    def move_files(files, img_dst, lbl_dst):
        for f in files:
            shutil.copy(os.path.join(images_path, f), os.path.join(output_dir, img_dst, f))
            shutil.copy(os.path.join(labels_path, f.replace(".jpg", ".txt")), os.path.join(output_dir, lbl_dst, f.replace(".jpg", ".txt")))

    # Move files
    move_files(train_files, "dataset/images/train", "dataset/labels/train")
    move_files(val_files, "dataset/images/val", "dataset/labels/val")
    move_files(test_files, "dataset/images/test", "dataset/labels/test")

    # Create dataset.yaml
    dataset_yaml = {
        "path": output_dir,
        "train": "dataset/images/train",
        "val": "dataset/images/val",
        "test": "dataset/images/test",
        "nc": 1,
        "names": ["lego_brick"] if mode == "bricks" else ["lego_stud"]
    }
    
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logging.info(f"✅ Dataset structure created at {output_dir}")
    return output_dir

#splitting the dataset
def split_dataset(mode):
    """
    Splits dataset into train (70%), val (20%), test (10%) and updates dataset.yaml.
    """
    repo_root = get_repo_root()
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)
    output_dir = os.path.join(repo_root, "cache/datasets", f"{mode}_yolo")

    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    random.shuffle(image_files)

    num_train = int(len(image_files) * 0.7)
    num_val = int(len(image_files) * 0.2)

    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]

    def move_files(files, img_dst, lbl_dst):
        for f in files:
            shutil.copy(os.path.join(images_path, f), os.path.join(output_dir, img_dst, f))
            shutil.copy(os.path.join(labels_path, f.replace(".jpg", ".txt")), os.path.join(output_dir, lbl_dst, f.replace(".jpg", ".txt")))

    move_files(train_files, "dataset/images/train", "dataset/labels/train")
    move_files(val_files, "dataset/images/val", "dataset/labels/val")
    move_files(test_files, "dataset/images/test", "dataset/labels/test")

    dataset_yaml = {
        "path": output_dir,
        "train": "dataset/images/train",
        "val": "dataset/images/val",
        "test": "dataset/images/test",
        "nc": 1,
        "names": ["lego_brick"] if mode == "bricks" else ["lego_stud"]
    }

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logging.info(f"✅ Dataset split completed. Updated dataset.yaml at {output_dir}")
    return output_dir

#augmenting the dataset
def augment_data(dataset_path, augmentations=2):
    """
    Augments training dataset using Albumentations.

    Args:
        dataset_path (str): Path to YOLO dataset.
        augmentations (int): Number of augmentations per image.
    """
    train_images_path = os.path.join(dataset_path, "dataset/images/train")
    train_labels_path = os.path.join(dataset_path, "dataset/labels/train")

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.GaussianBlur(p=0.2),
        A.ColorJitter(p=0.2),
        ToTensorV2()
    ])

    for img_file in os.listdir(train_images_path):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(train_images_path, img_file)
        label_path = os.path.join(train_labels_path, img_file.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(augmentations):
            augmented = augmentation_pipeline(image=image)["image"]
            aug_img_name = img_file.replace(".jpg", f"_aug{i}.jpg")
            aug_img_path = os.path.join(train_images_path, aug_img_name)
            cv2.imwrite(aug_img_path, augmented)

            aug_label_name = label_path.replace(".txt", f"_aug{i}.txt")
            shutil.copy(label_path, aug_label_name)

    logging.info(f"✅ Data augmentation completed with {augmentations} augmentations per image.")

# slecting the model to train
def select_model(mode, use_pretrained=False):
    """
    Selects a pre-trained model from the repository or defaults to YOLOv8n.

    Args:
        mode (str): 'bricks' or 'studs'.
        use_pretrained (bool): If True, selects a LEGO-trained model, else defaults to YOLOv8n.

    Returns:
        str: Path to the selected model checkpoint.
    """
    repo_root = get_repo_root()
    
    if not use_pretrained:
        logging.info("✅ Using default YOLOv8n model.")
        return "yolov8n.pt"
    
    model_dir = os.path.join(repo_root, "presentation/Models_DEMO")
    model_filename = "Brick_Model_best20250123_192838t.pt" if mode == "bricks" else "Stud_Model_best20250124_170824.pt"
    model_path = os.path.join(model_dir, model_filename)
    
    if os.path.exists(model_path):
        logging.info(f"✅ Model selected: {model_path}")
        return model_path
    else:
        logging.error(f"❌ Model not found at {model_path}")
        raise FileNotFoundError(f"Required model file is missing: {model_path}")

# Training the model

def train_model(dataset_path, model_path, device, epochs, batch_size):
    """
    Trains the YOLOv8 model with real-time logging and CLI streaming.
    """
    logging.info(f"🚀 Starting training with model: {model_path}")
    
    model = YOLO(model_path)
    
    # Create command for training with real-time output capture
    command = [
        "yolo",
        "train",
        f"--data={dataset_path}",
        f"--epochs={epochs}",
        f"--batch={batch_size}",
        f"--device={device}",
        "--project=cache/results/TrainingSessions",
        f"--name=training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--exist-ok"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream logs in real-time
    for line in iter(process.stdout.readline, ""):
        logging.info(line.strip())  # Log immediately
        print(line.strip())  # Stream to CLI

    process.stdout.close()
    process.wait()

    logging.info("✅ Training completed.")


#zip results

def zip_training_results(training_dir):
    """
    Compresses training results into a zip file for easy retrieval.
    """
    zip_filename = f"{training_dir}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(training_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, training_dir))
    logging.info(f"✅ Training results compressed into: {zip_filename}")
    return zip_filename

# Export logs

def export_logs(log_file, output_format="json"):
    """
    Exports logs in JSON or TXT format for easy debugging.
    """
    log_path = os.path.join("logs", log_file)
    if not os.path.exists(log_path):
        logging.error("❌ Log file not found.")
        return None
    
    export_path = log_path.replace(".log", f".{output_format}")
    
    if output_format == "json":
        with open(log_path, "r") as f:
            log_entries = [line.strip() for line in f.readlines()]
        with open(export_path, "w") as f:
            json.dump(log_entries, f, indent=4)
    else:
        shutil.copy(log_path, export_path)
    
    logging.info(f"✅ Logs exported to {export_path}")
    return export_path

# Parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for the training pipeline.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="LEGO ML Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs"], required=True, help="Training mode: 'bricks' or 'studs'")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--zip-results", action="store_true", help="Compress training results after completion")
    parser.add_argument("--cleanup", action="store_true", help="Remove cached datasets after training")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of dataset")
    parser.add_argument("--use-pretrained", action="store_true", help="Use LEGO-trained model instead of YOLOv8n")
    return parser.parse_args()

# Main execution

def main():
    """
    Main execution pipeline for training setup and initialization.
    """
    # Setup logging to capture all events
    setup_logging()
    
    # Parse command-line arguments
    args = parse_args()
    
    logging.info("=== LEGO ML Training Pipeline Starting ===")
    
    # Detect available hardware (GPU, MPS, or CPU)
    device = detect_hardware()
    logging.info(f"Using device: {device}")
    
    # Setup necessary directory structure for execution
    setup_execution_structure()
    
    # Unzip and prepare the dataset
    dataset_path = unzip_dataset(args.mode, args.force_extract)
    logging.info(f"Dataset ready at: {dataset_path}")
    
    # Validate the dataset to ensure integrity
    validate_dataset(args.mode)
    
    # Split dataset into training, validation, and test sets
    dataset_split_path = split_dataset(args.mode)
    logging.info(f"Dataset split and saved at: {dataset_split_path}")
    
    # Apply data augmentation to the training dataset
    augment_data(dataset_split_path)
    logging.info(f"Augmentation applied to training dataset: {dataset_split_path}")
    
    # Select the model to use for training (pre-trained or default YOLOv8n)
    model_path = select_model(args.mode, args.use_pretrained)
    logging.info(f"Using model: {model_path}")
    
    # Train the model with the prepared dataset
    train_model(dataset_split_path, model_path, device, args.epochs, args.batch_size)
    
    logging.info("✅ Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
