import os
import logging
import shutil
import sys
import argparse
import torch
import zipfile
import requests
import yaml
import albumentations as A
import cv2
from datetime import datetime
from sklearn.model_selection import train_test_split
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
PREPROCESS_DIR = "/app/data/preprocessed"
MODEL_REPO = "https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision/raw/main/models"

AVAILABLE_MODELS = {
    "base": "yolov8n.pt",
    "bricks": "Brick_Model_best20250123_192838t.pt",
    "studs": "Stud_Model_best20250124_170824.pt"
}

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

def split_dataset(images_path, labels_path, output_di, mode):
    """
    Splits the dataset into train, validation, and test sets, moves them to the corresponding folders,
    and generates the dataset.yaml file for YOLO training.

    Args:
        images_path (str): Path to the images folder.
        labels_path (str): Path to the labels folder.
        output_dir (str): Path to the preprocessed dataset directory.
    """
    train_img_dir = os.path.join(output_dir, "dataset/images/train")
    val_img_dir = os.path.join(output_dir, "dataset/images/val")
    test_img_dir = os.path.join(output_dir, "dataset/images/test")
    train_lbl_dir = os.path.join(output_dir, "dataset/labels/train")
    val_lbl_dir = os.path.join(output_dir, "dataset/labels/val")
    test_lbl_dir = os.path.join(output_dir, "dataset/labels/test")
    
    # Get sorted list of images and corresponding labels
    images = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    labels = sorted([f for f in os.listdir(labels_path) if f.endswith(".txt")])
    
    if len(images) != len(labels):
        logging.error("Image-label count mismatch.")
        sys.exit(1)
    
    image_paths = [os.path.join(images_path, img) for img in images]
    label_paths = [os.path.join(labels_path, lbl) for lbl in labels]
    
    # Train/val/test split (70/20/10)
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(temp_imgs, temp_lbls, test_size=0.33, random_state=42)
    
    # Move files into correct directories
    for img, lbl in zip(train_imgs, train_lbls):
        shutil.move(img, train_img_dir)
        shutil.move(lbl, train_lbl_dir)
    for img, lbl in zip(val_imgs, val_lbls):
        shutil.move(img, val_img_dir)
        shutil.move(lbl, val_lbl_dir)
    for img, lbl in zip(test_imgs, test_lbls):
        shutil.move(img, test_img_dir)
        shutil.move(lbl, test_lbl_dir)
    
    # Create dataset.yaml
    dataset_yaml = {
        "path": output_dir,
        "train": os.path.join(output_dir, "dataset/images/train"),
        "val": os.path.join(output_dir, "dataset/images/val"),
        "nc": 1,  # Adjust according to dataset class count
        "names": [mode[:-1]],  # Adjust class name accordingly
    }
    yaml_path = os.path.join(output_dir, "dataset/dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    logging.info(f"Dataset split complete. YAML saved at: {yaml_path}")
    
    # Optionally remove original dataset directory
    shutil.rmtree(images_path)
    shutil.rmtree(labels_path)
    logging.info("Original dataset folders removed after preprocessing.")

def augment_data(images_path, labels_path, output_dir, num_augmentations=2):
    """
    Applies augmentations to training images and saves the augmented data.
    
    Args:
        images_path (str): Path to the training images folder.
        labels_path (str): Path to the training labels folder.
        output_dir (str): Path to the dataset preprocessing directory.
        num_augmentations (int): Number of augmentations per image (default: 2).
    """
    aug_images_dir = os.path.join(output_dir, "dataset/images/train")
    aug_labels_dir = os.path.join(output_dir, "dataset/labels/train")
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Resize(height=640, width=640),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    images = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    for img_file in images:
        img_path = os.path.join(images_path, img_file)
        label_path = os.path.join(labels_path, img_file.replace(".jpg", ".txt"))
        
        if not os.path.exists(label_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Skipping corrupted image: {img_path}")
            continue
        
        bboxes, class_labels = load_labels(label_path)
        
        for i in range(num_augmentations):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]
            
            aug_image_path = os.path.join(aug_images_dir, f"{img_file.split('.')[0]}_aug{i}.jpg")
            cv2.imwrite(aug_image_path, aug_image)
            
            aug_label_path = os.path.join(aug_labels_dir, f"{img_file.split('.')[0]}_aug{i}.txt")
            save_labels(aug_label_path, aug_bboxes, aug_labels)
    
    logging.info("Data augmentation completed.")

def select_model(model_type):
    """
    Selects and downloads the appropriate model checkpoint.
    
    Args:
        model_type (str): One of 'base', 'bricks', or 'studs'.
    
    Returns:
        str: Path to the selected model file.
    """
    if model_type not in AVAILABLE_MODELS:
        logging.error(f"Invalid model type: {model_type}. Must be one of {list(AVAILABLE_MODELS.keys())}.")
        sys.exit(1)
    
    model_filename = AVAILABLE_MODELS[model_type]
    model_path = os.path.join("/app/data/models", model_filename)
    
    if not os.path.exists(model_path):
        os.makedirs("/app/data/models", exist_ok=True)
        model_url = f"{MODEL_REPO}/{model_filename}"
        logging.info(f"Downloading model: {model_filename} from {model_url}")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info(f"Model downloaded and saved to {model_path}")
        except requests.RequestException as e:
            logging.error(f"Model download failed: {e}")
            sys.exit(1)
    
    return model_path

def train_model(dataset_yaml, output_dir, device, model_type, epochs=20, batch_size=16):
    """
    Trains the YOLO model using the selected dataset and parameters.
    
    Args:
        dataset_yaml (str): Path to the dataset.yaml file.
        output_dir (str): Path to save training outputs.
        device (str): Computation device ('cpu' or '0,1' for GPUs).
        model_type (str): Selected model type ('base', 'bricks', 'studs').
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
    """
    model_path = select_model(model_type)
    model = YOLO(model_path)
    
    logging.info(f"Starting training with model: {model_path}")
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=output_dir,
        name="train"
    )
    
    logging.info(f"Training completed. Results saved in {output_dir}")

def export_logs(log_file=LOG_FILE, output_format="json"):
    """
    Exports logs in the specified format.
    
    Args:
        log_file (str): Path to the log file.
        output_format (str): Format to export ('json' or 'txt').
    
    Returns:
        str: Path to the exported log file.
    """
    if not os.path.exists(log_file):
        logging.warning("No log file found to export.")
        return None
    
    export_filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
    export_path = os.path.join(EXPORT_DIR, export_filename)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    if output_format == "json":
        log_entries = []
        with open(log_file, "r") as f:
            for line in f:
                parts = line.strip().split(" - ", 2)
                if len(parts) == 3:
                    log_entries.append({"timestamp": parts[0], "level": parts[1], "message": parts[2]})
        with open(export_path, "w") as f:
            json.dump(log_entries, f, indent=4)
    else:
        shutil.copy(log_file, export_path)
    
    logging.info(f"Logs exported to {export_path}")
    return export_path

def zip_training_results(training_dir):
    """
    Compresses the training results into a timestamped ZIP file for easy retrieval.
    
    Args:
        training_dir (str): Path to the training results directory.
    
    Returns:
        str: Path to the generated ZIP archive.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"training_results_{timestamp}.zip"
    zip_path = os.path.join(EXPORT_DIR, zip_filename)
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Export logs to JSON before zipping
    log_export_path = export_logs()
    if log_export_path:
        shutil.copy(log_export_path, training_dir)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(training_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, training_dir))
    
    logging.info(f"Training results archived at: {zip_path}")
    return zip_path

def parse_args():
    """Parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="LEGO ML Training Pipeline")
    parser.add_argument("--images", type=str, default=None, help="Path to user-provided image folder")
    parser.add_argument("--labels", type=str, default=None, help="Path to user-provided label folder")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs"], required=True, help="Training mode: 'bricks' or 'studs'")
    parser.add_argument("--model", type=str, choices=["base", "bricks", "studs"], required=True, help="Model type: 'base', 'bricks', or 'studs'")
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
                  os.path.join(DATA_DIR, args.mode, "labels"), dataset_dir, args.mode)
    augment_data(os.path.join(dataset_dir, "dataset/images/train"),
                 os.path.join(dataset_dir, "dataset/labels/train"),
                 dataset_dir)
    
    # 4️⃣ Training Model
    model_output_dir = os.path.join("/app/data/training", datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(model_output_dir, exist_ok=True)
    logging.info("Starting YOLO training...")
    train_model(os.path.join(dataset_dir, "dataset/dataset.yaml"), model_output_dir, device,
                args.model, epochs=args.epochs, batch_size=args.batch_size)
    
    # 5️⃣ Post-Processing
    zip_training_results(model_output_dir)
    logging.info("✅ Pipeline Execution Completed Successfully! ✅")
    
if __name__ == "__main__":
    main()