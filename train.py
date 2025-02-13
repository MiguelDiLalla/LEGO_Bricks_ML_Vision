import os
import logging
import torch
import shutil
import zipfile
import json
import random
import argparse
from datetime import datetime
import subprocess
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from ultralytics import YOLO
from pathlib import Path


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
    logging.info(f"Logging initialized: {log_file}")

def cleanup_after_training(dataset_path, dataset_yolo_path):
    """
    Cleans up temporary data after training.
    """
    logging.info("🧹 Cleaning up temporary files...")
    # Remove extracted dataset
    if Path(dataset_path).exists():
        shutil.rmtree(dataset_path)
        logging.info(f"Deleted dataset from: {dataset_path}")

    # Remove augmented images
    train_images_path = Path(dataset_yolo_path) / "dataset" / "images" / "train"
    if train_images_path.exists():
        for img_file in train_images_path.iterdir():
            if img_file.suffix in [".jpg", ".png"]:
                img_file.unlink()
        logging.info(f"Deleted augmented images from: {train_images_path}")

    # Remove intermediate YOLO artifacts
    yolo_cache_path = Path(dataset_yolo_path) / "dataset"
    if yolo_cache_path.exists():
        shutil.rmtree(yolo_cache_path)
    logging.info("✅ Cleanup complete.")


def get_repo_root() -> Path:
    """
    Auto-detects the repository root directory.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()

def detect_hardware():
    """
    Detects available hardware for training.
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
def zip_training_results(training_dir, output_dir):
    """
    Compresses training results into a zip file for easy retrieval.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = os.path.join(output_dir, f"training_results_{timestamp}.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(training_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, training_dir))
    logging.info(f"✅ Training results compressed into: {zip_filename}")
    return zip_filename

def export_logs(log_name="train_session"):
    """
    Exports logs and hardware details in JSON format.
    """
    log_path = os.path.join("logs", f"{log_name}.log")
    export_path = log_path.replace(".log", ".json")
    
    hardware_info = {
        "python_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_gpus": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(log_path, "r") as f:
        log_entries = [line.strip() for line in f.readlines()]
    
    session_data = {
        "hardware_info": hardware_info,
        "logs": log_entries
    }
    
    with open(export_path, "w") as f:
        json.dump(session_data, f, indent=4)
    
    logging.info(f"✅ Logs exported to {export_path}")
    return export_path

def setup_execution_structure():
    """
    Ensures all necessary cache directories are created before execution.
    """
    repo_root = get_repo_root()
    required_dirs = [
        repo_root / "cache" / "datasets",
        repo_root / "cache" / "models",
        repo_root / "cache" / "logs",
        repo_root / "cache" / "results" / "TrainingSessions",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
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
    Validates dataset integrity by dynamically detecting the images and labels folders,
    ensuring image-label parity and file integrity.

    Args:
        mode (str): 'bricks' or 'studs', defining dataset location.
    """
    repo_root = get_repo_root()
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)

    # Detect folders
    subfolders = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    # Identify images and labels based on dominant file extensions
    images_path, labels_path = None, None
    for folder in subfolders:
        files = os.listdir(folder)
        jpg_count = sum(f.endswith('.jpg') for f in files)
        txt_count = sum(f.endswith('.txt') for f in files)

        if jpg_count > txt_count:
            images_path = folder
        elif txt_count > jpg_count:
            labels_path = folder

    # If paths are missing, raise an error
    if images_path is None or labels_path is None:
        logging.error(f"Dataset structure invalid. Could not identify images and labels in {dataset_path}.")
        raise FileNotFoundError(f"Could not determine image/label folders in: {dataset_path}")

    # Rename folders to standard structure if needed
    expected_images_path = os.path.join(dataset_path, "images")
    expected_labels_path = os.path.join(dataset_path, "labels")

    if images_path != expected_images_path:
        os.rename(images_path, expected_images_path)
        logging.info(f"Renamed {images_path} -> {expected_images_path}")

    if labels_path != expected_labels_path:
        os.rename(labels_path, expected_labels_path)
        logging.info(f"Renamed {labels_path} -> {expected_labels_path}")

    # Validate dataset integrity
    image_files = sorted([f for f in os.listdir(expected_images_path) if f.endswith(".jpg")])
    label_files = sorted([f for f in os.listdir(expected_labels_path) if f.endswith(".txt")])

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
    Creates necessary dataset directories for YOLO.
    """
    repo_root = get_repo_root()
    output_dir = repo_root / "cache" / "datasets" / f"{mode}_yolo"
    yolo_dirs = [
        output_dir / "dataset" / "images" / "train",
        output_dir / "dataset" / "images" / "val",
        output_dir / "dataset" / "images" / "test",
        output_dir / "dataset" / "labels" / "train",
        output_dir / "dataset" / "labels" / "val",
        output_dir / "dataset" / "labels" / "test"
    ]
    
    for yolo_dir in yolo_dirs:
        yolo_dir.mkdir(parents=True, exist_ok=True)
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

    # Ensure destination folders exist
    def ensure_folder_exists(path):
        os.makedirs(path, exist_ok=True)

    ensure_folder_exists(os.path.join(output_dir, "dataset/images/train"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/images/val"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/images/test"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/train"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/val"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/test"))

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
        A.ColorJitter(p=0.2)
    ])

    for img_file in os.listdir(train_images_path):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(train_images_path, img_file)
        label_path = os.path.join(train_labels_path, img_file.replace(".jpg", ".txt"))

        # ✅ Check if image exists
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Skipping {img_file}: Unable to read image file.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(augmentations):
            augmented = augmentation_pipeline(image=image)["image"]

            # ✅ Ensure augmented image is NumPy array
            if isinstance(augmented, torch.Tensor):
                augmented = augmented.permute(1, 2, 0).cpu().numpy()
                augmented = (augmented * 255).astype(np.uint8)

            aug_img_name = img_file.replace(".jpg", f"_aug{i}.jpg")
            aug_img_path = os.path.join(train_images_path, aug_img_name)
            cv2.imwrite(aug_img_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

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

def save_model(model, output_dir, model_name="trained_model.pt"):
    """
    Saves the trained model to the specified directory.
    """
    model_save_path = os.path.join(output_dir, model_name)
    model.save(model_save_path)
    logging.info(f"✅ Model saved to: {model_save_path}")
    return model_save_path

def train_model(dataset_path, model_path, device, epochs, batch_size, output_dir):
    """
    Trains the YOLOv8 model with real-time logging and CLI streaming.
    """
    logging.info(f"🚀 Starting training with model: {model_path}")
    
    model = YOLO(model_path)
    training_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    command = [
        "yolo",
        "train",
        f"data={dataset_path}/dataset.yaml",
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"device={device}",
        f"project={output_dir}",  # ✅ Ensure YOLO saves results to the right place
        f"name={training_name}",
        #early stopping
        "patience=3",
        "exist_ok=True"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in iter(process.stdout.readline, ""):
        logging.info(line.strip())  # ✅ Log everything
        print(line.strip())  # ✅ Print to CLI for visibility

    process.stdout.close()
    process.wait()
    logging.info("✅ Training completed.")

    # Save the trained model
    save_model(model, results_dir)

    return results_dir  # ✅ Return path to the saved results
#zip results

def zip_training_results(training_dir):
    """
    Compresses the content of training_dir into a zip file named results_[timestamp].zip,
    logs a download link, and returns the zip filename.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = os.path.join(os.path.dirname(training_dir), f"results_{timestamp}.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(training_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, training_dir))
    abs_zip_path = os.path.abspath(zip_filename)
    download_link = f"file://{abs_zip_path}"
    logging.info(f"✅ Training results compressed into: {zip_filename}")
    logging.info(f"Download link: {download_link}")
    print(f"\nDownload link for results: {download_link}\n")
    return zip_filename

# Export logs

def export_logs(log_name="train_session", output_format="json"):
    """
    Exports logs in JSON or TXT format for easy debugging.

    Args:
        log_name (str): The base name for the log file (without extension).
        output_format (str): Format to save logs (default: JSON).
    """
    log_path = os.path.join("logs", f"{log_name}.log")
    if not os.path.exists(log_path):
        logging.error(f"❌ Log file not found: {log_path}")
        return None

    repo_root = get_repo_root()
    export_dir = os.path.join(repo_root, "cache", "results")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{log_name}.{output_format}")

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
    # By default, zip_results is enabled. Users can disable it by using --no-zip-results.
    parser.add_argument("--zip-results", dest="zip_results", action="store_true", default=True, help="Compress training results after completion (default: enabled)")
    parser.add_argument("--no-zip-results", dest="zip_results", action="store_false", help="Disable compressing training results")
    parser.add_argument("--cleanup", action="store_true", help="Remove cached datasets after training")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of dataset")
    parser.add_argument("--use-pretrained", action="store_true", help="Use LEGO-trained model instead of YOLOv8n")
    return parser.parse_args()

# Main execution

def main():
    setup_logging()
    args = parse_args()
    
    logging.info("=== LEGO ML Training Pipeline Starting ===")
    device = detect_hardware()
    logging.info(f"Using device: {device}")
    
    # Optional: Validate batch size if GPUs are available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.batch_size % num_gpus != 0:
            raise ValueError(f"Batch size ({args.batch_size}) must be a multiple of GPU count ({num_gpus}). "
                             f"Try --batch-size {num_gpus if num_gpus else 'appropriate_multiple'}.")
    
    setup_execution_structure()
    dataset_path = unzip_dataset(args.mode, args.force_extract)
    logging.info(f"Dataset ready at: {dataset_path}")
    
    validate_dataset(args.mode)
    
    # ✅ FIX: Ensure dataset structure exists before splitting
    create_dataset_structure(args.mode)  
    
    dataset_yolo_path = split_dataset(args.mode)
    logging.info(f"Dataset split and saved at: {dataset_yolo_path}")
    
    augment_data(dataset_yolo_path)
    logging.info(f"Augmentation applied to training dataset: {dataset_yolo_path}")
    
    model_path = select_model(args.mode, args.use_pretrained)
    logging.info(f"Using model: {model_path}")
    
    output_dir = os.path.join(get_repo_root(), "cache", "results")
    os.makedirs(output_dir, exist_ok=True)
    train_model(dataset_yolo_path, model_path, device, args.epochs, args.batch_size, output_dir)
    
    # Export logs once with correct parameter name
    export_logs(log_name="train_session", output_format="json")

    if args.zip_results:
        zip_training_results(output_dir, output_dir)
    

    if args.cleanup:
        cleanup_after_training(dataset_path, dataset_yolo_path)
    
    logging.info("✅ Training pipeline completed successfully.")


if __name__ == "__main__":
    main()