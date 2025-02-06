import os
import logging
import shutil
import sys
import argparse
import torch
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
        images_path, labels_path = args.images, args.labels
    else:
        logging.info("Downloading dataset from GitHub repository...")
        images_path, labels_path = download_dataset(args.mode)
    
    # Validate dataset integrity
    validate_dataset(images_path, labels_path)
    
    # 3️⃣ Preprocessing & Augmentations
    dataset_dir = create_dataset_structure()
    split_dataset(images_path, labels_path, dataset_dir)
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
