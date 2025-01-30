import argparse
import sys
import logging
import os
import yaml
import glob

# Ensure Python can find the scripts folder
sys.path.append("scripts")

from scripts.pipeline_setup import setup_environment, verify_dataset_structure, create_preprocessing_structure, copy_and_partition_data, augment_data, copy_augmented_to_train, create_dataset_yaml, validate_final_structure
from scripts.pipeline_train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_latest_training_session(output_dir="regular_yolo_training"):
    """Finds the most recent training session inside regular_yolo_training/."""
    session_folders = sorted(glob.glob(os.path.join(output_dir, "*")), reverse=True)
    return session_folders[0] if session_folders else None

def run_setup(args):
    """ Handles dataset setup with custom user parameters. """
    
    paths = setup_environment(dataset_name=args.dataset)
    logging.info({"Dataset Paths": paths})

    # If --verify-only is enabled, just check dataset structure and exit
    if args.verify_only:
        verify_dataset_structure(paths["raw_images_path"], paths["raw_labels_path"])
        logging.info("[INFO] Dataset verification complete. No modifications were made.")
        return

    # Create necessary folders
    create_preprocessing_structure(paths["output_path"])

    # Handle custom train-val-test split if provided
    if args.split:
        train_ratio, val_ratio, test_ratio = args.split
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("[ERROR] Split ratios must sum to 1.0")
    else:
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1  # Default split

    copy_and_partition_data(paths["raw_images_path"], paths["raw_labels_path"], paths["output_path"],
                            train_ratio, val_ratio, test_ratio)

    # Handle augmentation
    if not args.no_augment:
        augment_data(
            input_images=os.path.join(paths["output_path"], "dataset/images/train"),
            input_labels=os.path.join(paths["output_path"], "dataset/labels/train"),
            output_dir=os.path.join(paths["output_path"], "augmented_dataset"),
            num_augmentations=args.augment
        )
        copy_augmented_to_train(
            augmented_dir=os.path.join(paths["output_path"], "augmented_dataset"),
            output_path=paths["output_path"]
        )
    else:
        logging.info("[INFO] Data augmentation skipped.")

    # Generate dataset.yaml
    create_dataset_yaml(
        output_path=os.path.join(paths["output_path"], "dataset"),
        num_classes=1,  # Modify if needed
        class_names=["brick"]  # Modify based on dataset
    )

    validate_final_structure(paths["output_path"])
    logging.info("\n[INFO] Dataset setup complete.\n")

def run_train(args):
    """ Handles model training with user-defined parameters. """
    
    dataset_yaml_path = "working/output/dataset/dataset.yaml"

    # Check if dataset.yaml exists
    if not os.path.exists(dataset_yaml_path):
        logging.error("[ERROR] dataset.yaml not found! Run 'cli.py run-setup' first.")
        return

    # Read dataset.yaml to determine dataset being used
    with open(dataset_yaml_path, "r") as f:
        dataset_info = yaml.safe_load(f)
    
    dataset_name = dataset_info.get("names", {}).get(0, "Unknown Dataset")
    logging.info(f"[INFO] Training on dataset: {dataset_name}")
    
    model_path = args.resume if args.resume else args.model

    # Display training parameters
    logging.info(f"[INFO] Training Parameters:")
    logging.info(f" - Model: {model_path}")
    logging.info(f" - Epochs: {args.epochs}")
    logging.info(f" - Batch Size: {args.batch_size}")
    logging.info(f" - Learning Rate: {args.learning_rate}")
    logging.info(f" - Momentum: {args.momentum}")
    logging.info(f" - Image Size: {args.imgsz}")

    # Train YOLO model with given parameters
    train_model(
        dataset_yaml=dataset_yaml_path,
        pretrained_model=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        imgsz=args.imgsz
    )

    # Locate latest training session folder
    latest_session = get_latest_training_session()
    if latest_session:
        weights_path = os.path.join(latest_session, "train/weights/")
        results_path = os.path.join(latest_session, "train/results.png")
        
        logging.info("\n[INFO] Training complete!")
        logging.info(f"[INFO] Model weights saved at: {weights_path}")
        logging.info(f"[INFO] Summary results available at: {results_path}")
    else:
        logging.warning("[WARNING] Could not locate training session folder. Check manually.")

def main():
    parser = argparse.ArgumentParser(description="LEGO Bricks ML CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train command
    train_parser = subparsers.add_parser("run-train", help="Train the YOLO model")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Number of training epochs (default: 50)")
    train_parser.add_argument("--batch-size", type=int, default=16,
                              help="Batch size (default: 16)")
    train_parser.add_argument("--learning-rate", type=float, default=0.001,
                              help="Initial learning rate (default: 0.001)")
    train_parser.add_argument("--momentum", type=float, default=0.9,
                              help="Momentum for optimizer (default: 0.9)")
    train_parser.add_argument("--imgsz", type=int, default=640,
                              help="Image size for YOLO training (default: 640)")
    train_parser.add_argument("--model", type=str, default="yolov8n.pt",
                              help="Path to the pre-trained YOLO model (default: yolov8n.pt)")
    train_parser.add_argument("--resume", type=str, help="Resume training from a checkpoint")
    train_parser.set_defaults(func=run_train)

    # Parse Arguments
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
