import argparse
import sys
import logging
import os
import yaml
import glob
import time
import shutil

# Ensure Python can find the scripts folder
sys.path.append("scripts")

from scripts.pipeline_setup import setup_environment, verify_dataset_structure, create_preprocessing_structure, copy_and_partition_data, augment_data, copy_augmented_to_train, create_dataset_yaml, validate_final_structure
from scripts.pipeline_train import train_model, pre_training_cleanup, get_device
from scripts.pipeline_utils import labelme_to_yolo, convert_keypoints_json, visualize_yolo_annotations, run_inference



# Configure logging
log_filename = "cli_execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"[INFO] CLI execution started. Logs will be saved in {log_filename}.")

def get_latest_training_session(output_dir="regular_yolo_training"):
    """Finds the most recent training session inside regular_yolo_training/."""
    session_folders = sorted(glob.glob(os.path.join(output_dir, "*")), reverse=True)
    return session_folders[0] if session_folders else None

def run_setup(args):
    """Handles dataset setup with custom user parameters."""
    
    start_time = time.time()
    logging.info("[INFO] Starting dataset setup...")
    paths = setup_environment(dataset_name=args.dataset)
    logging.info("[INFO] Environment setup completed. Paths:")
    logging.info(paths)

    if args.verify_only:
        logging.info("[INFO] Verifying dataset structure...")
        verify_dataset_structure(paths["raw_images_path"], paths["raw_labels_path"])
        logging.info("[INFO] Dataset verification complete. No modifications were made.")
        return

    create_preprocessing_structure(paths["output_path"])

    if args.split:
        train_ratio, val_ratio, test_ratio = args.split
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("[ERROR] Split ratios must sum to 1.0")
    else:
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    copy_and_partition_data(paths["raw_images_path"], paths["raw_labels_path"], paths["output_path"],
                            train_ratio, val_ratio, test_ratio)

    if not args.no_augment:
        logging.info("[INFO] Applying data augmentation...")
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

    create_dataset_yaml(
        output_path=os.path.join(paths["output_path"], "dataset"),
        num_classes=1,
        class_names=["brick"]
    )

    validate_final_structure(paths["output_path"])
    logging.info("\n[INFO] Dataset setup complete. Ready for training.\n")
    end_time = time.time()
    logging.info(f"[INFO] Execution completed in {end_time - start_time:.2f} seconds.")

def run_train(args):
    """Handles model training with user-defined parameters."""
    
    start_time = time.time()
    dataset_yaml_path = "working/output/dataset/dataset.yaml"

    if not os.path.exists(dataset_yaml_path):
        logging.error("[ERROR] dataset.yaml not found! Run 'cli.py run-setup' first.")
        return

    # Cleanup old training results before starting
    pre_training_cleanup()

    with open(dataset_yaml_path, "r") as f:
        dataset_info = yaml.safe_load(f)
    
    dataset_name = dataset_info.get("names", {}).get(0, "Unknown Dataset")
    logging.info(f"[INFO] Training on dataset: {dataset_name}")
    
    model_path = args.resume if args.resume else args.model

    # Detect and log the device
    device = args.device if args.device else get_device()
    logging.info(f"[INFO] Using device: {device}")

    logging.info(f"[INFO] Training Parameters:")
    logging.info(f" - Model: {model_path}")
    logging.info(f" - Epochs: {args.epochs}")
    logging.info(f" - Batch Size: {args.batch_size}")
    logging.info(f" - Learning Rate: {args.learning_rate}")
    logging.info(f" - Momentum: {args.momentum}")
    logging.info(f" - Image Size: {args.imgsz}")

    train_model(
        dataset_yaml=dataset_yaml_path,
        pretrained_model=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        imgsz=args.imgsz,
        device=device
    )

    latest_session = get_latest_training_session()
    if latest_session:
        weights_path = os.path.join(latest_session, "train/weights/")
        results_path = os.path.join(latest_session, "train/results.png")
        
        logging.info("\n[INFO] Training complete!")
        logging.info(f"[INFO] Model weights saved at: {weights_path}")
        logging.info(f"[INFO] Summary results available at: {results_path}")
    else:
        logging.warning("[WARNING] Could not locate training session folder. Check manually.")
    
    end_time = time.time()
    logging.info(f"[INFO] Execution completed in {end_time - start_time:.2f} seconds.")

def run_utils(args):
    """Handles utility functions."""
    
    start_time = time.time()
    if args.convert_labels:
        if not args.input_folder or not args.output_folder:
            logging.error("[ERROR] Input and output folders are required for label conversion.")
            return
        logging.info(f"[INFO] Converting labels from {args.input_folder} to {args.output_folder}")
        labelme_to_yolo(args.input_folder, args.output_folder)

    elif args.convert_keypoints:
        if not args.input_folder or not args.output_folder:
            logging.error("[ERROR] Input and output folders are required for keypoint conversion.")
            return
        logging.info(f"[INFO] Converting keypoints from {args.input_folder} to {args.output_folder}")
        convert_keypoints_json(args.input_folder, args.output_folder)

    elif args.visualize_annotations:
        if not args.input_folder or not args.output_folder:
            logging.error("[ERROR] Input folder and output folder are required for annotation visualization.")
            return
        logging.info(f"[INFO] Visualizing annotations from {args.input_folder}")
        visualize_yolo_annotations(args.input_folder, args.output_folder, args.mode)

    elif args.infer:
        if not args.input_folder or not args.output_folder or not args.model:
            logging.error("[ERROR] Model, input, and output folders are required for inference.")
            return
        logging.info(f"[INFO] Running inference using {args.model} on {args.input_folder}")
        run_inference(args.model, args.input_folder, args.output_folder)
import shutil

def run_export_results(args):
    """Finds and exports the latest training results as a .zip file."""
    
    export_dir = "/app/data/exports/"
    
    if not os.path.exists(export_dir) or not os.listdir(export_dir):
        logging.error("[ERROR] No training results found in the export directory.")
        return
    
    # Find the latest zip archive
    zip_files = sorted(glob.glob(os.path.join(export_dir, "*.zip")), reverse=True)
    latest_zip = zip_files[0] if zip_files else None
    
    if not latest_zip:
        logging.error("[ERROR] No zip archives found in the export directory.")
        return
    
    logging.info(f"[INFO] Latest training archive found: {latest_zip}")

    # Copy the archive to user-specified location or provide Docker instructions
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        shutil.copy(latest_zip, args.output)
        logging.info(f"[INFO] Archive copied to: {args.output}")
    else:
        logging.info("[INFO] To retrieve the file in Docker, use:")
        logging.info(f"docker cp <container_id>:{latest_zip} <destination_path>")


def main():
    parser = argparse.ArgumentParser(description="LEGO Bricks ML CLI")
    parser.add_argument("--version", action="store_true", help="Display CLI version.")
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train command (define before usage)
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
    train_parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                            help="Specify device for training ('cpu' or 'cuda'). If not set, auto-detect.")
    train_parser.set_defaults(func=run_train)

    # Export command (comes after train_parser is defined)
    export_parser = subparsers.add_parser("export-results", help="Export latest training results as a .zip file")
    export_parser.add_argument("--output", type=str, help="Specify a destination folder to copy the exported results.")
    export_parser.set_defaults(func=run_export_results)

    # Utilities command
    utils_parser = subparsers.add_parser("run-utils", help="Execute utility functions")

    export_parser = subparsers.add_parser("export-results", help="Export latest training results as a .zip file")
    export_parser.add_argument("--output", type=str, help="Specify a destination folder to copy the exported results.")
    export_parser.set_defaults(func=run_export_results)

    utils_parser.add_argument("--convert-labels", action="store_true",
                              help="Convert LabelMe JSON to YOLO format")
    utils_parser.add_argument("--convert-keypoints", action="store_true",
                              help="Convert keypoint JSON annotations to YOLO format")
    utils_parser.add_argument("--visualize-annotations", action="store_true",
                              help="Visualize YOLO annotations on images")
    utils_parser.add_argument("--infer", action="store_true",
                              help="Run YOLO model inference on a folder of images")
    
    utils_parser.add_argument("--input-folder", type=str, required=False,
                              help="Input folder path")
    utils_parser.add_argument("--output-folder", type=str, required=False,
                              help="Output folder path")
    utils_parser.add_argument("--model", type=str, required=False,
                              help="Path to trained YOLO model (required for inference)")
    utils_parser.add_argument("--mode", type=int, default=1,
                              help="Mode for visualization: (1=Single image, 2=Save single, 3=Grid, 4=Save all)")
    utils_parser.add_argument("--dry-run", action="store_true",
                              help="Preview what the command will do without executing it.")
    train_parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                              help="Specify device for training ('cpu' or 'cuda'). If not set, auto-detect.")

    
    utils_parser.set_defaults(func=run_utils)

    # Parse Arguments
    args = parser.parse_args()

    # Detect and log the training device
    device = args.device if args.device else get_device()
    logging.info(f"[INFO] Using device: {device}")
    
    if args.version:
        print("LEGO Bricks ML CLI v1.0.0")
        return
    
    if hasattr(args, "func"):
        if args.dry_run:
            logging.info("[DRY-RUN] This is a test run. No changes will be made.")
        else:
            args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
