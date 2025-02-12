import argparse
import subprocess
import logging
import os
import shutil

def setup_logging():
    """Configures logging for the CLI."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "cli.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a")
        ]
    )

def train_model(args):
    """Executes the training pipeline."""
    logging.info("Starting training...")
    command = [
        "python3", "train.py",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size)
    ]
    
    if args.use_pretrained:
        command.append("--use-pretrained")
    if args.cleanup:
        command.append("--cleanup")
    if args.zip_results:
        command.append("--zip-results")
    
    subprocess.run(command)

def predict_brick(args):
    """Runs inference using model_utils.py."""
    logging.info(f"Running inference on {args.image}")
    command = [
        "python3", "model_utils.py",
        "--image", args.image
    ]
    if args.save_annotated:
        command.append("--save-annotated")
    if args.output:
        command.extend(["--output", args.output])
    subprocess.run(command)

def cleanup_cache():
    """Removes cached datasets and models with user confirmation."""
    cache_dirs = ["cache/datasets", "cache/models", "cache/results", "cache/logs"]
    logging.info("⚠️ WARNING: This will delete cached datasets and models.")
    confirm = input("Are you sure? (y/N): ").strip().lower()
    if confirm == 'y':
        for folder in cache_dirs:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logging.info(f"Deleted {folder}")
        logging.info("✅ Cache cleanup complete.")
    else:
        logging.info("Cache cleanup aborted.")

def process_data(args):
    """Handles data processing commands."""
    command = ["python3", "data_utils.py", f"--{args.operation}", "--input", args.input, "--output", args.output]
    if args.operation == "keypoints-to-bboxes" and args.area_ratio:
        command.extend(["--area-ratio", str(args.area_ratio)])
    logging.info(f"Executing data processing: {args.operation}")
    subprocess.run(command)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="LEGO ML CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--mode", required=True, choices=["bricks", "studs"], help="Training mode")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--use-pretrained", action="store_true", help="Use LEGO-trained model instead of YOLOv8n")
    train_parser.add_argument("--cleanup", action="store_true", help="Remove cached datasets after training")
    train_parser.add_argument("--zip-results", action="store_true", help="Compress training results after completion")
    train_parser.set_defaults(func=train_model)

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Run inference on an image")
    predict_parser.add_argument("--image", required=True, help="Path to image file")
    predict_parser.add_argument("--save-annotated", action="store_true", help="Save annotated results")
    predict_parser.add_argument("--output", required=False, help="Output folder for predictions")
    predict_parser.set_defaults(func=predict_brick)

    # Cleanup Command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clear cached files")
    cleanup_parser.set_defaults(func=lambda args: cleanup_cache())
    
    # Data Processing Command
    data_parser = subparsers.add_parser("data-processing", help="Perform dataset processing")
    data_parser.add_argument("operation", choices=["labelme-to-yolo", "keypoints-to-bboxes", "visualize"], help="Data operation")
    data_parser.add_argument("--input", required=True, help="Input folder path")
    data_parser.add_argument("--output", required=True, help="Output folder path")
    data_parser.add_argument("--area-ratio", type=float, help="Total area ratio for bounding boxes (only for keypoints-to-bboxes)")
    data_parser.set_defaults(func=process_data)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
