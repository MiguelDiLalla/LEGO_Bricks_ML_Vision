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
    subprocess.run(command)

def predict_brick(args):
    """Runs inference on an image."""
    logging.info(f"Running inference on {args.image}")
    command = [
        "python3", "pipeline_utils.py",
        "--image", args.image
    ]
    if args.save_annotated:
        command.append("--save-annotated")
    subprocess.run(command)

def cleanup_cache():
    """Removes cached datasets and models."""
    cache_dirs = ["cache", "models", "data"]
    for folder in cache_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logging.info(f"Deleted {folder}")
    logging.info("Cache cleanup complete.")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="LEGO ML CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--mode", required=True, choices=["bricks", "studs"], help="Training mode")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.set_defaults(func=train_model)

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Run inference on an image")
    predict_parser.add_argument("--image", required=True, help="Path to image file")
    predict_parser.add_argument("--save-annotated", action="store_true", help="Save annotated results")
    predict_parser.set_defaults(func=predict_brick)

    # Cleanup Command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clear cached files")
    cleanup_parser.set_defaults(func=lambda args: cleanup_cache())

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
