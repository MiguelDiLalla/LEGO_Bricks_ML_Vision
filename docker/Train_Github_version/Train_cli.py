import argparse
import subprocess
import logging

def main():
    parser = argparse.ArgumentParser(description="LEGO ML Training Pipeline CLI")
    
    # CLI Arguments
    parser.add_argument("--images", type=str, default=None, help="Path to user-provided image folder")
    parser.add_argument("--labels", type=str, default=None, help="Path to user-provided label folder")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs"], required=True, help="Training mode: 'bricks' or 'studs'")
    parser.add_argument("--model", type=str, choices=["base", "bricks", "studs"], required=True, help="Model type: 'base', 'bricks', or 'studs'")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without running any process")
    
    args = parser.parse_args()
    
    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.dry_run:
        logging.info("=== DRY RUN MODE ENABLED ===")
        logging.info(f"Training Mode: {args.mode}")
        logging.info(f"Model: {args.model}")
        logging.info(f"Epochs: {args.epochs}")
        logging.info(f"Batch Size: {args.batch_size}")
        if args.images and args.labels:
            logging.info(f"Using User Dataset - Images: {args.images}, Labels: {args.labels}")
        else:
            logging.info("Using default dataset from GitHub repository.")
        logging.info("Dry run completed. No actual training executed.")
        return
    
    # Construct command to run the pipeline script
    command = [
        "python3", "Train_models_docker.py",
        "--mode", args.mode,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size)
    ]
    
    if args.images and args.labels:
        command.extend(["--images", args.images, "--labels", args.labels])
    
    # Run the pipeline script
    subprocess.run(command)
    
if __name__ == "__main__":
    main()
