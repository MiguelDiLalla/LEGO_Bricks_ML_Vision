import click
import subprocess
import logging
import os
import shutil
import sys
from train import get_repo_root

class EmojiFormatter(logging.Formatter):
    def format(self, record):
        base_msg = super().format(record)
        if record.levelno >= logging.ERROR:
            emoji = "❌"
        elif record.levelno >= logging.WARNING:
            emoji = "⚠️"
        elif record.levelno >= logging.INFO:
            emoji = "✅"
        else:
            emoji = "💬"
        return f"{base_msg} {emoji}"

# Setup logging
def setup_logging():
    """Configures logging for the CLI."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "cli.log")
    
    formatter = EmojiFormatter("%(asctime)s - %(levelname)s - %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    
    # Clear existing handlers to enforce our configuration
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler]
    )

@click.group()
def cli():
    """LEGO ML CLI - Command Line Interface for training, inference, and dataset processing."""
    setup_logging()

@click.command()
@click.option("--mode", type=click.Choice(["bricks", "studs"]), required=True, help="Training mode")
@click.option("--epochs", type=int, default=20, help="Number of epochs")
@click.option("--batch-size", type=int, default=16, help="Batch size")
@click.option("--use-pretrained", is_flag=True, help="Use LEGO-trained model")
@click.option("--cleanup/--no-cleanup", default=True, help="Remove cached datasets, logs and results after training")
@click.option("--force-extract", is_flag=True, help="Force re-extraction of dataset")
@click.option("--show-results", is_flag=True, default=True, help="Display results after training")
def train(mode, epochs, batch_size, use_pretrained, cleanup, force_extract, show_results):
    """Train a YOLO model."""
    logging.info("Starting training...")
    command = [
        "python3",
        "train.py",
        "--mode", mode,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]
    if use_pretrained:
        command.append("--use-pretrained")
    if cleanup:
        command.append("--cleanup")
    else:
        command.append("--no-cleanup")
    if force_extract:
        command.append("--force-extract")
    if show_results:
        command.append("--show-results")
    else:
        command.append("--no-show-results")
    subprocess.run(command)

@click.command()
@click.option("--image", type=str, required=True, help="Path to image file")
@click.option("--save-annotated", is_flag=True, help="Save annotated results")
@click.option("--output", type=str, help="Output folder for predictions")
def predict(image, save_annotated, output):
    """Run inference on an image."""
    logging.info(f"Running inference on {image}")
    command = [sys.executable, "utils/model_utils.py", "--image", image]
    if save_annotated:
        command.append("--save-annotated")
    if output:
        command.extend(["--output", output])
    subprocess.run(command)

@click.command()
@click.option("--images", type=str, multiple=True, required=True, help="Paths to input images")
@click.option("--pipeline", type=click.Choice(["default", "detect_classify"]), default="detect_classify", help="Processing pipeline to use")
@click.option("--classification-method", type=click.Choice(["stud_count", "stud_positions"]), default="stud_count", help="Classification method")
@click.option("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
@click.option("--overlap", type=float, default=0.5, help="Overlap threshold for NMS")
@click.option("--user-comment", type=str, help="User comment to add to image metadata")
@click.option("--export-results", is_flag=True, help="Export results as zip")
def infer(images, pipeline, classification_method, confidence, overlap, user_comment, export_results):
    """Run inference workflows defined in model_utils."""
    logging.info("Running inference workflows...")
    command = [
        sys.executable,
        "utils/model_utils.py",
        "--images"
    ]
    # Append image paths
    for img in images:
        command.append(img)
    # Append parameters required by model_utils.py
    command.extend(["--pipeline", pipeline])
    command.extend(["--classification_method", classification_method])
    command.extend(["--confidence", str(confidence)])
    command.extend(["--overlap", str(overlap)])
    if user_comment:
        command.extend(["--user_comment", user_comment])
    if export_results:
        command.append("--export_results")
    subprocess.run(command)

@click.command()
def cleanup():
    """
    Cleans up temporary directories:
      - cache/
      - logs/
      - results/
    """
    logging.info("Cleaning up...")
    repo_root = get_repo_root()
    folders = ["cache", "results"]
    for folder in folders:
        folder_path = os.path.join(repo_root, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"✅ Removed: {folder_path}")
        else:
            logging.warning(f"❌ Not found: {folder_path}")



@click.command()
@click.argument("operation", type=click.Choice(["labelme-to-yolo", "keypoints-to-bboxes", "visualize"]))
@click.option("--input", type=str, required=True, help="Input folder path")
@click.option("--output", type=str, required=True, help="Output folder path")
@click.option("--area-ratio", type=float, help="Total area ratio for bounding boxes (only for keypoints-to-bboxes)")
def data_processing(operation, input, output, area_ratio):
    """Perform dataset processing tasks."""
    command = ["python3", "data_utils.py", f"--{operation}", "--input", input, "--output", output]
    if operation == "keypoints-to-bboxes" and area_ratio:
        command.extend(["--area-ratio", str(area_ratio)])
    logging.info(f"Executing data processing: {operation}")
    subprocess.run(command)

# Register commands
cli.add_command(train)
cli.add_command(predict)
cli.add_command(cleanup)
cli.add_command(data_processing)
cli.add_command(infer)

if __name__ == "__main__":
    cli()
