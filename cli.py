import click
import subprocess
import logging
import os
import shutil

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
@click.option("--cleanup", is_flag=True, help="Remove cached datasets after training")
# By default, zip_results is True. To disable, we add a reverse flag.
@click.option("--zip-results", "zip_results", flag_value=True, default=True, help="Compress training results after completion (default: enabled)")
@click.option("--no-zip-results", "zip_results", flag_value=False, help="Do not compress training results after completion")
def train(mode, epochs, batch_size, use_pretrained, cleanup, zip_results):
    """Train a YOLO model."""
    logging.info("Starting training...")
    command = ["python3", "train.py", "--mode", mode, "--epochs", str(epochs), "--batch-size", str(batch_size)]
    if use_pretrained:
        command.append("--use-pretrained")
    if cleanup:
        command.append("--cleanup")
    # Pass the ZIP flag according to the parsed value.
    if zip_results:
        command.append("--zip-results")
    else:
        command.append("--no-zip-results")
    subprocess.run(command)

@click.command()
@click.option("--image", type=str, required=True, help="Path to image file")
@click.option("--save-annotated", is_flag=True, help="Save annotated results")
@click.option("--output", type=str, help="Output folder for predictions")
def predict(image, save_annotated, output):
    """Run inference on an image."""
    logging.info(f"Running inference on {image}")
    command = ["python3", "model_utils.py", "--image", image]
    if save_annotated:
        command.append("--save-annotated")
    if output:
        command.extend(["--output", output])
    subprocess.run(command)

@click.command()
def cleanup():
    """Clear cached files."""
    cache_dirs = ["cache/datasets", "cache/models", "cache/results", "cache/logs"]
    logging.info("⚠️ WARNING: This will delete cached datasets and models.")
    if click.confirm("Are you sure?", default=False):
        for folder in cache_dirs:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logging.info(f"Deleted {folder}")
        logging.info("✅ Cache cleanup complete.")
    else:
        logging.info("Cache cleanup aborted.")

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

if __name__ == "__main__":
    cli()
