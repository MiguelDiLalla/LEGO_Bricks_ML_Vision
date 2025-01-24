# LEGO Bricks Inference Script
# Developed to provide flexible, efficient, and didactic utilities for LEGO detection and annotation

import os
import cv2
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# === 1. Dynamic Parameter Configurator ===
def configure_parameters():
    """
    Interactive configurator for inference parameters.

    Returns:
        dict: Configured parameters including model path, thresholds, and output folders.
    """
    config = {}

    # Prompt for model selection
    print("[INFO] Configuring parameters...")
    model_path = input("Enter path to the YOLO model file (default: models/checkpoints/yolov8n.pt): ").strip()
    config['model_path'] = model_path if model_path else 'models/checkpoints/yolov8n.pt'

    # Confidence threshold
    try:
        config['conf_threshold'] = float(input("Enter confidence threshold (default: 0.5): ").strip())
    except ValueError:
        config['conf_threshold'] = 0.5

    # Output folders
    config['annotated_folder'] = input("Enter path for annotated images folder (default: results/predictions): ").strip()
    config['annotated_folder'] = config['annotated_folder'] if config['annotated_folder'] else 'results/predictions'

    config['cropped_folder'] = input("Enter path for cropped images folder (default: results/cropped_bricks): ").strip()
    config['cropped_folder'] = config['cropped_folder'] if config['cropped_folder'] else 'results/cropped_bricks'

    # Logging option
    log_choice = input("Enable logging? (y/n, default: y): ").strip().lower()
    config['enable_logging'] = log_choice != 'n'

    return config

# === 2. Predict and Annotate ===
def predict_and_annotate(image_path, model, conf_threshold=0.5, output_folder=None, display=False):
    """
    Run YOLO inference on a single image and save/display the annotated result.

    Args:
        image_path (str): Path to the input image.
        model: YOLO model object.
        conf_threshold (float): Confidence threshold for detection.
        output_folder (str): Folder to save the annotated image.
        display (bool): If True, display the annotated image.
    """
    img = cv2.imread(image_path)
    results = model.predict(source=image_path, conf=conf_threshold)

    # Draw bounding boxes
    annotated_img = img.copy()
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save or display the result
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_img)

    if display:
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# === 3. Crop Bounding Boxes ===
def crop_bounding_boxes(original_image_path, results, output_folder):
    """
    Crop bounding boxes from the original image based on YOLO predictions.

    Args:
        original_image_path (str): Path to the original high-resolution image.
        results: YOLO prediction results.
        output_folder (str): Folder to save cropped bounding boxes.
    """
    img = cv2.imread(original_image_path)
    os.makedirs(output_folder, exist_ok=True)

    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_img = img[y1:y2, x1:x2]
        output_path = os.path.join(output_folder, f"{Path(original_image_path).stem}_crop_{i}.jpg")
        cv2.imwrite(output_path, cropped_img)

# === 4. Batch Processing ===
def process_image_folder(folder_path, config):
    """
    Process all images in a folder: predict, annotate, and crop.

    Args:
        folder_path (str): Path to the folder containing images.
        config (dict): Configuration parameters.
    """
    model = YOLO(config['model_path'])

    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, image_file)
            results = model.predict(source=image_path, conf=config['conf_threshold'])

            # Annotate
            predict_and_annotate(image_path, model, config['conf_threshold'], config['annotated_folder'], display=False)

            # Crop
            crop_bounding_boxes(image_path, results, config['cropped_folder'])

# === 5. Logging and Reporting ===
def log_and_report(config, folder_path):
    """
    Generate logs and a basic report for processed files.

    Args:
        config (dict): Configuration parameters.
        folder_path (str): Folder containing processed images.
    """
    log_folder = os.path.join('results', 'logs')
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w') as f:
        f.write(f"Configuration: {json.dumps(config, indent=4)}\n")
        f.write(f"Processed files from folder: {folder_path}\n")
        f.write(f"Output folders: \n  Annotated: {config['annotated_folder']}\n  Cropped: {config['cropped_folder']}\n")

# === Main Execution ===
if __name__ == "__main__":
    # Dynamic parameter configuration
    config = configure_parameters()

    # Testing folder setup
    test_folder = os.path.join('test', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(test_folder, exist_ok=True)

    # Run processing pipeline
    folder_path = input("Enter the folder path with images to process: ").strip()
    if not os.path.exists(folder_path):
        print(f"[ERROR] The folder {folder_path} does not exist.")
    else:
        process_image_folder(folder_path, config)

    # Generate logs and report
    log_and_report(config, folder_path)
