import os
import sys
import json
import random
import logging
import platform
import datetime
import hashlib
import argparse
import cv2
import numpy as np
import torch
import shutil
import zipfile
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ExifTags

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(mode):
    """
    Loads the YOLO model based on the selected mode.
    
    Args:
        mode (str): 'bricks', 'studs', or 'classify'.
    
    Returns:
        YOLO: Loaded model object.
    """
    model_paths = {
        "bricks": "presentation/Models_DEMO/Brick_Model_best20250123_192838t.pt",
        "studs": "presentation/Models_DEMO/Stud_Model_best20250124_170824.pt"
    }
    
    if mode not in model_paths:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'bricks' or 'studs'.")
    
    model_path = model_paths[mode]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"🔹 Loading model: {model_path}")
    return YOLO(model_path)

def predict(image_path, model, mode, confidence_threshold=0.5, overlap_threshold=0.5):
    """
    Runs YOLO inference on an image to detect bricks or studs.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): The loaded YOLO model.
        mode (str): One of ["bricks", "studs", "classify"].
        confidence_threshold (float): Minimum confidence score for detections.
        overlap_threshold (float): Overlapping threshold for NMS.

    Returns:
        dict: Structured results including detected objects and metadata.
    """
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image file not found: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"[ERROR] Unable to read image: {image_path}")

    # Convert to RGB (YOLO expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    logging.info(f"🔍 Running inference on {image_path} with mode '{mode}'")
    results = model.predict(image_rgb)

    # Extract detections
    detected_objects = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box
        confidence = float(detection.conf[0])  # Confidence score

        # Apply confidence threshold
        if confidence < confidence_threshold:
            continue

        detected_objects.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence
        })

    # Apply Non-Maximum Suppression (NMS) to remove duplicate overlapping boxes
    filtered_objects = apply_nms(detected_objects, overlap_threshold)

    # Prepare metadata
    metadata = {
        "image_path": image_path,
        "mode": mode,
        "detections": len(filtered_objects),
        "objects": filtered_objects,
        "confidence_threshold": confidence_threshold,
        "overlap_threshold": overlap_threshold
    }

    return metadata


def apply_nms(detections, overlap_threshold):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        detections (list): List of detected objects with bounding boxes.
        overlap_threshold (float): IoU threshold for suppression.

    Returns:
        list: Filtered list of detections after NMS.
    """
    if not detections:
        return []

    # Convert to NumPy array for easier calculations
    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["confidence"] for det in detections])

    # Apply OpenCV NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=overlap_threshold)

    return [detections[i] for i in indices.flatten()]

def save_annotated_image(image, detections, results_folder, mode):
    """
    Draws bounding boxes on the image and saves the annotated image.

    Args:
        image (numpy array): The original image.
        detections (list): List of bounding boxes to draw.
        results_folder (str): Folder to save the annotated image.
        mode (str): Inference mode ('bricks' or 'studs').

    Returns:
        str: Path to the saved annotated image.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for bricks/studs

    # Define save path
    filename = f"{mode}_annotated.jpg"
    save_path = os.path.join(results_folder, filename)
    cv2.imwrite(save_path, image)
    logging.info(f"📸 Annotated image saved to {save_path}")

    return save_path

def zip_results(results_folder, output_path=None):
    """
    Compresses inference results into a zip file.

    Args:
        results_folder (str): Path to the folder containing inference results.
        output_path (str, optional): If provided, zip file will be saved here. Otherwise, it defaults to execution directory.

    Returns:
        str: Path to the generated zip file.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"results_{timestamp}.zip"

    if output_path is None:
        output_path = os.getcwd()  # Default: Execution directory

    zip_filepath = os.path.join(output_path, zip_filename)

    shutil.make_archive(zip_filepath.replace('.zip', ''), 'zip', results_folder)
    logging.info(f"✅ Results exported to {zip_filepath}")
    return zip_filepath

def main():
    """
    Main execution function for model inference.
    """
    parser = argparse.ArgumentParser(description="LEGO Brick Classification & Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs", "classify"], required=True, help="Select mode: 'bricks', 'studs', or 'classify'")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--plt-annotated", action="store_true", help="Display annotated images")
    parser.add_argument("--export-results", action="store_true", help="Export results as a zip file to execution directory")

    args = parser.parse_args()

    logging.info("🚀 Starting LEGO Brick Inference...")

    # Load the model
    model = load_model(args.mode)

    # Create results folder inside cache
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = os.path.join("cache/results", f"{args.mode}_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    # Run inference
    results = predict(args.image, model, args.mode, args.save_annotated, args.plt_annotated, results_folder)

    # Print results
    logging.info("✅ Inference complete.")
    logging.info(json.dumps(results, indent=4))

    # Zip results if requested
    if args.export_results:
        zip_results(results_folder)

if __name__ == "__main__":
    main()
