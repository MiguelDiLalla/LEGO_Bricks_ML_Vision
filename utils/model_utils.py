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

def predict(image_paths, model, mode, confidence_threshold=0.5, overlap_threshold=0.5):
    """
    Runs YOLO inference on a batch of images to detect bricks or studs.

    Args:
        image_paths (list of str): List of paths to the input images.
        model (YOLO): The loaded YOLO model.
        mode (str): One of ["bricks", "studs", "classify"].
        confidence_threshold (float): Minimum confidence score for detections.
        overlap_threshold (float): Overlapping threshold for NMS.

    Returns:
        list of dict: Structured results for each image, including detected objects and metadata.
    """
    # Validate image paths
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Image file not found: {path}")

    # Load and preprocess images
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"[ERROR] Unable to read image: {path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)

    # Run YOLO inference on the batch of images
    logging.info(f"🔍 Running inference on {len(images)} images with mode '{mode}'")
    results = model.predict(images)

    batch_metadata = []
    for i, result in enumerate(results):
        detected_objects = []
        for detection in result.boxes:
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

        # Prepare metadata for the current image
        metadata = {
            "image_path": image_paths[i],
            "mode": mode,
            "detections": len(filtered_objects),
            "objects": filtered_objects,
            "confidence_threshold": confidence_threshold,
            "overlap_threshold": overlap_threshold
        }
        batch_metadata.append(metadata)

    return batch_metadata

def classify_brick(brick_image, model_studs, confidence_threshold=0.5):
    """
    Classifies the dimensions of a single LEGO brick by detecting and counting studs.

    Args:
        brick_image (numpy.ndarray): Cropped image of the brick.
        model_studs (YOLO): YOLO model trained to detect studs.
        confidence_threshold (float): Minimum confidence score for stud detections.

    Returns:
        str: Classified dimension of the brick (e.g., "2x2", "4x1", or "UKN").
    """
    # Detect studs in the brick image
    results = model_studs.predict(brick_image)

    # Filter detections based on confidence threshold
    studs = [detection for detection in results[0].boxes if detection.conf[0] >= confidence_threshold]

    # Count the number of detected studs
    stud_count = len(studs)

    # Determine possible dimensions based on stud count
    possible_dimensions = STUD_TO_DIMENSION_MAP.get(stud_count, "UKN")

    if possible_dimensions == "UKN":
        return "UKN"

    # If multiple possible dimensions, apply heuristics to select the most likely one
    if isinstance(possible_dimensions, list):
        # Example heuristic: Use aspect ratio to distinguish between dimensions
        height, width = brick_image.shape[:2]
        aspect_ratio = width / height

        # Define aspect ratio thresholds (these values may need tuning)
        if aspect_ratio > 1.5:
            return possible_dimensions[1]  # Likely a longer brick (e.g., "4x1")
        else:
            return possible_dimensions[0]  # Likely a square brick (e.g., "2x2")

    return possible_dimensions

def detect_and_classify(image_paths, model_bricks, model_studs, confidence_threshold=0.5, overlap_threshold=0.5):
    """
    Detects LEGO bricks in images and classifies their dimensions.

    Args:
        image_paths (list of str): Paths to the input images.
        model_bricks (YOLO): YOLO model trained to detect bricks.
        model_studs (YOLO): YOLO model trained to detect studs.
        confidence_threshold (float): Minimum confidence score for detections.
        overlap_threshold (float): Overlap threshold for Non-Maximum Suppression.

    Returns:
        list of dict: Detection results for each image, including bounding boxes, confidence scores, and classified dimensions.
    """
    results = []

    # Load and preprocess images
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            logging.warning(f"Unable to read image: {path}")
            continue
        images.append((path, image))

    # Perform batch inference to detect bricks
    brick_detections = model_bricks.predict([img[1] for img in images])

    for (path, image), detections in zip(images, brick_detections):
        image_results = []
        for detection in detections.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            confidence = float(detection.conf[0])  # Confidence score

            # Apply confidence threshold
            if confidence < confidence_threshold:
                continue

            # Crop the detected brick from the image
            brick_image = image[y1:y2, x1:x2]

            # Classify the brick's dimensions
            dimension_label = classify_brick(brick_image, model_studs, confidence_threshold)

            # Store the detection result
            image_results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "dimension": dimension_label
            })

        results.append({
            "image_path": path,
            "detections": image_results
        })

    return results


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
    Main execution function for LEGO Brick Classification & Detection.
    """
    parser = argparse.ArgumentParser(description="LEGO Brick Classification & Detection")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Paths to the input images")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs", "classify"], required=True, help="Select mode: 'bricks', 'studs', or 'classify'")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--plt-annotated", action="store_true", help="Display annotated images")
    parser.add_argument("--export-results", action="store_true", help="Export results as a zip file to execution directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--overlap-threshold", type=float, default=0.5, help="Overlap threshold for Non-Maximum Suppression")

    args = parser.parse_args()

    logging.info("🚀 Starting LEGO Brick Inference...")

    # Load the appropriate model(s) based on the mode
    if args.mode == "classify":
        model_bricks = load_model("bricks")
        model_studs = load_model("studs")
    else:
        model = load_model(args.mode)

    # Create results folder inside cache
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = os.path.join("cache/results", f"{args.mode}_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    # Run inference
    if args.mode == "classify":
        results = detect_and_classify(
            image_paths=args.images,
            model_bricks=model_bricks,
            model_studs=model_studs,
            confidence_threshold=args.confidence_threshold,
            overlap_threshold=args.overlap_threshold
        )
    else:
        results = predict(
            image_paths=args.images,
            model=model,
            mode=args.mode,
            confidence_threshold=args.confidence_threshold,
            overlap_threshold=args.overlap_threshold
        )

    # Print results
    logging.info("✅ Inference complete.")
    for result in results:
        logging.info(json.dumps(result, indent=4))

    # Zip results if requested
    if args.export_results:
        zip_results(results_folder)


if __name__ == "__main__":
    main()
