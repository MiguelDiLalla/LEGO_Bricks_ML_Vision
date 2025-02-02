import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import platform
import datetime
import argparse
import logging
from ultralytics import YOLO
from PIL import Image, ExifTags

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths inside Docker container
DATASET_PATH = os.getenv("DATASET_PATH", "/app/data/datasets")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/app/data/output")
MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")

# Ensure necessary directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

def labelme_to_yolo(input_folder, output_folder):
    """
    Convert LabelMe JSON annotations to YOLO format.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for json_file in os.listdir(input_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(input_folder, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_width = data["imageWidth"]
            image_height = data["imageHeight"]

            output_path = os.path.join(output_folder, json_file.replace('.json', '.txt'))
            with open(output_path, 'w') as yolo_file:
                for shape in data["shapes"]:
                    points = shape["points"]
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, x_max = sorted([x1, x2])
                    y_min, y_max = sorted([y1, y2])

                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    yolo_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    logging.info(f"YOLO annotations saved to: {output_folder}")

def detect_objects(image, model, label, confidence_threshold=0.5):
    """Run YOLO detection on an image."""
    results = model(image)
    objects = [box.xyxy[0].tolist() for box in results[0].boxes if box.conf >= confidence_threshold]
    return objects

def detect_bricks(image, model):
    return detect_objects(image, model, "Brick")

def detect_studs(image, model):
    return detect_objects(image, model, "Stud")

def classify_brick_from_studs(studs):
    """Classifies the brick dimension based on detected stud positions."""
    if len(studs) == 0:
        return "Unknown"
    return f"Detected {len(studs)} studs"

def main():
    parser = argparse.ArgumentParser(description="Pipeline Utilities for LEGO Brick Detection")
    parser.add_argument("--convert-labelme", nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Convert LabelMe JSON annotations to YOLO format")
    parser.add_argument("--detect-bricks", metavar="IMAGE", help="Run brick detection on an image")
    parser.add_argument("--detect-studs", metavar="IMAGE", help="Run stud detection on an image")
    parser.add_argument("--classify-studs", metavar="IMAGE", help="Classify LEGO brick based on detected studs")
    
    args = parser.parse_args()
    
    if args.convert_labelme:
        labelme_to_yolo(args.convert_labelme[0], args.convert_labelme[1])
    elif args.detect_bricks:
        model = YOLO(os.path.join(MODELS_PATH, "bricks.pt"))
        image = cv2.imread(args.detect_bricks)
        bricks = detect_bricks(image, model)
        logging.info(f"Bricks detected: {bricks}")
    elif args.detect_studs:
        model = YOLO(os.path.join(MODELS_PATH, "studs.pt"))
        image = cv2.imread(args.detect_studs)
        studs = detect_studs(image, model)
        logging.info(f"Studs detected: {studs}")
    elif args.classify_studs:
        model = YOLO(os.path.join(MODELS_PATH, "studs.pt"))
        image = cv2.imread(args.classify_studs)
        studs = detect_studs(image, model)
        classification = classify_brick_from_studs(studs)
        logging.info(f"Brick classification: {classification}")

if __name__ == "__main__":
    main()
