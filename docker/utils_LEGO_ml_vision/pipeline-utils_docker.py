import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from ultralytics import YOLO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths inside Docker container
DATASET_PATH = os.getenv("DATASET_PATH", "/app/data/datasets")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/app/data/output")
MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")
KAGGLE_SECRET_PATH = "/root/.kaggle/kaggle.json"

def setup_kaggle_auth():
    if os.path.exists("/run/secrets/kaggle_secret"):
        os.makedirs("/root/.kaggle", exist_ok=True)
        os.system("cp /run/secrets/kaggle_secret /root/.kaggle/kaggle.json")
        os.chmod("/root/.kaggle/kaggle.json", 0o600)

def download_kaggle_model(model_name, output_path):
    kaggle_models = {
        "bricks": "migueldilalla/lego_bricks_machinevisonyolofinetune",
        "studs": "migueldilalla/lego_brickstuds_machinevisionyolofinetune"
    }
    
    if model_name in kaggle_models:
        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, f"{model_name}.pt")
        if not os.path.exists(model_path):
            logging.info(f"[INFO] Downloading {model_name} model from Kaggle...")
            os.system(f"kaggle models download -m {kaggle_models[model_name]} -p {output_path} --unzip")
        return model_path
    
    logging.error(f"[ERROR] Model {model_name} not found in predefined Kaggle models.")
    return None

def detect_objects(image, model, confidence_threshold=0.5):
    results = model(image)
    return [box.xyxy[0].tolist() for box in results[0].boxes if box.conf >= confidence_threshold]

def classify_brick_from_studs(studs):
    STUD_TO_DIMENSION_MAP = {
        1: "1x1", 2: "2x1", 3: "3x1", 4: ["2x2", "4x1"],
        6: ["3x2", "6x1"], 8: ["4x2", "8x1"], 10: "10x1",
        12: "6x2", 16: "8x2"
    }
    if len(studs) == 0:
        return "Unknown"
    return STUD_TO_DIMENSION_MAP.get(len(studs), "Unclassified")

def main():
    setup_kaggle_auth()
    
    parser = argparse.ArgumentParser(description="Pipeline Utilities for LEGO Brick Detection")
    parser.add_argument("--detect-bricks", metavar="IMAGE", help="Run brick detection on an image")
    parser.add_argument("--detect-studs", metavar="IMAGE", help="Run stud detection on an image")
    parser.add_argument("--classify-studs", metavar="IMAGE", help="Classify LEGO brick based on detected studs")
    args = parser.parse_args()
    
    model_bricks_path = download_kaggle_model("bricks", MODELS_PATH)
    model_studs_path = download_kaggle_model("studs", MODELS_PATH)
    model_bricks = YOLO(model_bricks_path) if model_bricks_path else None
    model_studs = YOLO(model_studs_path) if model_studs_path else None
    
    if args.detect_bricks and model_bricks:
        image = cv2.imread(args.detect_bricks)
        bricks = detect_objects(image, model_bricks)
        logging.info(f"Bricks detected: {bricks}")
    
    if args.detect_studs and model_studs:
        image = cv2.imread(args.detect_studs)
        studs = detect_objects(image, model_studs)
        logging.info(f"Studs detected: {studs}")
    
    if args.classify_studs and model_studs:
        image = cv2.imread(args.classify_studs)
        studs = detect_objects(image, model_studs)
        classification = classify_brick_from_studs(studs)
        logging.info(f"Brick classification: {classification}")

if __name__ == "__main__":
    main()
