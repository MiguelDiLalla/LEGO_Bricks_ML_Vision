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



def main():
    """
    Main execution function for model inference.
    """
    parser = argparse.ArgumentParser(description="LEGO Brick Classification & Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs", "classify"], required=True, help="Select mode: 'bricks', 'studs', or 'classify'")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--plt-annotated", action="store_true", help="Display annotated images")
    
    args = parser.parse_args()
    
    logging.info("🚀 Starting LEGO Brick Inference...")

    # Load the model
    model = load_model(args.mode)

    # Run inference (we will implement predict() next)
    results = predict(args.image, model, args.mode, args.save_annotated, args.plt_annotated)

    # Print results
    logging.info("✅ Inference complete.")
    logging.info(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
