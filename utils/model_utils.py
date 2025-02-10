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
import piexif
import cv2
from PIL import Image, ImageDraw, ImageFont, ExifTags


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

STUD_TO_DIMENSION_MAP = {
    1: "1x1",
    2: "2x1",
    3: "3x1",
    4: ["2x2", "4x1"],
    6: ["3x2", "6x1"],
    8: ["4x2", "8x1"],
    10: "10x1",
    12: "6x2",
    16: "8x2",
}

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

def draw_bboxes(image, detections, with_labels=True, box_color=(0, 255, 0), text_color=(255, 255, 255), thickness=2, font_scale=0.5):
    """
    Draws bounding boxes with optional labels on an image.

    Args:
        image (numpy.ndarray): The input image on which to draw.
        detections (list of dict): Each dict should have 'bbox' (list of [x1, y1, x2, y2]),
                                   'confidence' (float), and 'dimension' (str) keys.
        with_labels (bool): If True, labels the boxes with dimension and confidence.
        box_color (tuple): Color of the bounding box in BGR format.
        text_color (tuple): Color of the text in BGR format.
        thickness (int): Thickness of the bounding box lines.
        font_scale (float): Scale of the font for labels.

    Returns:
        numpy.ndarray: The image with drawn bounding boxes.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        if with_labels:
            label = f"{det['dimension']} ({det['confidence']:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), box_color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return image

def visualize_single_image(image_path, detections=None, with_labels=True, cache_dir="cache"):
    """
    Generates an annotated image with metadata displayed below and saves it to the cache.

    Args:
        image_path (str): Path to the image file.
        detections (list of dict, optional): Detections to draw on the image.
        with_labels (bool): If True, displays labels on the bounding boxes.
        cache_dir (str): Directory to save the cached image.

    Returns:
        str: Path to the saved annotated image in the cache.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes if detections are provided
    if detections:
        image = draw_bboxes(image, detections, with_labels)

    # Extract EXIF metadata
    pil_image = Image.open(image_path)
    exif_data = pil_image._getexif()
    metadata = ""
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata += f"{tag}: {value}\n"

    # Create figure with space for metadata
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image)
    ax.axis('off')

    # Add metadata below the image
    plt.figtext(0.5, 0.01, metadata, wrap=True, horizontalalignment='center', fontsize=10)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    cached_image_path = os.path.join(cache_dir, os.path.basename(image_path))
    plt.savefig(cached_image_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    return cached_image_path

def visualize_grid(images_folder_path, detections_dict=None, mode='bricks', grid_dimensions=(2, 2), cache_dir="cache"):
    """
    Creates a grid of images with optional bounding boxes and labels, and saves it to the cache.

    Args:
        images_folder_path (str): Path to the folder containing images.
        detections_dict (dict, optional): Dictionary where keys are image filenames and values are detection lists.
        mode (str): Mode of visualization ('bricks', 'studs', 'classify').
        grid_dimensions (tuple): Dimensions of the grid (rows, cols).
        cache_dir (str): Directory to save the generated grid image.

    Returns:
        str: Path to the saved grid image in the cache.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Retrieve image files
    image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = image_files[:grid_dimensions[0] * grid_dimensions[1]]

    # Load images and apply detections if available
    images = []
    for img_file in selected_images:
        img_path = os.path.join(images_folder_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if detections_dict and img_file in detections_dict:
            image = draw_bboxes(image, detections_dict[img_file], with_labels=True)
        images.append(image)

    # Determine max width and height for resizing
    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    # Resize images to have the same dimensions
    resized_images = [cv2.resize(image, (max_width, max_height)) for image in images]

    # Create the grid
    rows, cols = grid_dimensions
    grid_image = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)

    for idx, image in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        grid_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = image

    # Save the grid image to cache
    grid_filename = f"{mode}_grid.jpg"
    grid_path = os.path.join(cache_dir, grid_filename)
    grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(grid_path, grid_image_bgr)

    return grid_path

def add_metadata(image_path, output_path, user_comment):
    """
    Adds a user comment to the EXIF metadata of an image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with added metadata.
        user_comment (str): Comment to add to the image's EXIF data.
    """
    image = Image.open(image_path)
    exif_dict = piexif.load(image.info.get('exif', b''))

    # Add user comment
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment, encoding="unicode")

    # Save image with new EXIF data
    exif_bytes = piexif.dump(exif_dict)
    image.save(output_path, "jpeg", exif=exif_bytes)

def save_annotated_image(image_path, detections=None, destination_folder=None, logo_path="presentation/logo.png"):
    """
    Saves an annotated image with a logo overlay and formatted EXIF metadata.

    Args:
        image_path (str): Path to the input image.
        detections (list of dict, optional): Detections to draw on the image.
        destination_folder (str, optional): Directory to save the final image. Defaults to current working directory.
        logo_path (str): Path to the logo image for branding.

    Returns:
        str: Path to the saved annotated image.
    """
    # Load the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes if detections are provided
    if detections:
        image_rgb = draw_bboxes(image_rgb, detections, with_labels=True)

    # Convert to PIL Image for easier manipulation
    pil_image = Image.fromarray(image_rgb)

    # Load the logo image
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_width, logo_height = logo.size
    else:
        raise FileNotFoundError(f"Logo file not found at {logo_path}")

    # Position the logo at the bottom-right corner with a margin
    margin = 10
    image_width, image_height = pil_image.size
    logo_position = (image_width - logo_width - margin, image_height - logo_height - margin)

    # Overlay the logo onto the image
    pil_image.paste(logo, logo_position, logo)

    # Extract EXIF metadata
    exif_data = pil_image._getexif()
    if exif_data:
        exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    else:
        exif = {}

    # Format metadata as a console-style dictionary
    metadata_str = "Metadata:\n" + "\n".join(f"{key}: {value}" for key, value in exif.items())

    # Create a new image to append below the original for metadata
    font = ImageFont.load_default()
    text_size = font.getsize_multiline(metadata_str)
    metadata_image = Image.new("RGB", (image_width, text_size[1] + margin), (255, 255, 255))
    draw = ImageDraw.Draw(metadata_image)
    draw.text((margin, margin // 2), metadata_str, font=font, fill=(0, 0, 0))

    # Combine the original image with the metadata image
    combined_image = Image.new("RGB", (image_width, image_height + metadata_image.height))
    combined_image.paste(pil_image, (0, 0))
    combined_image.paste(metadata_image, (0, image_height))

    # Ensure the destination directory exists
    if destination_folder is None:
        destination_folder = os.getcwd()
    os.makedirs(destination_folder, exist_ok=True)

    # Save the final image
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(destination_folder, output_filename)
    combined_image.save(output_path)

    return output_path

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
    Main function to process and brand images for portfolio presentation.
    """
    parser = argparse.ArgumentParser(description="LEGO Brick Branding Processor")
    parser.add_argument("--source", type=str, required=True, help="Path to an image or a directory of images")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs", "classify"], required=True, help="Processing mode")
    parser.add_argument("--destination", type=str, default=os.getcwd(), help="Destination folder to save processed images")
    parser.add_argument("--grid-dimensions", type=int, nargs=2, metavar=('rows', 'cols'), default=(2, 2), help="Grid dimensions for multiple images (rows cols)")

    args = parser.parse_args()

    logging.info("🚀 Starting the branding process...")

    # Ensure the destination directory exists
    os.makedirs(args.destination, exist_ok=True)

    if os.path.isfile(args.source):
        # Process a single image
        detections = predict(args.source, args.mode)
        save_annotated_image(args.source, detections, destination_folder=args.destination)
    elif os.path.isdir(args.source):
        # Process a directory of images
        image_files = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_images = random.sample(image_files, min(len(image_files), args.grid_dimensions[0] * args.grid_dimensions[1]))
        detections_dict = {img: predict(img, args.mode) for img in selected_images}
        visualize_grid(selected_images, detections_dict, args.mode, args.grid_dimensions, args.destination)
    else:
        logging.error("Invalid source path provided. Please provide a valid image file or directory.")

    logging.info("✅ Branding process completed.")

if __name__ == "__main__":
    main()
