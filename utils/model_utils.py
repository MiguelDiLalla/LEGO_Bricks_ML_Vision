import os
import sys
import json
import random
import logging
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

import sys
import os
# Append the project root folder (one level up from the utils folder)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure lego_cli.py is in the PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "notebooks/LEGO_Bricks_ML_Vision"))

from lego_cli import EmojiFormatter  # Import the custom EmojiFormatter class from lego_cli.py

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(EmojiFormatter("%(asctime)s - %(levelname)s - %(message)s"))

def load_model(mode):
    """
    Loads the YOLO model based on the selected mode.
    """
    # Treat 'classify' mode as 'bricks' for model loading.
    if mode == "classify":
        mode = "bricks"

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


def predict(image_paths, model, mode, batch_size=8, save_annotated=False, plt_annotated=False, results_folder=None):
    """
    Perform batch prediction on a list of images.

    Args:
        image_paths (list): List of file paths for input images.
        model (YOLO): A loaded YOLO model.
        mode (str): Inference mode, such as "bricks", "studs", or "classify".
        batch_size (int): Number of images processed per batch.
        save_annotated (bool): Flag to save annotated images.
        plt_annotated (bool): Flag to display annotated images.
        results_folder (str): Directory to store results.

    Returns:
        list: A list of dictionaries with image paths and detection results.
    """
    results = []
    num_images = len(image_paths)
    logging.info(f"Total images to process: {num_images}")

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        # Load images; use list comprehension with error handling
        batch_images = []
        valid_paths = []
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"[Warning] Unable to load image: {img_path}")
            else:
                batch_images.append(img)
                valid_paths.append(img_path)

        if not batch_images:
            continue

        batch_results = model(batch_images)
        for img_path, result in zip(valid_paths, batch_results):
            annotated_img = None
            if save_annotated or plt_annotated:
                annotated_img = result.plot()
                if save_annotated and results_folder:
                    base_name = os.path.basename(img_path)
                    save_path = os.path.join(results_folder, f"annotated_{base_name}")
                    cv2.imwrite(save_path, annotated_img)
                    logging.info(f"✅ Annotated image saved to: {save_path}")
                if plt_annotated:
                    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Annotated {os.path.basename(img_path)}")
                    plt.axis('off')
                    plt.show()

            # Example of processing detections; additional mode-specific code can go here
            results.append({
                'image_path': img_path,
                'detections': result.boxes.data.cpu().numpy().tolist()
            })
    return results

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

def classify_brick_from_studs(studs, image, working_folder, save_annotated=False, plt_annotated=False):
    """
    Classifies the brick dimension based on detected stud positions.

    Args:
        studs (list): List of tuples representing stud bounding boxes (x_min, y_min, x_max, y_max).
        image (numpy.ndarray): The original image containing the detected studs.
        working_folder (str): Directory to save annotated images.
        save_annotated (bool): Flag to save annotated images.
        plt_annotated (bool): Flag to display annotated images.

    Returns:
        str: Classified brick dimension or an error message.
    """
    if len(studs) == 0:
        print("[INFO] No studs detected. Returning 'Unknown'.")
        return "Unknown"

    # Validate number of studs
    num_studs = len(studs)
    valid_stud_counts = STUD_TO_DIMENSION_MAP.keys()
    if num_studs not in valid_stud_counts:
        print(f"[ERROR] Deviant number of studs detected ({num_studs}). Returning 'Error'.")
        return "Deviant number of studs. Definitely bad inference. Sorry."

    # Extract stud center coordinates
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in studs]

    # Mean bounding box size for spacing calculation
    box_sizes = [((x_max - x_min + y_max - y_min) / 2) for x_min, y_min, x_max, y_max in studs]

    # Fit a regression line using least squares
    xs, ys = zip(*centers)
    m, b = np.polyfit(xs, ys, 1)  # Linear regression (y = mx + b)

    # Compute deviation from the line
    deviations = [abs(y - (m * x + b)) for x, y in centers]

    # Decision: Nx1 or Nx2
    threshold = np.mean(box_sizes) / 2
    classification_aux = "Nx1" if max(deviations) < threshold else "Nx2"

    # Determine final brick dimension
    possible_dimensions = STUD_TO_DIMENSION_MAP.get(num_studs, "Unknown")

    if isinstance(possible_dimensions, list):  # If there is ambiguity
        final_dimension = possible_dimensions[0] if classification_aux == "Nx2" else possible_dimensions[1]
    else:
        final_dimension = possible_dimensions

    # Visualization
    if save_annotated or plt_annotated:
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Plot studs
        for x, y in centers:
            plt.scatter(x, y, color='red', s=40)

        # Plot regression line
        x_line = np.array([min(xs), max(xs)])
        y_line = m * x_line + b
        plt.plot(x_line, y_line, color='cyan', linestyle='dashed')

        # Display classification decision
        plt.text(10, 30, f"Classification: {final_dimension}", fontsize=14, color='white',
                 bbox=dict(facecolor='black', alpha=0.7))

        plt.axis('off')
        if save_annotated:
            os.makedirs(working_folder, exist_ok=True)
            save_path = os.path.join(working_folder, "classification_result.png")
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[INFO] Annotated image saved to: {save_path}")
        if plt_annotated:
            plt.show()

    return final_dimension

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

def draw_bboxes(image, detections, with_labels=True, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draws bounding boxes with optional labels on an image. The thickness and font scale
    are determined dynamically based on the image dimensions.

    Args:
        image (numpy.ndarray): The input image on which to draw.
        detections (list of dict): Each dict should have 'bbox' (list of [x1, y1, x2, y2]),
                                   'confidence' (float), and 'dimension' (str) keys.
        with_labels (bool): If True, labels the boxes with dimension and confidence.
        box_color (tuple): Color of the bounding box in BGR format.
        text_color (tuple): Color of the text in BGR format.

    Returns:
        numpy.ndarray: The image with drawn bounding boxes.
    """
    h, w = image.shape[:2]
    # Dynamically determine thickness and font_scale based on image dimensions
    dynamic_thickness = max(1, int(round(min(h, w) / 500)))
    dynamic_font_scale = max(0.5, min(h, w) / 1000)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, dynamic_thickness)

        if with_labels:
            label = f"{det['dimension']} ({det['confidence']:.2f})"
            (w_box, h_box), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, dynamic_thickness)
            cv2.rectangle(image, (x1, y1 - h_box - 5), (x1 + w_box, y1), box_color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, text_color, dynamic_thickness)

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

def compose_final_image(image_path, detections, logo_path="presentation/logo.png", output_folder="composed_results"):
    """
    Composes a final output image that integrates:
    - The annotated image (using detections via draw_bboxes)
    - A red vertical frame on the right (half the width of the original image)
    - In the red area, the metadata (from EXIF) written in white with a console style font
    - The logo placed at the bottom-right of the red area (dynamically resized)
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the image using cv2 and create an annotated copy
    original_cv = cv2.imread(image_path)
    if original_cv is None:
        logging.error(f"Unable to load image: {image_path}")
        return None
    annotated_cv = draw_bboxes(original_cv.copy(), detections, with_labels=True)
    
    # Convert annotated image to PIL format (RGB)
    annotated_img = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
    orig_width, orig_height = annotated_img.size
    
    # Create a new canvas that is wider: original width + half of original width (red frame)
    frame_width = orig_width // 2
    new_width = orig_width + frame_width
    new_image = Image.new("RGB", (new_width, orig_height), (255, 0, 0))  # red background
    
    # Paste the annotated image on left portion
    new_image.paste(annotated_img, (0, 0))
    
    # Extract metadata from the original image (if exists)
    pil_img = Image.open(image_path)
    exif_data = pil_img._getexif()
    metadata_str = ""
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata_str += f"{tag}: {value}\n"
    else:
        metadata_str = "No metadata found."
    
    # Prepare to draw metadata text on the red frame area
    draw = ImageDraw.Draw(new_image)
    
    # Try to load a monospaced (console style) font. This will use a default if not available.
    try:
        # Adjust font path as necessary for Windows; for example, 'Consola.ttf' from Consolas
        font = ImageFont.truetype("consola.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    
    # Define text area starting at orig_width with some margin
    text_x = orig_width + 10
    text_y = 10
    # Set spacing and text color
    text_color = (255, 255, 255)
    # Draw the metadata text (wrap as needed manually, here we simply draw each line)
    for line in metadata_str.splitlines():
        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += font.getsize(line)[1] + 2
    
    # Load and dynamically resize the logo to fit into red area: for example, width=30% of frame width
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        desired_logo_width = frame_width * 0.3
        # maintain aspect ratio
        logo_ratio = logo.height / logo.width
        new_logo_size = (int(desired_logo_width), int(desired_logo_width * logo_ratio))
        logo = logo.resize(new_logo_size, Image.ANTIALIAS)
        # Compute position: bottom-right in red area with a small margin
        margin = 10
        logo_x = orig_width + frame_width - new_logo_size[0] - margin
        logo_y = orig_height - new_logo_size[1] - margin
        # Paste the logo (using its alpha as mask)
        new_image.paste(logo, (logo_x, logo_y), logo)
    else:
        logging.warning(f"Logo file not found at {logo_path}")
    
    # Save the composed image
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, output_filename)
    new_image.save(output_path)
    logging.info(f"✅ Final composed image saved to: {output_path}")
    return output_path

def main():
    """
    Main execution function for model inference and image composition.
    """
    parser = argparse.ArgumentParser(description="LEGO Brick Classification & Detection")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Paths to the input images")
    parser.add_argument("--mode", type=str, choices=["bricks", "studs", "classify"], required=True, help="Select mode: 'bricks', 'studs', or 'classify'")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of images to process in a batch")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--plt-annotated", action="store_true", help="Display annotated images")
    parser.add_argument("--export-results", action="store_true", help="Export results as a zip file to execution directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("🚀 Starting LEGO Brick Inference...")
    
    # Load the model based on the provided mode
    model = load_model(args.mode)
    
    # Create a results folder inside cache (for annotated outputs, if not using composed images)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = os.path.join("cache/results", f"{args.mode}_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    
    # Run inference to get detection results
    results = predict(
        image_paths=args.images,
        model=model,
        mode=args.mode,
        batch_size=args.batch_size,
        save_annotated=args.save_annotated,
        plt_annotated=args.plt_annotated,
        results_folder=results_folder
    )
    
    logging.info("✅ Inference complete.")
    logging.info(json.dumps(results, indent=4))
    
    # For every input image, compose final image with red frame and metadata
    composed_folder = os.path.join("composed_results", f"{args.mode}_{timestamp}")
    os.makedirs(composed_folder, exist_ok=True)
    
    for result in results:
        image_path = result.get("image_path")
        detections = result.get("detections", [])
        compose_final_image(image_path, detections, logo_path="presentation/logo.png", output_folder=composed_folder)
    
    # Zip results if requested
    if args.export_results:
        # Uncomment the next line if you wish to zip the results folder
        # zip_results(results_folder)

if __name__ == "__main__":
    main()