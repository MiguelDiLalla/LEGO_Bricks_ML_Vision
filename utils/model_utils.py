#!/usr/bin/env python

import os
import sys
import json
import random
import logging
import datetime
import argparse
import cv2
import numpy as np
import shutil
import zipfile
import matplotlib.pyplot as plt
from ultralytics import YOLO
import piexif
from PIL import Image, ImageDraw, ImageFont, ExifTags

# Append project root and ensure lego_cli.py is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.getcwd(), "notebooks/LEGO_Bricks_ML_Vision"))

from lego_cli import EmojiFormatter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(EmojiFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# Global mapping for stud count to dimension
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

def load_model(mode):
    """
    Loads the YOLO model based on the selected mode.
    """
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
    """
    if not detections:
        return []
    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["confidence"] for det in detections])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=overlap_threshold)
    return [detections[i] for i in indices.flatten()]

def classify_brick(brick_image, model_studs, confidence_threshold=0.5):
    """
    Classifies a brick based on the number of detected studs.
    """
    results = model_studs.predict(brick_image)
    studs = [detection for detection in results[0].boxes if detection.conf[0] >= confidence_threshold]
    stud_count = len(studs)
    possible_dimensions = STUD_TO_DIMENSION_MAP.get(stud_count, "UKN")
    if possible_dimensions == "UKN":
        return "UKN"
    if isinstance(possible_dimensions, list):
        height, width = brick_image.shape[:2]
        aspect_ratio = width / height
        return possible_dimensions[1] if aspect_ratio > 1.5 else possible_dimensions[0]
    return possible_dimensions

def classify_brick_from_studs(studs, image, working_folder, save_annotated=False, plt_annotated=False):
    """
    Alternative brick classification using stud positions.
    """
    if len(studs) == 0:
        logging.info("No studs detected. Returning 'Unknown'.")
        return "Unknown"
    num_studs = len(studs)
    if num_studs not in STUD_TO_DIMENSION_MAP.keys():
        logging.error(f"Deviant number of studs detected ({num_studs}).")
        return "Error"
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in studs]
    box_sizes = [((x_max - x_min + y_max - y_min) / 2) for x_min, y_min, x_max, y_max in studs]
    xs, ys = zip(*centers)
    m, b = np.polyfit(xs, ys, 1)
    deviations = [abs(y - (m * x + b)) for x, y in centers]
    threshold = np.mean(box_sizes) / 2
    classification_aux = "Nx1" if max(deviations) < threshold else "Nx2"
    possible_dimensions = STUD_TO_DIMENSION_MAP.get(num_studs, "Unknown")
    if isinstance(possible_dimensions, list):
        final_dimension = possible_dimensions[0] if classification_aux == "Nx2" else possible_dimensions[1]
    else:
        final_dimension = possible_dimensions
    if save_annotated or plt_annotated:
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for x, y in centers:
            plt.scatter(x, y, color='red', s=40)
        x_line = np.array([min(xs), max(xs)])
        y_line = m * x_line + b
        plt.plot(x_line, y_line, color='cyan', linestyle='dashed')
        plt.text(10, 30, f"Classification: {final_dimension}", fontsize=14, color='white',
                 bbox=dict(facecolor='black', alpha=0.7))
        plt.axis('off')
        if save_annotated:
            os.makedirs(working_folder, exist_ok=True)
            save_path = os.path.join(working_folder, "classification_result.png")
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Annotated image saved to: {save_path}")
        if plt_annotated:
            plt.show()
    return final_dimension

def detect_and_classify(image_paths, model_bricks, model_studs, classification_method="stud_count",
                        confidence_threshold=0.5, overlap_threshold=0.5):
    """
    Detects bricks and classifies their dimensions using either stud count or stud positions.
    """
    results = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            logging.warning(f"Unable to read image: {path}")
            continue
        brick_results = model_bricks.predict(image)
        image_detections = []
        for detection in brick_results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            conf = float(detection.conf[0])
            if conf < confidence_threshold:
                continue
            brick_image = image[y1:y2, x1:x2]
            if classification_method == "stud_count":
                dimension = classify_brick(brick_image, model_studs, confidence_threshold)
            else:
                stud_results = model_studs.predict(brick_image)
                studs = stud_results[0].boxes.data.cpu().numpy().tolist()
                dimension = classify_brick_from_studs(studs, brick_image, working_folder="stud_classification")
            image_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "dimension": dimension
            })
        # Filter overlapping detections using NMS
        image_detections = apply_nms(image_detections, overlap_threshold)
        results.append({
            "image_path": path,
            "detections": image_detections
        })
    return results

def draw_bboxes(image, detections, with_labels=True, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draws bounding boxes and labels on an image.
    """
    h, w = image.shape[:2]
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
    Generates an annotated image with metadata below and saves it.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if detections:
        image = draw_bboxes(image, detections, with_labels)
    pil_image = Image.open(image_path)
    exif_data = pil_image._getexif()
    metadata = ""
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata += f"{tag}: {value}\n"
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image)
    ax.axis('off')
    plt.figtext(0.5, 0.01, metadata, wrap=True, horizontalalignment='center', fontsize=10)
    os.makedirs(cache_dir, exist_ok=True)
    cached_image_path = os.path.join(cache_dir, os.path.basename(image_path))
    plt.savefig(cached_image_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    return cached_image_path

def visualize_grid(images_folder_path, detections_dict=None, mode='bricks', grid_dimensions=(2, 2), cache_dir="cache"):
    """
    Creates a grid visualization of images.
    """
    os.makedirs(cache_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = image_files[:grid_dimensions[0] * grid_dimensions[1]]
    images = []
    for img_file in selected_images:
        img_path = os.path.join(images_folder_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if detections_dict and img_file in detections_dict:
            image = draw_bboxes(image, detections_dict[img_file], with_labels=True)
        images.append(image)
    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    resized_images = [cv2.resize(image, (max_width, max_height)) for image in images]
    rows, cols = grid_dimensions
    grid_image = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)
    for idx, image in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        grid_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = image
    grid_filename = f"{mode}_grid.jpg"
    grid_path = os.path.join(cache_dir, grid_filename)
    grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(grid_path, grid_image_bgr)
    return grid_path

def add_metadata(image_path, output_path, user_comment):
    """
    Adds a user comment to an image's EXIF metadata.
    """
    image = Image.open(image_path)
    exif_dict = piexif.load(image.info.get('exif', b''))
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment, encoding="unicode")
    exif_bytes = piexif.dump(exif_dict)
    image.save(output_path, "jpeg", exif=exif_bytes)

def save_annotated_image(image_path, detections=None, destination_folder=None, logo_path="presentation/logo.png"):
    """
    Saves an annotated image with a logo overlay and metadata.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if detections:
        image_rgb = draw_bboxes(image_rgb, detections, with_labels=True)
    pil_image = Image.fromarray(image_rgb)
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
    else:
        raise FileNotFoundError(f"Logo file not found at {logo_path}")
    margin = 10
    image_width, image_height = pil_image.size
    logo_position = (image_width - logo.width - margin, image_height - logo.height - margin)
    pil_image.paste(logo, logo_position, logo)
    exif_data = pil_image._getexif()
    if exif_data:
        exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    else:
        exif = {}
    metadata_str = "Metadata:\n" + "\n".join(f"{key}: {value}" for key, value in exif.items())
    font = ImageFont.load_default()
    text_size = font.getsize_multiline(metadata_str)
    metadata_image = Image.new("RGB", (image_width, text_size[1] + margin), (255, 255, 255))
    draw = ImageDraw.Draw(metadata_image)
    draw.text((margin, margin // 2), metadata_str, font=font, fill=(0, 0, 0))
    combined_image = Image.new("RGB", (image_width, image_height + metadata_image.height))
    combined_image.paste(pil_image, (0, 0))
    combined_image.paste(metadata_image, (0, image_height))
    if destination_folder is None:
        destination_folder = os.getcwd()
    os.makedirs(destination_folder, exist_ok=True)
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(destination_folder, output_filename)
    combined_image.save(output_path)
    return output_path

def compose_final_image(image_path, detections, logo_path="presentation/logo.png", output_folder="composed_results"):
    """
    Composes a final image with the annotated image on the left and a red frame with metadata on the right.
    """
    os.makedirs(output_folder, exist_ok=True)
    original_cv = cv2.imread(image_path)
    if original_cv is None:
        logging.error(f"Unable to load image: {image_path}")
        return None
    annotated_cv = draw_bboxes(original_cv.copy(), detections, with_labels=True)
    annotated_img = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
    orig_width, orig_height = annotated_img.size
    frame_width = orig_width // 2
    new_width = orig_width + frame_width
    new_image = Image.new("RGB", (new_width, orig_height), (255, 0, 0))
    new_image.paste(annotated_img, (0, 0))
    pil_img = Image.open(image_path)
    exif_data = pil_img._getexif()
    metadata_str = ""
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata_str += f"{tag}: {value}\n"
    else:
        metadata_str = "No metadata found."
    draw_obj = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("consola.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    text_x = orig_width + 10
    text_y = 10
    text_color = (255, 255, 255)
    for line in metadata_str.splitlines():
        draw_obj.text((text_x, text_y), line, font=font, fill=text_color)
        bbox = font.getbbox(line)    # get the bounding box for the text
        line_height = bbox[3] - bbox[1]
        text_y += line_height + 2
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        desired_logo_width = frame_width * 0.3
        logo_ratio = logo.height / logo.width
        new_logo_size = (int(desired_logo_width), int(desired_logo_width * logo_ratio))
        logo = logo.resize(new_logo_size, Image.LANCZOS)
        margin = 10
        logo_x = orig_width + frame_width - new_logo_size[0] - margin
        logo_y = orig_height - new_logo_size[1] - margin
        new_image.paste(logo, (logo_x, logo_y), logo)
    else:
        logging.warning(f"Logo file not found at {logo_path}")
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, output_filename)
    new_image.save(output_path)
    logging.info(f"Final composed image saved to: {output_path}")
    return output_path

def zip_results(results_folder, output_path=None):
    """
    Compresses the specified folder into a zip file.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"results_{timestamp}.zip"
    if output_path is None:
        output_path = os.getcwd()
    zip_filepath = os.path.join(output_path, zip_filename)
    shutil.make_archive(zip_filepath.replace('.zip', ''), 'zip', results_folder)
    logging.info(f"Results exported to {zip_filepath}")
    return zip_filepath

def compose_all_outputs(results, output_base="final_outputs", user_comment=None):
    """
    Creates a variety of composed outputs for each image, integrating all the composition functions.
    """
    os.makedirs(output_base, exist_ok=True)
    composed_images = []
    for res in results:
        image_path = res["image_path"]
        detections = res.get("detections", [])
        composed_path = compose_final_image(image_path, detections, output_folder=os.path.join(output_base, "composed"))
        annotated_path = save_annotated_image(image_path, detections, destination_folder=os.path.join(output_base, "annotated"))
        single_vis_path = visualize_single_image(image_path, detections, cache_dir=os.path.join(output_base, "single_vis"))
        composed_images.append({
            "original": image_path,
            "composed": composed_path,
            "annotated": annotated_path,
            "single_visualization": single_vis_path
        })
        if user_comment:
            meta_output = os.path.join(output_base, "metadata")
            os.makedirs(meta_output, exist_ok=True)
            meta_image_path = os.path.join(meta_output, os.path.basename(image_path))
            add_metadata(image_path, meta_image_path, user_comment)
    grid_path = visualize_grid(os.path.join(output_base, "annotated"), mode="annotated", grid_dimensions=(2, 2), cache_dir=output_base)
    return composed_images, grid_path

def main():
    parser = argparse.ArgumentParser(description="Refactored LEGO Brick Composition Pipeline")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Paths to input images")
    parser.add_argument("--pipeline", type=str, choices=["default", "detect_classify"], default="detect_classify",
                        help="Choose processing pipeline")
    parser.add_argument("--classification_method", type=str, choices=["stud_count", "stud_positions"],
                        default="stud_count", help="Choose classification method")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap threshold for NMS")
    parser.add_argument("--user_comment", type=str, default=None, help="User comment to add to image metadata")
    parser.add_argument("--export_results", action="store_true", help="Zip final outputs")
    args = parser.parse_args()

    logging.info("Starting Refactored LEGO Brick Pipeline...")

    if args.pipeline == "detect_classify":
        model_bricks = load_model("bricks")
        model_studs = load_model("studs")
        results = detect_and_classify(args.images, model_bricks, model_studs,
                                      classification_method=args.classification_method,
                                      confidence_threshold=args.confidence,
                                      overlap_threshold=args.overlap)
    else:
        model = load_model("bricks")
        results = []
        for image in args.images:
            res = model(image)
            detections = []
            for det in res[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf[0])
                detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "dimension": "N/A"})
            results.append({"image_path": image, "detections": detections})

    composed_images, grid_path = compose_all_outputs(results, user_comment=args.user_comment)
    logging.info("Composition complete. Outputs:")
    logging.info(json.dumps(composed_images, indent=4))
    logging.info(f"Grid visualization saved at: {grid_path}")

    if args.export_results:
        zip_results(os.path.join("final_outputs"))

if __name__ == "__main__":
    main()
