"""
Main pipeline for detecting LEGO bricks and studs using YOLO models.

This script includes:
1. Brick and stud detection.
2. EXIF metadata saving.
3. Classification of LEGO dimensions based on stud count and alignment.
"""

# Imports (Consolidated)
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import platform
import datetime
from ultralytics import YOLO
from PIL import Image, ExifTags

def labelme_to_yolo(input_folder, output_folder):
    """
    Convert LabelMe JSON annotations to YOLO format, ensuring positive width and height.

    Args:
        input_folder (str): Path to the folder containing LabelMe JSON files.
        output_folder (str): Path to the folder where YOLO .txt files will be saved.
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
                    
                    # Extract coordinates and ensure proper ordering
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    x_min, x_max = sorted([x1, x2])  # Ensure x_min is smaller
                    y_min, y_max = sorted([y1, y2])  # Ensure y_min is smaller

                    # Compute YOLO format values
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    # Write to YOLO .txt format
                    yolo_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Conversion completed. YOLO .txt files saved to: {output_folder}")


def convert_keypoints_json(input_folder, output_folder, total_area_ratio=0.4):
    """
    Convert LabelMe keypoints into bounding boxes and save a new JSON file.

    Args:
        input_folder (str): Path to the folder containing LabelMe JSON files.
        output_folder (str): Path to save transformed JSON files.
        total_area_ratio (float): The fraction of the image area that all bounding boxes should occupy.
                                  Default is 0.4 (40% of the image area).
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
            image_area = image_width * image_height

            # Extract keypoints
            keypoints = [shape["points"][0] for shape in data["shapes"] if shape["shape_type"] == "point"]
            num_keypoints = len(keypoints)

            if num_keypoints == 0:
                continue  # Skip images with no keypoints

            # Compute the target bounding box area per keypoint
            total_target_area = total_area_ratio * image_area  # 40% of image area
            box_area_per_keypoint = total_target_area / num_keypoints

            # Compute bounding box size (assuming square boxes for simplicity)
            box_size = math.sqrt(box_area_per_keypoint)

            # Update JSON to replace keypoints with bounding boxes
            new_shapes = []
            for x_center, y_center in keypoints:
                x1 = max(0, x_center - box_size / 2)
                y1 = max(0, y_center - box_size / 2)
                x2 = min(image_width, x_center + box_size / 2)
                y2 = min(image_height, y_center + box_size / 2)

                new_shapes.append({
                    "label": "Stud",
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                })

            # Replace old shapes with new bounding boxes
            data["shapes"] = new_shapes

            # Save the transformed JSON file
            output_path = os.path.join(output_folder, json_file)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

    print(f"JSON conversion completed. Transformed files saved to: {output_folder}")

def visualize_yolo_annotations(image_path_or_folder, labels_folder, mode, extra_param="default"):
    """
    Visualize YOLO format annotations in Jupyter Notebook.

    Args:
        image_path_or_folder (str): Path to a single image or folder containing images.
        labels_folder (str): Path to the folder containing YOLO annotation .txt files.
        mode (int): 
            1 - Display a single annotated image
            2 - Save a single annotated image
            3 - Display a grid of images
            4 - Save all images with annotations
        extra_param (str/int): Extra options per mode (e.g., max images for grid, output folder for saves).
    """
    def load_yolo_annotations(label_path, image_width, image_height):
        """Load YOLO annotations and return bounding boxes."""
        bboxes = []
        with open(label_path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                _, x_center, y_center, width, height = map(float, parts)
                x1 = int((x_center - width / 2) * image_width)
                y1 = int((y_center - height / 2) * image_height)
                x2 = int((x_center + width / 2) * image_width)
                y2 = int((y_center + height / 2) * image_height)
                bboxes.append((x1, y1, x2, y2))
        return bboxes

    def annotate_image(image_path, label_path):
        """Draw bounding boxes on an image and return as a NumPy array."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for Matplotlib
        height, width, _ = image.shape
        bboxes = load_yolo_annotations(label_path, width, height)
        
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    if mode == 1:  # Display single image
        label_path = os.path.join(labels_folder, os.path.basename(image_path_or_folder).replace('.jpg', '.txt'))
        image = annotate_image(image_path_or_folder, label_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    elif mode == 2:  # Save single annotated image
        output_path = image_path_or_folder.replace('.jpg', '_annotated.jpg') if extra_param == "default" else os.path.join(extra_param, os.path.basename(image_path_or_folder).replace('.jpg', '_annotated.jpg'))
        label_path = os.path.join(labels_folder, os.path.basename(image_path_or_folder).replace('.jpg', '.txt'))
        cv2.imwrite(output_path, cv2.cvtColor(annotate_image(image_path_or_folder, label_path), cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

    elif mode == 3:  # Display grid of images
        images = [f for f in os.listdir(image_path_or_folder) if f.endswith('.jpg')]
        random.shuffle(images)
        max_images = extra_param if isinstance(extra_param, int) else 6
        selected_images = images[:max_images]

        cols = 3
        rows = (len(selected_images) + cols - 1) // cols

        fig = plt.figure(figsize=(12, 4 * rows), facecolor='black')
        grid = plt.GridSpec(rows, cols, wspace=0, hspace=0)  # Removes spacing

        for idx, img in enumerate(selected_images):
            ax = fig.add_subplot(grid[idx])
            label_path = os.path.join(labels_folder, img.replace('.jpg', '.txt'))
            annotated = annotate_image(os.path.join(image_path_or_folder, img), label_path)
            ax.imshow(annotated)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        plt.show()

    elif mode == 4:  # Save all images with annotations
        if extra_param == "default":
            print("Error: Please specify an output folder.")
            return
        os.makedirs(extra_param, exist_ok=True)
        for img in os.listdir(image_path_or_folder):
            if img.endswith('.jpg'):
                label_path = os.path.join(labels_folder, img.replace('.jpg', '.txt'))
                output_path = os.path.join(extra_param, img.replace('.jpg', '_annotated.jpg'))
                cv2.imwrite(output_path, cv2.cvtColor(annotate_image(os.path.join(image_path_or_folder, img), label_path), cv2.COLOR_RGB2BGR))

    print("Processing completed.")


# Mapping Table for Stud Count to Brick Dimensions


def classify_brick_from_studs(studs, image, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED):
    """Classifies the brick dimension based on detected stud positions."""
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
    centers = [( (x1 + x2) / 2, (y1 + y2) / 2 ) for x1, y1, x2, y2 in studs]

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
    if SAVE_ANNOTATED or PLT_ANNOTATED:
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
        plt.text(10, 30, f"Classification: {final_dimension}", fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        plt.axis('off')
        if SAVE_ANNOTATED:
            plt.savefig(os.path.join(working_folder, "classification_result.png"), bbox_inches='tight')
        if PLT_ANNOTATED:
            plt.show()
    
    return final_dimension


def get_model_hash(model_path):
    """Generate a SHA256 hash for a model file."""
    with open(model_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def save_exif_metadata(image_path, metadata):
    """Save EXIF metadata inside an image file."""
    image = Image.open(image_path)
    exif = image.getexif()  # Get existing EXIF data or create a new one

    # Convert metadata to a JSON string
    metadata_str = json.dumps(metadata)

    # Assign metadata to EXIF fields using tag IDs
    exif[270] = metadata_str  # ImageDescription
    exif[37510] = "LEGO Brick Inference Log"  # UserComment
    exif[305] = "LEGO Detection v1.0"  # Software
    exif[315] = "https://www.kaggle.com/yourprofile"  # Artist

    try:
        # Save image with updated EXIF
        image.save(image_path, exif=exif)
        print("[INFO] EXIF metadata saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save EXIF metadata: {e}")

def retrieve_exif_metadata(image_path, CLEAN_IF_TRUE=False):
    """Retrieve and optionally clean EXIF metadata from an image file."""
    image = Image.open(image_path)
    exif_data = image.getexif()

    if not exif_data:
        return False, {}

    # Correctly map the tag IDs for reading EXIF fields
    tag_map = {
        "ImageDescription": 270,
        "UserComment": 37510,
        "Software": 305,
        "Artist": 315
    }

    # Retrieve metadata from the EXIF tags
    metadata = exif_data.get(tag_map["ImageDescription"], None)
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=4))
        except json.JSONDecodeError:
            print("[WARNING] Failed to parse JSON from EXIF metadata.")
            return False, {}
    else:
        return False, {}

    # Optionally clean the EXIF metadata
    if CLEAN_IF_TRUE:
        for tag in tag_map.values():
            exif_data.pop(tag, None)  # Remove the tags
        image.save(image_path, exif=exif_data)
        print("[INFO] EXIF metadata cleaned.")

    return True, metadata_dict

def detect_objects(image, model, label, confidence_threshold=0.5, working_folder=None, SAVE_ANNOTATED=False, PLT_ANNOTATED=False):
    """
    Generalized function for object detection.

    Args:
        image (numpy array): Input image for detection.
        model (YOLO): Pre-trained YOLO model.
        label (str): Label for visualization (e.g., 'Brick', 'Stud').
        confidence_threshold (float): Minimum confidence score for keeping detections.
        working_folder (str, optional): Path to save annotated images.
        SAVE_ANNOTATED (bool): If True, saves annotated images.
        PLT_ANNOTATED (bool): If True, displays annotated images.

    Returns:
        list: List of bounding boxes [x_min, y_min, x_max, y_max].
    """
    results = model(image)
    objects = [box.xyxy[0].tolist() for box in results[0].boxes if box.conf >= confidence_threshold]

    # Optional visualization
    if SAVE_ANNOTATED or PLT_ANNOTATED:
        for (x1, y1, x2, y2) in objects:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0) if label == 'Brick' else (255, 0, 0), 2)
        if SAVE_ANNOTATED and working_folder:
            filename = f"annotated_{label.lower()}s.jpg"
            cv2.imwrite(os.path.join(working_folder, filename), image)
        if PLT_ANNOTATED:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    return objects

def detect_bricks(image, model_bricks, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED, confidence_threshold=0.5):
    return detect_objects(image, model_bricks, "Brick", confidence_threshold, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)

def detect_studs(image, model_studs, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED, confidence_threshold=0.5):
    return detect_objects(image, model_studs, "Stud", confidence_threshold, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)

def crop_brick(image, brick_bbox, working_folder):
    """Crops a detected brick from the image and saves it."""
    x1, y1, x2, y2 = map(int, brick_bbox)
    cropped_brick = image[y1:y2, x1:x2]
    cropped_path = os.path.join(working_folder, f"cropped_brick_{x1}_{y1}.jpg")
    cv2.imwrite(cropped_path, cropped_brick)
    return cropped_brick

def save_metadata(metadata, working_folder, output_format="json"):
    """
    Save metadata to a specified format (JSON/CSV).

    Args:
        metadata (dict): Metadata dictionary.
        working_folder (str): Path to save the metadata.
        output_format (str): Output format ('json' or 'csv').

    Raises:
        ValueError: If the specified output format is not supported.
    """
    if output_format == "json":
        json_path = os.path.join(working_folder, "inference_metadata.json")
        with open(json_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
    elif output_format == "csv":
        import pandas as pd
        csv_path = os.path.join(working_folder, "inference_metadata.csv")
        pd.DataFrame([metadata]).to_csv(csv_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

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

def predict_brick_dimensions(image_path, model_bricks, model_studs, mode, working_folder=None, SAVE_ANNOTATED=False, PLT_ANNOTATED=False, SAVE_JSON=False):
    """
    Predicts the dimensions of a LEGO brick using object detection models.
    
    Args:
        image_path (str): Path to the image file.
        model_bricks (YOLO): Pre-trained YOLO model for detecting LEGO bricks.
        model_studs (YOLO): Pre-trained YOLO model for detecting studs.
        mode (str): One of ['BricksOnly', 'StudsOnly', 'BricksAndStuds'].
        working_folder (str, optional): Folder to save intermediate results. Defaults to 'inferences/' inside the image's folder.
        SAVE_ANNOTATED (bool): If True, saves annotated images.
        PLT_ANNOTATED (bool): If True, displays annotated images.
        SAVE_JSON (bool): If True, saves metadata as JSON.
    """
    # 1. Set up the working folder
    if working_folder is None:
        working_folder = os.path.join(os.path.dirname(image_path), "inferences")
    os.makedirs(working_folder, exist_ok=True)
    
    # 2. Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image file not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"[ERROR] Unable to read the image file: {image_path}. Ensure it's a valid image format.")

    if image is None:
        raise ValueError("Error loading image. Check the file path.")
    
    # 3. Model Hashes
    model_bricks_hash = get_model_hash(model_bricks.ckpt_path) if model_bricks else "N/A"
    model_studs_hash = get_model_hash(model_studs.ckpt_path) if model_studs else "N/A"

    
    # 4. Environment Metadata
    metadata = {
        "image_name": os.path.basename(image_path),
        "inference_mode": mode,
        "model_bricks_hash": model_bricks_hash,
        "model_studs_hash": model_studs_hash,
        "inference_datetime": datetime.datetime.now().isoformat(),
        "script_execution_path": os.path.abspath(sys.argv[0]) if hasattr(sys, 'argv') else "Jupyter Notebook",
        "working_directory": os.getcwd(),
        "python_version": platform.python_version(),
        "os_info": platform.platform(),
        "hardware_info": platform.processor(),
    }
    
    # 5. Run Brick Detection if mode requires it
    if mode in ['BricksOnly', 'BricksAndStuds']:
        bricks = detect_bricks(image, model_bricks, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)
        metadata["total_bricks_detected"] = len(bricks)
    
    # 6. Run Stud Detection if mode requires it
    if mode in ['StudsOnly', 'BricksAndStuds']:
        studs = detect_studs(image, model_studs, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)
        metadata["total_studs_detected"] = len(studs)
        
        # 7. If StudsOnly, apply classification logic
        dimensions = classify_brick_from_studs(studs, image, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)
        metadata["classification_result"] = dimensions
        
    # 8. If BricksAndStuds, apply classification to each detected brick
    if mode == 'BricksAndStuds':
        metadata["bricks_analysis"] = []
        for brick in bricks:
            cropped_brick = crop_brick(image, brick, working_folder)
            studs_in_brick = detect_studs(cropped_brick, model_studs, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)
            dimension = classify_brick_from_studs(studs_in_brick, cropped_brick, working_folder, SAVE_ANNOTATED, PLT_ANNOTATED)
            metadata["bricks_analysis"].append({"brick_bbox": brick, "dimension": dimension})
    
    # 9. Save EXIF metadata inside image
    save_exif_metadata(image_path, metadata)
    print(f"[debug] Metadata saved to image: {image_path}")
    print(json.dumps(metadata, indent=4))
    
    # 10. Save JSON metadata if requested
    if SAVE_JSON:
        save_metadata(metadata, working_folder, output_format="json")
    
    print("Prediction completed.")

