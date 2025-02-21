"""
Model Utilities for LEGO Bricks ML Vision

Includes:
  - Model loading and inference functions with YOLO
  - EXIF metadata reading/writing (with scan count)
  - Image annotation and metadata panel rendering

All log messages include emoji markers for clear readability.
"""

import json
from PIL import Image, ExifTags
import numpy as np
import logging
import datetime
import platform
import os
import base64
import requests
from ultralytics import YOLO
import piexif
import cv2

# Set up professional logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Model Utils module loaded.")

def setup_utils(repo_download=False):
    """
    Initializes and returns a configuration dictionary with all global variables and defaults.
    """
    CONFIG_DICT = {}  # New configuration dictionary
    
    # Project repository configuration
    userGithub = "MiguelDiLalla"
    repoGithub = "LEGO_Bricks_ML_Vision"
    CONFIG_DICT["REPO_URL"] = f"https://api.github.com/repos/{userGithub}/{repoGithub}/contents/"
    logger.info("📌 REPO URL set to: %s", CONFIG_DICT["REPO_URL"])
    
    # Define model and test images folders relative to project structure.
    CONFIG_DICT["MODELS_PATHS"] = {
        "bricks": r"presentation/Models_DEMO/Brick_Model_best20250123_192838t.pt",
        "studs": r"presentation/Models_DEMO/Stud_Model_best20250124_170824.pt"
    }
    CONFIG_DICT["TEST_IMAGES_FOLDERS"] = {
        "bricks": r"presentation/Test_images/BricksPics",
        "studs": r"presentation/Test_images/StudsPics"
    }
    
    logger.info("📂 Current working directory: %s", os.getcwd())

    def get_image_files(folder):
        # ...existing code...
        full_path = os.path.join(os.getcwd(), folder)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        if not os.path.exists(full_path):
            return []
        return [os.path.join(full_path, f) for f in os.listdir(full_path) if f.lower().endswith(image_extensions)]
    
    CONFIG_DICT["TEST_IMAGES"] = {}
    for key, folder in CONFIG_DICT["TEST_IMAGES_FOLDERS"].items():
        full_folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(full_folder_path):
            files = get_image_files(folder)
            CONFIG_DICT["TEST_IMAGES"][key] = files
            logger.info("✅ Found %d images in %s", len(files), folder)
        else:
            CONFIG_DICT["TEST_IMAGES"][key] = []
            logger.info("⚠️ Folder %s does not exist; no images found.", folder)
    
    # Load models from disk
    CONFIG_DICT["LOADED_MODELS"] = {}
    for model_name, relative_path in CONFIG_DICT["MODELS_PATHS"].items():
        local_path = os.path.join(os.getcwd(), relative_path)
        if not os.path.exists(local_path):
            if repo_download:
                model_url = CONFIG_DICT["REPO_URL"] + relative_path
                logger.info("⬇️  Downloading %s model from %s", model_name, model_url)
                response = requests.get(model_url)
                if response.status_code == 200:
                    data = response.json()
                    model_data = base64.b64decode(data.get("content", ""))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as model_file:
                        model_file.write(model_data)
                else:
                    logger.error("❌ Failed to download %s model from %s", model_name, model_url)
            else:
                logger.error("❌ %s model not found locally and repo_download is disabled.", model_name)
        try:
            CONFIG_DICT["LOADED_MODELS"][model_name] = YOLO(local_path)
            logger.info("✅ %s model loaded.", model_name.capitalize())
        except Exception as e:
            logger.error("❌ Error loading %s model: %s", model_name, e)
            CONFIG_DICT["LOADED_MODELS"][model_name] = None
    
    # Retrieve project logo
    try:
        local_logo_path = os.path.join(os.getcwd(), "presentation", "logo.png")
        if os.path.exists(local_logo_path):
            CONFIG_DICT["LOGO_NUMPY"] = cv2.imread(local_logo_path)
            logger.info("🖼️ Logo found locally.")
        else:
            if repo_download:
                logo_url = CONFIG_DICT["REPO_URL"] + "presentation/logo.png"
                logger.info("⬇️ Logo not found locally. Downloading from %s", logo_url)
                response = requests.get(logo_url)
                if response.status_code == 200:
                    data = response.json()
                    logo_data = base64.b64decode(data.get("content", ""))
                    os.makedirs(os.path.dirname(local_logo_path), exist_ok=True)
                    with open(local_logo_path, "wb") as logo_file:
                        logo_file.write(logo_data)
                    CONFIG_DICT["LOGO_NUMPY"] = cv2.imread(local_logo_path)
                else:
                    logger.error("❌ Failed to download logo from %s", logo_url)
                    CONFIG_DICT["LOGO_NUMPY"] = None
            else:
                logger.error("❌ Logo not found locally and repo_download is disabled.")
                CONFIG_DICT["LOGO_NUMPY"] = None
    except Exception as e:
        logger.error("❌ Error loading logo: %s", e)
        CONFIG_DICT["LOGO_NUMPY"] = None

    # Mapping for studs to brick dimensions.
    CONFIG_DICT["STUDS_TO_DIMENSIONS_MAP"] = {
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

    # Default EXIF metadata backbone.
    CONFIG_DICT["EXIF_METADATA_DEFINITIONS"] = {
        "boxes_coordinates": {},       # Detected bounding box coordinates.
        "orig_shape": [0, 0],            # Original image dimensions.
        "speed": {                     # Processing time metrics.
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0
        },
        "mode": "",                    # Operation mode: detection/classification.
        "path": "",                    # Original image file path.
        "os_full_version_name": "",    # OS version information.
        "processor": "",               # Processor details.
        "architecture": "",            # System architecture.
        "hostname": "",                # Host machine name.
        "timestamp": "",               # Time of processing.
        "annotated_image_path": "",    # Path for annotated output.
        "json_results_path": "",       # Path for exported metadata.
        "TimesScanned": 0,             # Number of inference sessions.
        "Repository": CONFIG_DICT["REPO_URL"],        # Repository URL.
        "message": ""                  # Custom message.
    }
    
    return CONFIG_DICT

config = setup_utils()

# =============================================================================
# EXIF Functions
# =============================================================================

# --- Modify read_exif() to use configuration EXIF defaults if TREE not given ---
def read_exif(image_path, TREE=config["EXIF_METADATA_DEFINITIONS"]):
    """
    Reads EXIF metadata from an image file and logs scan status. 📸
    Returns:
      dict: Parsed metadata.
    """
    # Use default EXIF DEFINITIONS from CONFIG if TREE is None
    
    if TREE is None:
        TREE = config["EXIF_METADATA_DEFINITIONS"]

    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error("❌ Failed to open image %s: %s", image_path, e)
        return {}

    exif_bytes = image.info.get("exif")
    if not exif_bytes:
        logger.warning("⚠️ No EXIF data found in %s", image_path)
        return {}

    exif_dict = piexif.load(exif_bytes)
    user_comment_tag = piexif.ExifIFD.UserComment
    user_comment = exif_dict.get("Exif", {}).get(user_comment_tag, b"")
    if not user_comment:
        logger.warning("⚠️ No UserComment tag found in %s", image_path)
        return {}

    try:
        comment_str = user_comment.decode('utf-8', errors='ignore')
        metadata = json.loads(comment_str)
        # Ensure defaults from TREE are present
        for key, default in TREE.items():
            metadata.setdefault(key, default)
        times = metadata.get("TimesScanned", 0)
        if times:
            logger.info("🔄 Image %s has been scanned %d time(s)", image_path, times)
        else:
            logger.info("🆕 Image %s has not been scanned before", image_path)
        return metadata
    except Exception as e:
        logger.error("❌ Failed to parse EXIF metadata from %s: %s", image_path, e)
        return {}
    
def write_exif(image_path, metadata):
    """
    Writes metadata to the image's EXIF UserComment tag.
    Erases any existing UserComment data and resets 'TimesScanned' to 1. ✍️
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error("❌ Failed to open image %s: %s", image_path, e)
        return

    exif_bytes = image.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    user_comment_tag = piexif.ExifIFD.UserComment

    # Erase any existing metadata in UserComment and reset scan count
    metadata["TimesScanned"] = 1
    logger.info("🆕 Setting TimesScanned to 1 for image %s (existing metadata erased)", image_path)

    formatted_metadata = json.dumps(metadata, indent=4)
    encoded_metadata = formatted_metadata.encode('utf-8')
    exif_dict["Exif"][user_comment_tag] = encoded_metadata

    new_exif_bytes = piexif.dump(exif_dict)
    try:
        image.save(image_path, image.format if image.format else "jpeg", exif=new_exif_bytes)
        logger.info("✅ EXIF metadata written to %s", image_path)
    except Exception as e:
        logger.error("❌ Failed to save image with updated EXIF: %s", e)

# =============================================================================
# Metadata Extraction from YOLO Results
# =============================================================================

def extract_metadata_from_yolo_result(results, orig_image):
    """
    Extracts relevant metadata from YOLO results and the original image.
    Handles image as numpy array or file path. 💡
    """
    if isinstance(orig_image, str):
        loaded_image = cv2.imread(orig_image)
        shape = list(loaded_image.shape[:2]) if loaded_image is not None else [0, 0]
    elif hasattr(orig_image, "shape"):
        shape = list(orig_image.shape[:2])
    else:
        shape = [0, 0]

    boxes = (results[0].boxes.xyxy.cpu().numpy()
             if results and results[0].boxes.xyxy is not None
             else np.array([]))
    image_path = results[0].path if hasattr(results[0], "path") and results[0].path else ""

    previous_metadata = {}
    if image_path:
        try:
            previous_metadata = read_exif(image_path)
        except Exception as e:
            logger.error("❌ Error reading EXIF from %s: %s", image_path, e)
            previous_metadata = {}

    times_scanned = previous_metadata.get("TimesScanned", 0)
    times_scanned = times_scanned + 1 if times_scanned else 1

    message_value = "Muchas gracias por ejecutar la DEMO del projecto"
    metadata = {
        "boxes_coordinates": {str(idx): box.tolist() for idx, box in enumerate(boxes)},
        "orig_shape": shape,
        "speed": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
        "mode": "bricks",
        "path": image_path,
        "os_full_version_name": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "timestamp": datetime.datetime.now().isoformat(),
        "annotated_image_path": "",
        "json_results_path": "",
        "TimesScanned": times_scanned,
        "Repository": config["REPO_URL"],
        "message": message_value
    }
    return metadata

def render_metadata(image, metadata):
    """
    Creates a metadata panel and combines it with the original image.
    The panel includes formatted metadata and project logo from config. 🖼️
    Returns:
      composite_image (numpy.ndarray): rendered metadata + logo at the bottom
    """
    img_height, img_width = image.shape[:2]
    metadata_width = img_width // 4
    # Create red background panel
    metadata_panel = np.full((img_height, metadata_width, 3), (0, 0, 255), dtype=np.uint8)
    metadata_text = json.dumps(metadata, indent=4)
    lines = metadata_text.splitlines()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)
    margin = 10
    line_height = 15
    # Reserve space for logo fetched from config
    logo_height = 0
    logo_resized = None
    try:
        logo_img = config.get("LOGO_NUMPY")
        if logo_img is not None:
            logo_h, logo_w = logo_img.shape[:2]
            scale = metadata_width / logo_w
            logo_height = int(logo_h * scale)
            logo_resized = cv2.resize(logo_img, (metadata_width, logo_height))
    except Exception:
        logo_resized = None
        logo_height = 0
    text_area_height = img_height - logo_height - 2 * margin
    y = margin + line_height
    for line in lines:
        if y > text_area_height:
            break
        cv2.putText(metadata_panel, line, (margin, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y += line_height
    if logo_resized is not None:
        start_y = img_height - logo_height
        metadata_panel[start_y:img_height, 0:metadata_width] = logo_resized
    # composite_image = np.hstack((image, metadata_panel))
    return metadata_panel

def composite_inference_branded_image(annotated_image, rendered_metadata, margin=10):
    """
    Combines the annotated image with the rendered metadata panel and adds a red border. 🎨
    Returns the final composite image.
    """
    composite_no_margin = np.hstack((annotated_image, rendered_metadata))
    composite_image = cv2.copyMakeBorder(composite_no_margin, margin, margin, margin, margin,
                                           borderType=cv2.BORDER_CONSTANT, value=(0, 0, 255))
    return composite_image


# =============================================================================
# Brick Detection Function
# =============================================================================

def detect_bricks(image_input, model=None, conf=0.25, save_json=False, save_annotated=False, output_folder=""):
    """
    Performs brick detection using the provided YOLO model.
    Accepts either an image file path or a numpy array.
    Defaults to the loaded brick model if model is None.
    Optionally saves annotated image and metadata JSON. 🎯
    Returns a dictionary with original image, annotated image, cropped detections, and metadata.
    """

    # message if output folder is not provided
    if not output_folder:
        logger.warning("⚠️ No output folder provided. Results will not be saved.")


    if model is None:
        model = config.get("LOADED_MODELS", {}).get("bricks")
        if model is None:
            logger.error("❌ No brick model loaded.")
            return None

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            logger.error("❌ Image path does not exist: %s", image_input)
            return None
        image = cv2.imread(image_input)
        if image is None:
            logger.error("❌ Failed to load image from path: %s", image_input)
            return None
    else:
        image = image_input

    try:
        orig_image = image.copy()
        annotated_image = image.copy()
        results = model.predict(source=image, conf=conf)
        boxes_np = (results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else np.array([]))
        if boxes_np.size == 0:
            logger.warning("⚠️ No detections found. Using dummy detection.")
            boxes_np = np.array([[10, 10, 100, 100]])

        annotated_image = results[0].plot() 
        
        # [DEBUG] display the annotated image
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cropped_detections = []
        for idx, box in enumerate(boxes_np):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = orig_image[y1:y2, x1:x2]
            cropped_detections.append(crop)
            # cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(orig_image, f"Brick {idx}", (x1, y1-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        metadata = extract_metadata_from_yolo_result(results, orig_image)

        if not output_folder:
            output_folder = os.path.join(os.getcwd(), "results", "bricks")
        os.makedirs(output_folder, exist_ok=True)
        
        if save_annotated:
            rendered_metadata = render_metadata(annotated_image, metadata)
            composite_image = composite_inference_branded_image(annotated_image, rendered_metadata)
            composite_path = os.path.join(output_folder, "composite_image.jpg")
            cv2.imwrite(composite_path, composite_image)
            metadata["annotated_image_path"] = composite_path
            logger.info("💾 Composite image saved at: %s", composite_path)
        if save_json:
            json_path = os.path.join(output_folder, "metadata.json")
            with open(json_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)
            metadata["json_results_path"] = json_path
            logger.info("💾 Metadata JSON saved at: %s", json_path)

        if isinstance(image_input, str):
            write_exif(image_input, metadata)
        elif save_annotated and metadata.get("annotated_image_path"):
            write_exif(metadata["annotated_image_path"], metadata)
        
        return {
            "orig_image": orig_image,
            "annotated_image": annotated_image,
            "cropped_detections": cropped_detections,
            "metadata": metadata
        }
    except Exception as e:
        logger.error("❌ Error during brick detection: %s", e)
        return None

# =============================================================================
# Placeholder Functions for Stud Detection and Dimension Classification
# =============================================================================

def classify_dimensions(image):
    """
    Placeholder for dimension classification logic.
    """
    # ...existing code...
    return None

def detect_studs(image):
    """
    Placeholder for stud detection logic.
    """
    # ...existing code...
    return None

# =============================================================================
# Metadata Rendering Functions
# =============================================================================


# =============================================================================
# End of Model Utils Module
# =============================================================================
