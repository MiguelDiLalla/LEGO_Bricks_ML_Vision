# model_utils.py



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
import requests
from io import BytesIO

# Append project root and ensure lego_cli.py is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.getcwd(), "notebooks/LEGO_Bricks_ML_Vision"))


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
# Project repo:
userGithub = "MiguelDiLalla"
repoGithub = "LEGO_Bricks_ML_Vision"
REPO_URL = f"https://api.github.com/repos/{userGithub}/{repoGithub}/contents/"

# models paths
MODELS_PATHS = {
    "bricks": r"models/checkpoints/Brick_Model_best20250123_192838t.pt",
    "studs": r"models/checkpoints/Stud_Model_best20250124_170824.pt"
}

# project featured models
GITHUB_URLS = {
    "bricks": os.path.join(REPO_URL, MODELS_PATHS["bricks"]),
    "studs": os.path.join(REPO_URL, MODELS_PATHS["studs"])
}

def load_models(urls=GITHUB_URLS):
    '''
    Load bricks model and studs model from fixed GitHub URLs.
    If fetching from GitHub fails, load the model from a local file.
    Returns a dictionary with the YOLO models ready to use.
    '''
    
    import tempfile
    import requests
    from ultralytics import YOLO
    import os

    models = {}
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    
    for key, url in urls.items():
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(response.content)
                tmp.flush()
                temp_filepath = tmp.name
            model = YOLO(temp_filepath)
        except Exception as e:
            print(f"Failed to fetch {key} model from GitHub (error: {e}). Fetching locally...")
            # Local model paths are relative to the parent folder of the current working directory.
            parent_dir = os.path.dirname(os.getcwd())
            local_path = os.path.join(parent_dir, MODELS_PATHS[key])
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local model file not found: {local_path}")
            model = YOLO(local_path)
        models[key] = model
    return models

import json
import piexif
import numpy as np
def write_metadata_to_exif(enriched_results=None):
    """
    Given the enriched results dictionary and the path to the original image file,
    this function writes inference metadata into the image's EXIF UserComment tag.
    If the file already stores our metadata, the function will update the
    "TimesScanned" key (incrementing it) while merging new metadata from enriched_results.
    Any non-serializable components (like numpy arrays) are skipped.
    The metadata is stored as a JSON string.
    Repository: https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision
    """
    import json, os, piexif

    if enriched_results is None or enriched_results.get("encoded_metadata") is None:
        return

    metadata = json.loads(enriched_results["encoded_metadata"])
    image_path = metadata.get("path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path) or not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        print("The image path is invalid or the file is not a supported image format.")
        return

    # Build a serializable dictionary from enriched_results (skip numpy arrays)
    meta_to_store = {}
    for key, value in enriched_results.items():
        if hasattr(value, "tolist"):
            continue
        try:
            json.dumps(value)
            meta_to_store[key] = value
        except Exception:
            meta_to_store[key] = str(value)
    
    # Load existing EXIF metadata if available
    try:
        exif_dict = piexif.load(image_path)
    except Exception as e:
        print("Failed to load EXIF metadata:", e)
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "Interop":{}, "1st":{}, "thumbnail": None}

    user_comment_tag = piexif.ExifIFD.UserComment
    existing_comment = exif_dict.get("Exif", {}).get(user_comment_tag, b'')
    if existing_comment:
        try:
            decoded_comment = existing_comment.decode('utf-8', errors='ignore').strip('\x00')
            existing_meta = json.loads(decoded_comment)
        except Exception:
            existing_meta = {}
    else:
        existing_meta = {}

    # Update or set "TimesScanned"
    try:
        existing_meta["TimesScanned"] = int(existing_meta.get("TimesScanned", 0)) + 1
    except Exception:
        existing_meta["TimesScanned"] = 1

    # If both meta_to_store and existing_meta have a "metadata" key,
    # Merge new metadata with the existing metadata (if any)
    if "metadata" in meta_to_store and "metadata" in existing_meta:
        merged_metadata = existing_meta["metadata"].copy()
        merged_metadata.update(meta_to_store["metadata"])
        meta_to_store["metadata"] = merged_metadata
    
    # Remove 'metadata' from existing_meta to avoid overwriting the enriched results
    existing_meta.pop("metadata", None)
    
    # Now merge the remaining keys
    meta_to_store.update(existing_meta)
    meta_to_store["Repository"] = "https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision"

    # Convert to JSON and insert back into the image's EXIF UserComment tag.
    metadata_json_str = json.dumps(meta_to_store, indent=4)
    user_comment_bytes = metadata_json_str.encode('utf-8')
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][user_comment_tag] = user_comment_bytes
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)

def detect_bricks(model=None, numpy_image=None, working_folder=os.getcwd(), SAVE_ANNOTATED=False, PLT_ANNOTATED=False, SAVE_JSON=False):
    '''
    Detect bricks in an image using the bricks model.
    
    returns dictionary with enriched results of the detection. yolo outputs have attributes such as:
    orig_img, orig_shape, boxes, probs, obb, speed, names, path plus additional 
    enriched keys (cropped_numpys, annotated_plot, metadata).
    '''
    import os
    import cv2
    import json
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    image_path = None

    if model is None:
        model = load_models()["bricks"]
    if numpy_image is None:
        raise ValueError("Se debe proporcionar la imagen en formato numpy (numpy_image) o una ruta válida.")
    if isinstance(numpy_image, str):
        if not os.path.exists(numpy_image):
            raise ValueError(f"La ruta {numpy_image} no es válida.")
        full_path = os.path.abspath(numpy_image)
        image_from_path = cv2.imread(numpy_image)
        if image_from_path is None:
            raise ValueError("La imagen no se pudo cargar desde la ruta proporcionada.")
        numpy_image = cv2.cvtColor(image_from_path, cv2.COLOR_BGR2RGB)
        image_path = numpy_image

    numpy_image_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    # Realizar la predicción
    results = model.predict(numpy_image_rgb)
    # Si results es una lista (lo que ocurre en algunas versiones), se usa el primer elemento
    if isinstance(results, list):
        results = results[0]

    # Inicializar el diccionario de resultados enriquecidos
    enriched_results = {}
    enriched_results["orig_img"] = numpy_image
    enriched_results["orig_shape"] = numpy_image.shape[:2]

    # Agregar atributos básicos si existen en results
    for attr in ["boxes", "speed", "names", "path"]:
        if hasattr(results, attr):
            enriched_results[attr] = getattr(results, attr)

    # Example to convert YOLO's Boxes object to a serializable format
    if "boxes" in enriched_results:
        try:
            # Assume boxes.xyxy is a tensor or numpy array that can be converted to a list:
            enriched_results["boxes"] = enriched_results["boxes"].xyxy.cpu().numpy().tolist()
        except Exception:
            # Fallback: convert to string representation if conversion fails
            enriched_results["boxes"] = str(enriched_results["boxes"])

    # Extraer y guardar los recortes de ladrillos usando los cuadros (boxes)
    cropped_numpys = {}
    if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
        # Se asume que boxes tiene un atributo xyxy con coordenadas
        # Convirtiendo a lista para iterar
        try:
            boxes_conf = results.boxes.xyxy.cpu().numpy().tolist()
        except Exception:
            # Fallback en caso de que boxes.xyxy no sea accesible
            boxes_conf = [box for box in results.boxes]

        for idx, box in enumerate(boxes_conf):
            x1, y1, x2, y2 = box[:4]
            crop_img = numpy_image[int(y1):int(y2), int(x1):int(x2)]
            cropped_numpys[f"brick_{idx}"] = crop_img
    enriched_results["cropped_numpys"] = cropped_numpys

    # Obtener el plot anotado usando el método plot() de los resultados
    annotated_plot = results.plot() if hasattr(results, "plot") else None
    enriched_results["annotated_plot"] = annotated_plot

    # Extraer metadata sobre el hardware de ejecución y la localizción del equipo si es posible
    metadata = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    enriched_results["metadata"] = metadata
    

    # Configurar carpeta de resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(working_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    # Guardar imagen anotada si se solicita y si existe
    if SAVE_ANNOTATED and annotated_plot is not None:
        annotated_path = os.path.join(results_folder, f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_path, annotated_plot)
        enriched_results["annotated_image_path"] = annotated_path

    # Mostrar imagen anotada usando matplotlib
    if PLT_ANNOTATED and annotated_plot is not None:
        # Convert BGR to RGB for display with matplotlib
        plt.imshow(cv2.cvtColor(annotated_plot, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Guardar resultados enriquecidos en un archivo JSON
    if SAVE_JSON:
        serializable_results = {}
        for key, value in enriched_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()  # Convert numpy arrays to lists
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except (TypeError, OverflowError):
                    serializable_results[key] = str(value)  # Otherwise convert to string
        json_path = os.path.join(results_folder, f"results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        enriched_results["json_results_path"] = json_path

    if image_path is not None:
        enriched_results["path"] = full_path

    # Restructure enriched_results: group raw images and encode the rest as UTF-8 JSON.
    bitmaps = {"orig_img": enriched_results.pop("orig_img", None),
               "cropped_numpys": enriched_results.pop("cropped_numpys", None)}
    encoded_metadata = json.dumps(enriched_results, indent=4, ensure_ascii=False)
    enriched_results = {"bitmaps_numpy": bitmaps, "encoded_metadata": encoded_metadata}

    if full_path is not None:
        write_metadata_to_exif(enriched_results)
    return enriched_results


# Load the bricks model

models = load_models()
bricks_model = models["bricks"] 
studs_model = models["studs"]

# test images folders paths
TEST_IMAGES_FOLDER = {
    "bricks": r"presentation/Test_images/BricksPics",
    "studs": r"presentation/Test_images/StudsPics"
}

# test images URLS

TEST_IMAGES_URLS = {
    "bricks": os.path.join(REPO_URL, TEST_IMAGES_FOLDER["bricks"]),
    "studs": os.path.join(REPO_URL, TEST_IMAGES_FOLDER["studs"])
}

# download all test image as a list of numoy arrays
def download_images(urls=TEST_IMAGES_URLS):
    '''
    Download all test images from the test images URLs
    '''
    images = {}
    for key, url in urls.items():
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        images[key] = np.array(img)
    return images

# download the test images
def download_images_from_url(url):
    # Convert the raw URL into the GitHub API URL for directory listing.
    # Example conversion:
    # "https://raw.githubusercontent.com/MiguelDiLalla/LEGO_Bricks_ML_Vision/main/presentation/Test_images/BricksPics"
    # becomes
    # "https://api.github.com/repos/MiguelDiLalla/LEGO_Bricks_ML_Vision/contents/presentation/Test_images/BricksPics"
    api_url = url.replace("raw.githubusercontent.com", "api.github.com/repos")\
                 .replace("/main/", "/contents/")
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()
    images = []
    for file in files:
        if file["type"] == "file" and file["name"].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            img_response = requests.get(file["download_url"])
            img_response.raise_for_status()
            img = Image.open(BytesIO(img_response.content))
            images.append(np.array(img))
    return images

test_images = {}

for key, url in TEST_IMAGES_URLS.items():
    test_images[key] = download_images_from_url(url)


LOGO_IMAGE_PATH = "presentation/logo.png"
LOGO_IMAGE_URL = os.path.join(REPO_URL, LOGO_IMAGE_PATH)
import requests
from io import BytesIO
from PIL import Image
import platform
import socket

response = requests.get(LOGO_IMAGE_URL)
response.raise_for_status()
# The GitHub API returns a JSON with metadata about the file.
api_data = response.json()
download_url = api_data.get("download_url")
if not download_url:
	raise ValueError("Download URL not found in API response.")
raw_response = requests.get(download_url)
raw_response.raise_for_status()
logo_img = Image.open(BytesIO(raw_response.content))
logo_img

# detect studs

import os
import cv2
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import platform
import socket
import cv2, numpy as np, json, os, datetime


def detect_studs(model=None, numpy_image=None, working_folder=os.getcwd(), SAVE_ANNOTATED=False, PLT_ANNOTATED=True, SAVE_JSON=False):
    '''
    Detect studs in an image using the studs model.
    
    Returns a dictionary with enriched results which contains:
      - orig_img       : Original image (numpy.ndarray)
      - orig_shape     : Tuple with image dimensions (height, width)
      - boxes          : Detection boxes (if available)
      - probs, obb, speed, names, path, as provided by the model prediction
      - annotated_plot : A new annotated image showing the box centers and a regression line (no labels)
      - keypoints      : a dictionary with relative coordinates (values between 0 and 1) representing the centers of bounding boxes
      - metadata       : A dictionary with execution hardware and environment info.
      
    This function can also:
      - Display the annotated image (if PLT_ANNOTATED is True)
      - Save the annotated image and metadata to JSON.
      - Write metadata to the image’s EXIF using write_metadata_to_exif().
    '''
    import cv2, os, json, datetime, numpy as np, matplotlib.pyplot as plt, platform, socket
    # Make sure STUD_TO_DIMENSION_MAP is imported or defined earlier in your code
    # from model_utils import STUD_TO_DIMENSION_MAP

    # Use the studs model if none is provided
    if model is None:
        model = load_models()["studs"]
    if numpy_image is None:
        raise ValueError("Se debe proporcionar la imagen en formato numpy (numpy_image) o una ruta válida.")
    
    full_path = None
    # If a path is provided, load and convert the image
    if isinstance(numpy_image, str):
        if not os.path.exists(numpy_image):
            raise ValueError(f"La ruta {numpy_image} no es válida.")
        full_path = os.path.abspath(numpy_image)
        image_from_path = cv2.imread(numpy_image)
        if image_from_path is None:
            raise ValueError("La imagen no se pudo cargar desde la ruta proporcionada.")
        # Model expects RGB
        numpy_image = cv2.cvtColor(image_from_path, cv2.COLOR_BGR2RGB)
    else:
        # Assume image is already loaded in proper format 
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    
    # Run prediction
    results = model.predict(numpy_image)
    if isinstance(results, list):
        results = results[0]
    
    # Construct the enriched results dictionary
    enriched_results = {}
    enriched_results["orig_img"] = numpy_image
    enriched_results["orig_shape"] = numpy_image.shape[:2]
    
    # Attach basic attributes if available
    for attr in ["boxes", "speed", "names", "path"]:
        if hasattr(results, attr):
            enriched_results[attr] = getattr(results, attr)
    
    # Example to convert YOLO's Boxes object to a serializable format
    if "boxes" in enriched_results:
        try:
            # Assume boxes.xyxy is a tensor or numpy array that can be converted to a list:
            enriched_results["boxes"] = enriched_results["boxes"].xyxy.cpu().numpy().tolist()
        except Exception:
            # Fallback: convert to string representation if conversion fails
            enriched_results["boxes"] = str(enriched_results["boxes"])

    # Calculate keypoints: centers of each bounding box (both absolute and relative)
    keypoints = {}
    abs_centers = []  # For regression, we need absolute coordinates
    boxes_conf = []
    if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
        try:
            boxes_conf = results.boxes.xyxy.cpu().numpy().tolist()
        except Exception:
            boxes_conf = [box for box in results.boxes]
        h, w = numpy_image.shape[:2]
        for idx, box in enumerate(boxes_conf):
            x1, y1, x2, y2 = box[:4]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            abs_centers.append((center_x, center_y))
            # Relative coordinates in [0,1]
            keypoints[f"stud_{idx}"] = (center_x / w, center_y / h)
    enriched_results["keypoints"] = keypoints

    # Create a new annotated image based on the original image:
    annotated_plot = numpy_image.copy()
    # Draw the center of each box (filled red circle)
    for center in abs_centers:
        cv2.circle(annotated_plot, (int(center[0]), int(center[1])), 4, (255, 0, 0), thickness=-1)
    
    # If there are at least 2 centers, compute a regression line and overlay it
    if len(abs_centers) > 1:
        abs_centers_arr = np.array(abs_centers)
        xs = abs_centers_arr[:, 0]
        ys = abs_centers_arr[:, 1]
        m, b = np.polyfit(xs, ys, 1)
        # Use full width of image for endpoints
        x_start, x_end = 0, w
        y_start = int(m * x_start + b)
        y_end = int(m * x_end + b)
        cv2.line(annotated_plot, (x_start, y_start), (x_end, y_end), (0, 255, 0), thickness=2)
    
    enriched_results["annotated_plot"] = annotated_plot

    # --- NEW LOGIC FOR STUD DIMENSION CLASSIFICATION ---
    if boxes_conf:
        num_studs = len(boxes_conf)
        print("[DEBUG] Number of studs:", num_studs)
        if num_studs not in STUD_TO_DIMENSION_MAP:
            print(f"[ERROR] Deviant number of studs detected ({num_studs}). Skipping classification.")
            enriched_results["DIMENSION"] = "Unknown"
        else:
            # Extract stud center coordinates from boxes_conf
            centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes_conf]
            print("[DEBUG] Centers:", centers)

            # Compute mean bounding box size for spacing calculation
            box_sizes = [((box[2] - box[0] + box[3] - box[1]) / 2.0) for box in boxes_conf]
            print("[DEBUG] Box sizes:", box_sizes)

            # Fit a regression line using the stud centers
            xs, ys = zip(*(centers))
            m, b = np.polyfit(xs, ys, 1)
            print("[DEBUG] Regression m, b:", m, b)

            # Compute deviation from the regression line for each center
            deviations = [abs(y - (m * x + b)) for x, y in centers]
            print("[DEBUG] Deviations:", deviations)

            # Decision boundary based on the max deviation and half the mean box size
            threshold = np.mean(box_sizes) / 2.0
            print("[DEBUG] Threshold:", threshold)

            classification_aux = "Nx1" if max(deviations) < threshold else "Nx2"
            print("[DEBUG] Classification auxiliary:", classification_aux)

            # Determine final brick dimension using STUD_TO_DIMENSION_MAP
            possible_dimensions = STUD_TO_DIMENSION_MAP.get(num_studs, "Unknown")
            print("[DEBUG] Possible dimensions:", possible_dimensions)
            if possible_dimensions != "Unknown":
                if isinstance(possible_dimensions, list):
                    new_name = possible_dimensions[0] if classification_aux == "Nx2" else possible_dimensions[1]
                else:
                    new_name = possible_dimensions
                print("[DEBUG] Final determined new_name:", new_name)
                enriched_results["DIMENSION"] = new_name
            else:
                enriched_results["DIMENSION"] = "Unknown"
    else:
        # Handle case when no studs (boxes) are detected
        enriched_results["DIMENSION"] = "No studs detected"
    # --- END NEW LOGIC ---

    # Add hardware and environment metadata
    enriched_results["metadata"] = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Configure results folder and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(working_folder, "results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Save annotated image if required
    if SAVE_ANNOTATED:
        annotated_path = os.path.join(results_folder, f"annotated_{timestamp}.jpg")
        # Save as BGR image
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_plot, cv2.COLOR_RGB2BGR))
        enriched_results["annotated_image_path"] = annotated_path

    # Plot annotated image if requested
    if PLT_ANNOTATED:
        plt.imshow(annotated_plot)
        plt.axis('off')
        plt.show()
    
    # Save JSON version of results (omitting non-serializable types)
    if SAVE_JSON:
        serializable_results = {}
        for key, value in enriched_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()  # Convert numpy arrays to lists
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except (TypeError, OverflowError):
                    serializable_results[key] = str(value)  # Otherwise convert to string
        json_path = os.path.join(results_folder, f"results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        enriched_results["json_results_path"] = json_path
    
    if full_path is not None:
        enriched_results["path"] = full_path

    # Restructure enriched_results for studs. Note: studs have only 'orig_img'
    bitmaps = {"orig_img": enriched_results.pop("orig_img", None)}
    encoded_metadata = json.dumps(enriched_results, indent=4, ensure_ascii=False)
    enriched_results = {"bitmaps_numpy": bitmaps, "encoded_metadata": encoded_metadata}

    if full_path is not None:
        write_metadata_to_exif(enriched_results)
    
    return enriched_results


LOGO_IMAGE_PATH = "presentation/logo.png"
LOGO_IMAGE_URL = os.path.join(REPO_URL, LOGO_IMAGE_PATH)
import requests
from io import BytesIO
from PIL import Image

response = requests.get(LOGO_IMAGE_URL)
response.raise_for_status()
# The GitHub API returns a JSON with metadata about the file.
api_data = response.json()
download_url = api_data.get("download_url")
if not download_url:
	raise ValueError("Download URL not found in API response.")
raw_response = requests.get(download_url)
raw_response.raise_for_status()
logo_img = Image.open(BytesIO(raw_response.content))
# logo_img



# function to plot and save enriched/branded bricks detection

def bricks_image_composer(bricks_results, logo_image=logo_img, save_path=None):
    '''
    Given a dictionary of enriched bricks detection results and an optional logo image,
    this function composes a results image with a red background and a right-side bar that
    renders selected enriched results in text. The logo is then placed in the bottom-right
    corner of the composed image.
    
    If a save_path is not provided, the composed image will be saved in a "results" folder
    under the current working directory.
    
    Returns the composed image as a numpy array.
    '''
