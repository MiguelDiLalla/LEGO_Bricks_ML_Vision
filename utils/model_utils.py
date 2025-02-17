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
    Load bricks model and studs model from fixed github URLs.
    return a dictionary with the Yolo models ready to use
    '''
    
    import tempfile
    import requests
    from ultralytics import YOLO

    models = {}
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    for key, url in urls.items():
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()
            temp_filepath = tmp.name
        model = YOLO(temp_filepath)
        models[key] = model
        models[key] = model
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
    if enriched_results is None:
        return
    if enriched_results.get("path") is None:
        return
    image_path = enriched_results.get("path")
    if not image_path or not os.path.isfile(image_path) or not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        print("The image path is invalid or the file is not a supported image format.")
        return

    # Build a serializable dictionary from enriched_results (skip numpy arrays)
    meta_to_store = {}
    for key, value in enriched_results.items():
        # Check for numpy arrays by the presence of tolist()
        if hasattr(value, "tolist"):
            continue
        try:
            json.dumps(value)
            meta_to_store[key] = value
        except Exception:
            meta_to_store[key] = str(value)
    
    # Try to load the current EXIF data
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

    # Update or set the "TimesScanned" key
    if "TimesScanned" in existing_meta:
        try:
            existing_meta["TimesScanned"] = int(existing_meta["TimesScanned"]) + 1
        except Exception:
            existing_meta["TimesScanned"] = 1
    else:
        existing_meta["TimesScanned"] = 1
    
    # Merge new metadata with the existing metadata (existing_meta takes precedence for TimesScanned)
    meta_to_store.update(existing_meta)
    # Add repository information
    meta_to_store["Repository"] = "https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision"

    # Convert metadata to JSON string and encode to bytes (UserComment expects bytes)
    metadata_json_str = json.dumps(meta_to_store, indent=4)
    user_comment_bytes = metadata_json_str.encode('utf-8')

    # Ensure the Exif section exists and update UserComment tag
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][user_comment_tag] = user_comment_bytes
    # Dump updated exif data and insert it back into the image file
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)

def detect_bricks(model=None, numpy_image=None, working_folder=os.getcwd(), SAVE_ANNOTATED=False, PLT_ANNOTATED=False, SAVE_JSON=False):
    '''
    Detect bricks in an image using the bricks model.
    
    returns dictionary with enriched results of the detection. yolo outputs :

      orig_img       numpy.ndarray  La imagen original como un array numpy.
      orig_shape     tuple          La forma original de la imagen en formato (alto, ancho).
      boxes          Boxes, optional  Un objeto Boxes que contiene las cajas delimitadoras de la detección.
      probs          Probs, optional  Un objeto Probs que contiene las probabilidades de cada clase para la tarea de clasificación.
      obb            OBB, optional  Un objeto OBB que contiene cuadros delimitadores orientados.
      speed          dict           Un diccionario de velocidades de preprocesamiento, inferencia y postprocesamiento en milisegundos por imagen.
      names          dict           Un diccionario de nombres de clases.
      path           str            La ruta al archivo de imagen.

    Además almacenamos los mismos datos en our_results y se añaden las siguientes claves usando las propiedades de la clase de resultados de ultralitycs (https://docs.ultralytics.com/es/modes/predict/):

      - cropped_numpys: dict of numpy arrays with the cropped bricks. Primero se generan imágenes .jpg usando el método save_crop().
      - annotated_plot: imagen anotada usando el método .plot() del objeto results.
      - metadata: un diccionario anidado con toda la metadata posible extraída del objeto results acerca del entorno y el modelo.

    La función puede:
      - Mostrar la imagen anotada con o3-mini .plot()
      - Guardar imágenes anotadas en la carpeta CWD/results/ con timestamp y nombre simple.
      - Guardar los resultados enriquecidos a un archivo JSON en la carpeta CWD/results/ con timestamp y nombre simple.
    '''
    # Validación de parámetros

    image_path = None

    if model is None:
        model = load_models()["bricks"]
    if numpy_image is None:
        raise ValueError("Se debe proporcionar la imagen en formato numpy (numpy_image) o una ruta válida.")
    if isinstance(numpy_image, str):
        if not os.path.exists(numpy_image):
            raise ValueError(f"La ruta {numpy_image} no es válida.")
        image_from_path = cv2.imread(numpy_image)
        if image_from_path is None:
            raise ValueError("La imagen no se pudo cargar desde la ruta proporcionada.")
        numpy_image = image_from_path
        img_path = numpy_image
    
    # Realizar la predicción
    results = model.predict(numpy_image)
    
    # Inicializar el diccionario de resultados enriquecidos
    enriched_results = {}
    enriched_results["orig_img"] = numpy_image
    enriched_results["orig_shape"] = numpy_image.shape[:2]
    
    # Asignar claves básicas si están presentes en results
    if hasattr(results, 'boxes'):
        enriched_results["boxes"] = results.boxes
    if hasattr(results, 'probs'):
        enriched_results["probs"] = results.probs
    if hasattr(results, 'obb'):
        enriched_results["obb"] = results.obb
    if hasattr(results, 'speed'):
        enriched_results["speed"] = results.speed
    if hasattr(results, 'names'):
        enriched_results["names"] = results.names
    if hasattr(results, 'path'):
        enriched_results["path"] = results.path

    # Extraer y guardar los recortes de ladrillos
    import io
    import os
    import cv2
    import numpy as np

    cropped_numpys = {}

    # If the model supports save_crop but you want to use in-memory buffers, you would need the API to support bytes output.
    # Here we assume that the API does not, so we use the fallback manual cropping approach with BytesIO.
    if hasattr(results, 'save_crop') and True:    # Change False to True if the API supports in-memory bytes output
        # Example hypothetical API usage:
        crop_bytes_list = results.save_crop(save_format="bytes")  # This is a hypothetical parameter!
        for idx, crop_data in enumerate(crop_bytes_list):
            # Wrap the bytes in a BytesIO object
            crop_bytes_io = io.BytesIO(crop_data)
            # Convert the BytesIO buffer to a numpy array and decode the image with OpenCV
            crop_array = np.frombuffer(crop_bytes_io.getbuffer(), dtype=np.uint8)
            crop_img = cv2.imdecode(crop_array, cv2.IMREAD_COLOR)
            cropped_numpys[f"brick_{idx}"] = crop_img
    else:
        # Fallback: perform manual cropping from the numpy image using the detection boxes
        if "boxes" in enriched_results and enriched_results["boxes"] is not None:
            for idx, box in enumerate(enriched_results["boxes"]):
                # Assuming box is defined as (x1, y1, x2, y2) in absolute coordinates
                x1, y1, x2, y2 = box
                crop_img = numpy_image[int(y1):int(y2), int(x1):int(x2)]
                # Encode the cropped image to JPEG format into memory
                success, buffer = cv2.imencode('.jpg', crop_img)
                if success:
                    # Wrap the encoded bytes into a BytesIO object
                    bytes_io = io.BytesIO(buffer)
                    # Optionally, decode back to a numpy array (if you need to work on the crop)
                    crop_array = np.frombuffer(bytes_io.getbuffer(), dtype=np.uint8)
                    crop_decoded = cv2.imdecode(crop_array, cv2.IMREAD_COLOR)
                    cropped_numpys[f"brick_{idx}"] = crop_decoded

    enriched_results["cropped_numpys"] = cropped_numpys

    # Obtener el plot anotado de la imagen usando el método plot() (se espera que retorne una imagen anotada en formato numpy)
    annotated_plot = results.plot() if hasattr(results, "plot") else None
    enriched_results["annotated_plot"] = annotated_plot

    # Extraer metadata extra disponible del objeto results
    metadata = {}
    for attr in ['device', 'version', 'time']:
        if hasattr(results, attr):
            metadata[attr] = getattr(results, attr)
    enriched_results["metadata"] = metadata

    # Configurar carpeta de resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(working_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    # Guardar imagen anotada si se solicita y si existe la imagen anotada
    if SAVE_ANNOTATED and annotated_plot is not None:
        annotated_path = os.path.join(results_folder, f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_path, annotated_plot)
        enriched_results["annotated_image_path"] = annotated_path

    # Mostrar imagen anotada usando matplotlib
    if PLT_ANNOTATED and annotated_plot is not None:
        plt.imshow(cv2.cvtColor(annotated_plot, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Guardar resultados enriquecidos en un archivo JSON
    if SAVE_JSON:
        # Debido a que algunos datos pueden no ser serializables (listas, arrays, objetos), se convierten a cadena mediante str()
        serializable_results = {}
        for key, value in enriched_results.items():
            try:
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                serializable_results[key] = str(value)
        json_path = os.path.join(results_folder, f"results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        enriched_results["json_results_path"] = json_path

    if image_path:
        enriched_results["path"] = image_path
        write_metadata_to_exif(enriched_results)
    return enriched_results

# --------------------------------------

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
    print()
    test_images[key] = download_images_from_url(url)


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
logo_img
