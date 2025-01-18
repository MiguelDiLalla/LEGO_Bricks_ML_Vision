# Introduction and Motivation

I have always been passionate about Lego, ever since I was young. The act of taking a seemingly chaotic pile of pieces and transforming them into something meaningful fascinated me then and continues to inspire me now. This project began as a way to combine that lifelong enthusiasm for Lego with my growing expertise in data science and computer vision. It has since evolved into something much more: a demonstration of how humans can teach intelligent systems the kind of nuanced recognition tasks that we perform almost instinctively.

The challenge is relatable for any Lego fan. Often, we don’t have a neatly organized collection—just a large container of bricks spilled out on a surface during a building session. Yet, despite the overwhelming mix of shapes and colors, we’re able to quickly identify the exact piece we need. I wanted to see if I could build a model capable of performing a similar task: detecting individual Lego pieces in a cluttered image and understanding their unique dimensions. This seemingly simple task encapsulates what makes human perception so sophisticated—and what makes replicating it with AI both challenging and rewarding.

My hope is that this project demonstrates my ability to break down complex problems, work through challenging datasets, and design scalable solutions. It also represents a potential building block for future technology—imagine a robotic assistant capable of organizing or assisting with assembly by recognizing pieces in real-time. While that’s an ambitious goal, this project serves as a foundation, a proof of concept for a potential future application. Ultimately, I want to showcase not only my technical skills but also my creativity, perseverance, and genuine curiosity about how AI can make our world more intuitive and efficient.




# Problem Definition

The problem at the heart of this project is one that many Lego enthusiasts know well: the challenge of locating specific pieces within a large, unorganized collection. During a building session without a neatly sorted inventory, fans often spill all their pieces out onto a surface and rely on their innate ability to visually scan through the chaos to find what they need. This seemingly effortless task for a human actually involves sophisticated cognitive skills like pattern recognition, color differentiation, and understanding shape and size—all within a cluttered environment.

The goal of this project was to create an AI model that could mimic this human ability: detecting individual Lego pieces within a mixed collection, identifying them, and determining their unique dimensions. Although this task seems simple, it reveals a complex problem when approached from an AI perspective. It involves not just identifying objects in an image but doing so with a level of detail that distinguishes between similar-looking items based on dimensions, proportions, and subtle differences in features.

To manage the complexity, I introduced a bottleneck by focusing on a limited set of Lego pieces. Rather than attempting to recognize every type of Lego piece, I decided to focus specifically on opaque bricks and tiles, excluding flat pieces and more specialized shapes. This decision allowed me to narrow the scope to 26 classes, each with distinct combinations of dimensions. The idea was to lay the groundwork for a scalable solution that could eventually be expanded to handle the full diversity of Lego pieces.

The broader vision is that if this model succeeds in recognizing and categorizing these limited types of Lego pieces, the same pipeline could be extended and adapted in the future for a wider variety of pieces and even other applications. Ultimately, this project serves as an exploration of how to bridge the gap between human visual perception and machine learning capabilities in a cluttered, real-world environment.



## Import libraries:


```python
import os
import json
import random
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, text, patheffects
from collections import defaultdict
import math

from PIL import Image
import shutil

import labelme

import importlib.util
import yaml

from ultralytics import YOLO

from pathlib import Path

import torch
from datetime import datetime
```

# Initial Data Collection

To begin this project, I knew that collecting a robust dataset was essential. My goal was to develop a model capable of identifying and classifying Lego pieces, so I needed diverse, real-world data. Given my sizable Lego collection, I chose to create the dataset from scratch, focusing on opaque bricks and tiles. I narrowed the scope to 26 distinct classes, excluding specialized, flat, and complex shapes.
For data collection, I scattered Lego samples across different surfaces in my flat to capture various lighting conditions and backgrounds. I rearranged the pieces three times, conducting photoshoots with my phone to capture different angles and zoom levels, resulting in over 2,000 images.

Next, I annotated the dataset using LabelMe, drawing bounding boxes around each piece. Though time-consuming, this step was crucial for ensuring model accuracy. Creating this dataset taught me the importance of careful planning and highlighted the effort required for data collection and annotation—an essential foundation for reliable machine learning results.




```python
spilt_bricks_raw_images_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images"

resized_spilt_bricks_raw_images_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized"
```


```python
# given a folder with images. create a 3x7 grid with square cells. random images are selected and showcased in the grid.

def showcase_dataset_images(folder_path):
    # Get all .jpg files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # Ensure there are at least 21 images available
    if len(image_files) < 21:
        print(f"The folder must contain at least 21 .jpg images. Found: {len(image_files)}")
        return

    # Randomly select 21 images from the folder
    selected_images = random.sample(image_files, 21)
    
    # Create a 3x7 grid for displaying the images
    fig, axes = plt.subplots(3, 7, figsize=(14, 6), facecolor='black')
    
    # Loop through each selected image and add it to the grid
    for idx, img_name in enumerate(selected_images):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        
        # Resize image to be square
        img = img.resize((150, 150))
        
        # Calculate row and column index for the grid
        row = idx // 7
        col = idx % 7
        
        # Display the image in the corresponding subplot
        axes[row, col].imshow(img)
        axes[row, col].axis('off')  # Hide the axes for a cleaner look
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
```


```python
showcase_dataset_images(resized_spilt_bricks_raw_images_folder)
```


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_7_0.png)
    



```python
# given a folder of images, normalize the images names. ex: "image_0034.jpg"

def normalize_images_names(folder_path):
    for i, filename in enumerate(os.listdir(folder_path)):
        os.rename(folder_path + "\\" + filename, folder_path + "\\" + f"image_{str(i).zfill(4)}.jpg")

# normalize_images_names(spilt_bricks_raw_images_folder)
```


```python
# Given an origin folder, an output folder, and a target size, resize all images in the origin folder to the target size and save them in the output folder.

def resize_images(origin_folder, output_folder, factor):
    """
    Resize all images in the origin folder by a given factor and save them in the output folder.

    Parameters:
    origin_folder (str): Path to the folder containing the original images.
    output_folder (str): Path to the folder where resized images will be saved.
    factor (float): Factor by which to resize images, e.g., 0.5 for half size, 2 for double size.

    Returns:
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(origin_folder):
        if filename.endswith(".jpg"):
            try:
                image_path = os.path.join(origin_folder, filename)
                with Image.open(image_path) as image:
                    new_size = (int(image.width * factor), int(image.height * factor))
                    resized_image = image.resize(new_size)
                    resized_image.save(os.path.join(output_folder, filename))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
```


```python
# the images sizes present in raw images folder

def get_image_sizes(folder):
    """
    Get the sizes of all images in a folder.

    Parameters:
    folder (str): Path to the folder containing the images.

    Returns:
    list: List of unique image sizes present in the folder.
    """
    sizes = set()
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            with Image.open(os.path.join(folder, filename)) as image:
                sizes.add(image.size)
    return list(sizes)

# get the image sizes of the raw images folder

# get_image_sizes(spilt_bricks_raw_images_folder)

```


```python
# number of images in the raw images folder

def get_number_of_images(folder):
    """
    Get the number of images in a folder.

    Parameters:
    folder (str): Path to the folder containing the images.

    Returns:
    int: Number of images in the folder.
    """
    return len([filename for filename in os.listdir(folder) if filename.endswith(".jpg")])

get_number_of_images(spilt_bricks_raw_images_folder)
```




    0




```python
resize_factor = 800 / 4000 # 800 is the target size, 4000 is the original size, factor is 1/5

# resize the images in the raw images folder

# resize_images(spilt_bricks_raw_images_folder, resized_spilt_bricks_raw_images_folder, resize_factor)
```


```python
# Labelme must be instll in the enviroment and executed from bash.

# !Labelme
```

During the annotation process blurred images remained un-touch. their lack of corresponding .json file can be used to remove them from the data in batch:


```python

def remove_invalid_jpg_files(folder_path):
    """
    Remove .jpg files from the folder that are not in the list of valid files.

    Parameters:
    folder_path (str): Path to the folder containing the images and labelme files.

    Returns:
    None
    """
    # get the list of json labelme files

    labelme_files = [filename for filename in os.listdir(resized_spilt_bricks_raw_images_folder) if filename.endswith(".json")]

    # turn .json extension to .jpg

    jpg_valid_files = [filename.replace(".json", ".jpg") for filename in labelme_files]

    # delete jpg files that are not valid

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") and filename not in jpg_valid_files:
            os.remove(os.path.join(folder_path, filename))

# Example usage
# remove_invalid_jpg_files(resized_spilt_bricks_raw_images_folder)

```

# Model Selection

When selecting a model for this project, I needed one that could efficiently detect and classify Lego pieces within a cluttered image. My primary focus was on finding a solution that balanced accuracy with ease of use, while also requiring minimal setup time. This allowed me to focus more on data preparation and analysis rather than configuring a complex model. After considering several options, I chose YOLO (You Only Look Once), specifically version 8n, due to its straightforward setup, speed, and accuracy. YOLO's real-time detection capabilities and simple implementation made it ideal for identifying multiple Lego pieces within a single image, even when those pieces were scattered and partially occluded.

YOLO's architecture is particularly well-suited for object detection tasks where speed is crucial and the ability to work in real-world, cluttered environments is essential. Unlike other models that might require significant computational power or intricate pre-processing steps, YOLO is designed to be efficient and can perform detection in just one pass through the neural network. This made it an ideal choice for my hardware setup, which has limited GPU capabilities, while also allowing for rapid iteration during the development phase.

Using YOLO, I trained a model capable of reliably detecting individual bricks and tiles across various lighting conditions and backgrounds. Once trained, I developed a simple script to crop each detected Lego piece from the image. This cropping script was essential because it allowed me to isolate each piece for further analysis. Processing the pieces individually simplified the classification task and provided more control over the input data for subsequent steps. This approach laid a solid foundation for further classification and more detailed analysis, paving the way for distinguishing features such as dimensions, color, and brick type.




```python
# given a folder with labelme json files, convert them to yolo format txt files
def labelme_jsons_to_yolos(folder):
    """
    Convert LabelMe JSON files to YOLO format text files.

    Parameters:
    folder (str): Path to the folder containing LabelMe JSON files.

    Returns:
    None
    """
    
    def get_image_size_from_json(json_file):
        """
        Extract image size from a LabelMe JSON file.

        Parameters:
        json_file (str): Path to the LabelMe JSON file.

        Returns:
        tuple: Image height and width.
        """
        with open(json_file, "r") as file:
            data = json.load(file)
        return data["imageHeight"], data["imageWidth"]
    
    def labelme_json_to_yolo(json_file, yolo_file, image_size):
        """
        Convert a single LabelMe JSON file to YOLO format.

        Parameters:
        json_file (str): Path to the LabelMe JSON file.
        yolo_file (str): Path to the output YOLO format text file.
        image_size (tuple): Image height and width.

        Returns:
        None
        """
        with open(json_file, "r") as file:
            data = json.load(file)

        with open(yolo_file, "w") as file:
            for shape in data["shapes"]:
                points = shape["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x1, y1, x2, y2 = x1 / image_size[1], y1 / image_size[0], x2 / image_size[1], y2 / image_size[0]
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                file.write(f"0 {x} {y} {width} {height}\n")

    def fix_negative_values(labels_dir: str) -> None:
        """
        Fix negative values in YOLO annotations.

        Args:
            labels_dir (str): Path to the directory containing the YOLO label files.

        Returns:
            None
        """

        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        for label_file in label_files:
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()

            with open(os.path.join(labels_dir, label_file), 'w') as f:
                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    if width < 0:
                        width = abs(width)
                    if height < 0:
                        height = abs(height)
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            json_file = os.path.join(folder, filename)
            yolo_file = os.path.join(folder, filename.replace(".json", ".txt"))
            image_size = get_image_size_from_json(json_file)
            labelme_json_to_yolo(json_file, yolo_file, image_size)

    fix_negative_values(folder)

```


```python
# labelme_jsons_to_yolos(resized_spilt_bricks_raw_images_folder)
```


```python
# Function to plot a grid of annotated images

def visualize_yolo_annotated_images(image_folder, annotation_folder, num_images=6, specific_files=[], class_names=["LegoBrick"]):
    """
    Visualize a given number of random images annotated with YOLO bounding boxes.

    Parameters:
    image_folder (str): Path to the folder containing images.
    annotation_folder (str): Path to the folder containing YOLO annotation files.
    num_images (int): Number of images to visualize. Default is 6.
    specific_files (list): List of specific image files to visualize. Default is empty list.
    class_names (list): List of class names corresponding to class IDs. Default is ["LegoBrick"].

    Returns:
    None
    """

    # Function to read YOLO annotations

    def read_yolo_annotation(annotation_path, image_width, image_height):
        """
        Read YOLO annotation file and convert normalized bounding box coordinates to pixel values.

        Parameters:
        annotation_path (str): Path to the YOLO annotation file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

        Returns:
        list: List of bounding boxes with pixel coordinates and class IDs.
        """
        boxes = []
        with open(annotation_path, 'r') as file:
            for line in file:
                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # Convert normalized values to actual pixel values
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height

                # Get coordinates for the bounding box
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                boxes.append((x_min, y_min, x_max, y_max, int(class_id)))
        return boxes

    # Get list of images and annotations
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith('.txt')]
    
    # Sort to match images and annotations correctly
    image_files.sort()
    annotation_files.sort()

    # Shuffle and select a random subset of images and annotations
    combined_files = list(zip(image_files, annotation_files))
    random.shuffle(combined_files)

    # If specific files are provided, filter to those files and add them on top
    if specific_files:
        specific_combined_files = [(img, ann) for img, ann in zip(image_files, annotation_files) if img in specific_files]
        combined_files = specific_combined_files + combined_files

    # Select the final subset of images and annotations
    combined_files = combined_files[:num_images]

    # Set up the plot grid
    num_images = min(num_images, len(combined_files))
    cols = 3
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.patch.set_facecolor('black')  # Set figure background to black
    axes = axes.flatten()

    # Loop through the selected number of images
    for i in range(num_images):
        image_file, annotation_file = combined_files[i]
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        # Load image
        image = Image.open(image_path)
        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        width, height = image.size

        # Read annotations
        boxes = read_yolo_annotation(annotation_path, width, height)

        # Draw bounding boxes
        for (x_min, y_min, x_max, y_max, class_id) in boxes:
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_cv, str(class_names[class_id]), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Plot the image with annotations
        axes[i].imshow(image_cv)
        axes[i].set_title(image_file, color='yellow')
        axes[i].axis('off')
        axes[i].set_facecolor('black')

    # Hide any extra subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


```


```python
# Example usage
visualize_yolo_annotated_images(resized_spilt_bricks_raw_images_folder, resized_spilt_bricks_raw_images_folder, num_images=6, 
                                # specific_files=["image_0.jpg", "image_1.jpg"]
                                )
```


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_20_0.png)
    



```python
# a function to prepare a YOLO dataset structure 
# by listing all image and label files in the origin folder,
# creating the required YOLO training directory structure, 
# and performing an 80-20 train-val split. Also creates the dataset.yaml file required for YOLO training.

def prepare_yolo_dataset(origin_folder, output_folder, class_names = ['LegoBrick']):
    """
    Prepares a YOLO dataset structure by listing all image and label files in the origin folder,
    creating the required YOLO training directory structure, and performing an 80-20 train-val split.
    Also creates the dataset.yaml file required for YOLO training.

    Args:
        origin_folder (str): Path to the origin folder containing images and YOLO formatted label files.
        output_folder (str): Path to the output folder where YOLO dataset structure will be created.
        class_names (list): List of class names for the dataset. Default is ['stud'].
    """
    # List all image and label files in the origin folder
    image_files = [f for f in os.listdir(origin_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(origin_folder) if f.endswith('.txt')]

    # Pair image files with their corresponding label files
    paired_files = []
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file in label_files:
            paired_files.append((image_file, label_file))

    if len(paired_files) == 0:
        print("No matching image-label pairs found in the origin folder.")
        return

    # Create YOLO directory structure if it doesn't exist, or clear it if it does
    data_folder = os.path.join(output_folder, 'data')
    train_images_folder = os.path.join(data_folder, 'train', 'images')
    train_labels_folder = os.path.join(data_folder, 'train', 'labels')
    val_images_folder = os.path.join(data_folder, 'val', 'images')
    val_labels_folder = os.path.join(data_folder, 'val', 'labels')

    for folder in [train_images_folder, train_labels_folder, val_images_folder, val_labels_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Remove existing folder and its contents
        os.makedirs(folder)  # Create the required folder

    # Shuffle and split the files into train and validation sets (80-20 split)
    random.shuffle(paired_files)
    split_index = int(len(paired_files) * 0.8)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    # Copy the files to the appropriate train/val directories
    for image_file, label_file in train_files:
        shutil.copy(os.path.join(origin_folder, image_file), train_images_folder)
        shutil.copy(os.path.join(origin_folder, label_file), train_labels_folder)

    for image_file, label_file in val_files:
        shutil.copy(os.path.join(origin_folder, image_file), val_images_folder)
        shutil.copy(os.path.join(origin_folder, label_file), val_labels_folder)

    # Create or clean the dataset.yaml file /data/dataset.yaml"
    yaml_path = os.path.join(data_folder, 'dataset.yaml')
    if os.path.exists(yaml_path):
        os.remove(yaml_path)  # Remove the existing yaml file if it exists

    # Number of classes
    num_classes = len(class_names)

    # YAML dictionary structure
    dataset_yaml = {
        'train': os.path.join(data_folder, 'train', 'images'),
        'val': os.path.join(data_folder, 'val', 'images'),
        'nc': num_classes,
        'names': class_names
    }

    # Write YAML file
    with open(yaml_path, 'w') as file:
        yaml.dump(dataset_yaml, file, default_flow_style=False)

    # Summary report
    print("\nYOLO Dataset Preparation Summary:")
    print(f"- Training set: {len(train_files)} images and labels")
    print(f"- Validation set: {len(val_files)} images and labels")
    print(f"- Dataset YAML file created at: {yaml_path}")

    return yaml_path
```


```python
# Example usage
origin_folder = resized_spilt_bricks_raw_images_folder  # Replace with the path to your origin folder
output_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\YOLO_finetune"  # Replace with the path to your output folder
yaml_brick_detectron_path = prepare_yolo_dataset(origin_folder, output_folder)  # Run the function to prepare the dataset

```

    
    YOLO Dataset Preparation Summary:
    - Training set: 1442 images and labels
    - Validation set: 361 images and labels
    - Dataset YAML file created at: C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\YOLO_finetune\data\dataset.yaml
    


```python
def YOLO_oneLabel_train(YAML_file_path, Imgsz, label = "LegoBrick", epochs = 150, model = YOLO('yolov8n.pt')):

    # Train the model using the dataset.yaml file
    results = model.train(
        data= YAML_file_path,
        epochs=epochs,
        imgsz= Imgsz, # IMPORTANT, DOUBLE CHECK THE IMAGE SIZE
        batch=16,
        lr0=0.001,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        mosaic=0.5,
        auto_augment='randaugment',
        mixup=0.2,
        # a name that resembles the label, the base model and the moment of training
        name = f'YOLOdetectron_{label}_{model.model_name.split(".")[0]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # run model evaulation
    metrics = model.val()
    
    # save the model to the parent directory from the Yaml file folder

    parent_dir = Path(YAML_file_path).parent.parent

    model.save(parent_dir / f'YOLOdetectron_{label}_{model.model_name.split(".")[0]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')

    return model


```


```python

# Bricks_model = YOLO_oneLabel_train(yaml_brick_detectron_path, 800, "LegoBrick", 50)
Brick_model_path = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Models\YOLO_LegoBrick_Detectron_v0.pt"
Bricks_model = YOLO(Brick_model_path)
```


```python
# given a image path and a model, run de predicction and plot the annotated image

def predict_and_plot(image_path, model, class_names=["LegoBrick"], conf_threshold=0.5):
    """
    Run prediction on a single image and plot the annotated image.

    Parameters:
    image_path (str): Path to the image file.
    model: YOLO model object.
    class_names (list): List of class names corresponding to class IDs. Default is ["LegoBrick"].
    """

    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for plotting

    # Get predictions from the model
    results = model.predict(source=image_path, save=False, show=False, imgsz=640, conf=conf_threshold)  # You can adjust the confidence threshold here

    # Extract predictions (bounding boxes and labels)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Plot the image with annotations
    plt.figure(figsize=(10, 10)).patch.set_facecolor('black')
    plt.imshow(img_rgb)

    # Plot bounding boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='black', facecolor='none'))
        plt.text(x1, y1 - 10, f'{class_names[0]} | Conf: {score:.2f}', color='yellow', fontsize=12, backgroundcolor='black')

    plt.axis('off')
    #plot base color black
    
    plt.title(f'{os.path.basename(image_path)}', color='yellow')
    plt.show()
```


```python
# predict and plot a random image in resized_spilt_bricks_raw_images_folder

image_files = [f for f in os.listdir(resized_spilt_bricks_raw_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
random_image = random.choice(image_files)

predict_and_plot(os.path.join(resized_spilt_bricks_raw_images_folder, random_image), Bricks_model)
```

    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1696.jpg: 480x640 6 LegoBricks, 230.4ms
    Speed: 11.0ms preprocess, 230.4ms inference, 18.0ms postprocess per image at shape (1, 3, 480, 640)
    


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_26_1.png)
    



```python
# given a origin folder,a an output folder, and a YOLO model. predict the bounding boxes of the images in the origin folder and save each box from each image in the output folder.
# naming for the new files {origon_file_name}_{box_identifier}_c{confidense_percentage_withoun_decimal_point}.jpg

def crop_bounding_boxes(origin_folder, output_folder, model, class_names=["LegoBrick"], conf_threshold=0.50):
    """
    Crop bounding boxes from images using a YOLO model and save them as separate images.

    Parameters:
    origin_folder (str): Path to the folder containing images.
    output_folder (str): Path to the folder where cropped images will be saved.
    model (YOLO): YOLO model object.
    class_names (list): List of class names corresponding to class IDs. Default is ["LegoBrick"].
    conf_threshold (float): Confidence threshold for object detection. Default is 0.50.

    Returns:    
    None
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image files
    image_files = [f for f in os.listdir(origin_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(origin_folder, image_file)
        image = Image.open(image_path)
        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        width, height = image.size

        # Perform inference
        results = model(image_path, conf=conf_threshold)

        # Get bounding boxes and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        labels = results[0].boxes.cls.cpu().numpy()  # Class labels
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

        # save cropped images

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x_min, y_min, x_max, y_max = box
            class_name = class_names[0]
            conf_percentage = int(score * 100)
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image.save(os.path.join(output_folder, f"{image_file.split(".")[0]}_{class_name}_{i}_c{conf_percentage}.jpg"))

        # for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        #     x_min, y_min, x_max, y_max, conf = box
        #     class_name = class_names[int(class_id)]
        #     conf_percentage = int(conf * 100)
        #     cropped_image = image.crop((x_min, y_min, x_max, y_max))
        #     cropped_image.save(os.path.join(output_folder, f"{image_file}_{class_name}_{i}_c{conf_percentage}.jpg"))


```


```python
cropped_bricks_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Cropped_bricks"
# test_path = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\test"

crop_bounding_boxes(resized_spilt_bricks_raw_images_folder, cropped_bricks_folder, Bricks_model, class_names=["LegoBrick"], conf_threshold=0.50)
# crop_bounding_boxes(test_path, cropped_bricks_folder, Bricks_model, class_names=["LegoBrick"], conf_threshold=0.50)

# takes 6 minutes to run
```

    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_0.jpg: 640x480 2 LegoBricks, 220.4ms
    Speed: 7.0ms preprocess, 220.4ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1.jpg: 640x480 9 LegoBricks, 175.5ms
    Speed: 14.0ms preprocess, 175.5ms inference, 3.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_10.jpg: 640x480 3 LegoBricks, 153.6ms
    Speed: 7.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_100.jpg: 480x640 7 LegoBricks, 173.5ms
    Speed: 5.0ms preprocess, 173.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1000.jpg: 480x640 4 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1001.jpg: 480x640 5 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1002.jpg: 480x640 5 LegoBricks, 165.6ms
    Speed: 5.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1003.jpg: 480x640 3 LegoBricks, 155.7ms
    Speed: 5.0ms preprocess, 155.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1004.jpg: 480x640 3 LegoBricks, 393.9ms
    Speed: 15.0ms preprocess, 393.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1006.jpg: 480x640 3 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1007.jpg: 480x640 6 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1008.jpg: 480x640 1 LegoBrick, 154.6ms
    Speed: 5.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1009.jpg: 480x640 1 LegoBrick, 106.7ms
    Speed: 5.0ms preprocess, 106.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_101.jpg: 480x640 4 LegoBricks, 117.7ms
    Speed: 5.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1010.jpg: 480x640 2 LegoBricks, 148.6ms
    Speed: 11.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1011.jpg: 480x640 3 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1012.jpg: 480x640 1 LegoBrick, 127.7ms
    Speed: 5.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1013.jpg: 480x640 3 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1014.jpg: 480x640 2 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1015.jpg: 480x640 2 LegoBricks, 214.4ms
    Speed: 7.0ms preprocess, 214.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1016.jpg: 480x640 1 LegoBrick, 310.2ms
    Speed: 8.0ms preprocess, 310.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1017.jpg: 480x640 1 LegoBrick, 168.5ms
    Speed: 16.0ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1018.jpg: 480x640 5 LegoBricks, 121.7ms
    Speed: 6.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1019.jpg: 480x640 1 LegoBrick, 127.7ms
    Speed: 5.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_102.jpg: 480x640 3 LegoBricks, 129.7ms
    Speed: 6.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1020.jpg: 480x640 2 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1021.jpg: 480x640 2 LegoBricks, 230.4ms
    Speed: 60.8ms preprocess, 230.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1022.jpg: 480x640 4 LegoBricks, 162.6ms
    Speed: 23.9ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1023.jpg: 480x640 7 LegoBricks, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1024.jpg: 480x640 5 LegoBricks, 148.5ms
    Speed: 5.1ms preprocess, 148.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1025.jpg: 480x640 4 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1026.jpg: 480x640 5 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1027.jpg: 480x640 3 LegoBricks, 120.7ms
    Speed: 7.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1028.jpg: 480x640 4 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1029.jpg: 480x640 4 LegoBricks, 121.2ms
    Speed: 5.7ms preprocess, 121.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_103.jpg: 480x640 7 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1030.jpg: 480x640 5 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1031.jpg: 480x640 1 LegoBrick, 140.6ms
    Speed: 5.0ms preprocess, 140.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1032.jpg: 480x640 3 LegoBricks, 191.5ms
    Speed: 7.0ms preprocess, 191.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1033.jpg: 480x640 3 LegoBricks, 352.1ms
    Speed: 16.0ms preprocess, 352.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1034.jpg: 480x640 4 LegoBricks, 190.5ms
    Speed: 7.0ms preprocess, 190.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1035.jpg: 480x640 4 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1036.jpg: 480x640 3 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1038.jpg: 480x640 5 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1039.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 6.0ms preprocess, 133.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_104.jpg: 480x640 6 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1040.jpg: 480x640 2 LegoBricks, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1042.jpg: 480x640 4 LegoBricks, 154.6ms
    Speed: 6.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1043.jpg: 480x640 4 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1045.jpg: 480x640 2 LegoBricks, 175.5ms
    Speed: 5.0ms preprocess, 175.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1046.jpg: 480x640 2 LegoBricks, 103.7ms
    Speed: 5.0ms preprocess, 103.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1047.jpg: 480x640 3 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1049.jpg: 480x640 1 LegoBrick, 114.7ms
    Speed: 4.9ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1050.jpg: 480x640 3 LegoBricks, 152.6ms
    Speed: 5.0ms preprocess, 152.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1051.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 4.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1052.jpg: 480x640 4 LegoBricks, 382.0ms
    Speed: 24.9ms preprocess, 382.0ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1053.jpg: 480x640 3 LegoBricks, 156.6ms
    Speed: 5.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1056.jpg: 480x640 1 LegoBrick, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1057.jpg: 480x640 5 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1058.jpg: 480x640 4 LegoBricks, 131.6ms
    Speed: 4.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1059.jpg: 480x640 4 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_106.jpg: 480x640 5 LegoBricks, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1060.jpg: 480x640 5 LegoBricks, 178.5ms
    Speed: 5.0ms preprocess, 178.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1061.jpg: 480x640 2 LegoBricks, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1062.jpg: 480x640 4 LegoBricks, 170.0ms
    Speed: 7.0ms preprocess, 170.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1063.jpg: 480x640 2 LegoBricks, 234.4ms
    Speed: 6.0ms preprocess, 234.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1064.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1065.jpg: 480x640 5 LegoBricks, 203.5ms
    Speed: 6.0ms preprocess, 203.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1066.jpg: 480x640 4 LegoBricks, 156.6ms
    Speed: 9.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1067.jpg: 480x640 5 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1068.jpg: 480x640 6 LegoBricks, 390.0ms
    Speed: 10.0ms preprocess, 390.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1069.jpg: 480x640 7 LegoBricks, 161.6ms
    Speed: 7.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1070.jpg: 480x640 5 LegoBricks, 292.2ms
    Speed: 5.0ms preprocess, 292.2ms inference, 11.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1071.jpg: 480x640 3 LegoBricks, 264.3ms
    Speed: 19.0ms preprocess, 264.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1072.jpg: 480x640 3 LegoBricks, 160.9ms
    Speed: 5.0ms preprocess, 160.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1073.jpg: 480x640 4 LegoBricks, 142.6ms
    Speed: 6.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1074.jpg: 480x640 4 LegoBricks, 152.6ms
    Speed: 5.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1075.jpg: 480x640 5 LegoBricks, 123.7ms
    Speed: 6.0ms preprocess, 123.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1076.jpg: 480x640 2 LegoBricks, 132.1ms
    Speed: 5.0ms preprocess, 132.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1077.jpg: 480x640 2 LegoBricks, 254.3ms
    Speed: 6.0ms preprocess, 254.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1078.jpg: 480x640 3 LegoBricks, 155.6ms
    Speed: 6.0ms preprocess, 155.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1079.jpg: 480x640 4 LegoBricks, 122.0ms
    Speed: 5.0ms preprocess, 122.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_108.jpg: 480x640 2 LegoBricks, 121.3ms
    Speed: 5.0ms preprocess, 121.3ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1080.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1083.jpg: 480x640 2 LegoBricks, 368.3ms
    Speed: 15.0ms preprocess, 368.3ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1084.jpg: 480x640 2 LegoBricks, 214.6ms
    Speed: 10.0ms preprocess, 214.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1086.jpg: 480x640 5 LegoBricks, 155.6ms
    Speed: 6.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1087.jpg: 480x640 2 LegoBricks, 175.5ms
    Speed: 6.0ms preprocess, 175.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1088.jpg: 480x640 2 LegoBricks, 193.5ms
    Speed: 5.0ms preprocess, 193.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1089.jpg: 480x640 1 LegoBrick, 152.0ms
    Speed: 5.0ms preprocess, 152.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_109.jpg: 480x640 2 LegoBricks, 156.0ms
    Speed: 6.0ms preprocess, 156.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1090.jpg: 480x640 1 LegoBrick, 149.8ms
    Speed: 11.0ms preprocess, 149.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1091.jpg: 480x640 1 LegoBrick, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1092.jpg: 480x640 2 LegoBricks, 135.6ms
    Speed: 7.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1093.jpg: 480x640 5 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1094.jpg: 480x640 5 LegoBricks, 142.6ms
    Speed: 6.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1095.jpg: 480x640 1 LegoBrick, 154.6ms
    Speed: 7.0ms preprocess, 154.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1096.jpg: 480x640 5 LegoBricks, 166.6ms
    Speed: 6.0ms preprocess, 166.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1097.jpg: 480x640 5 LegoBricks, 149.6ms
    Speed: 6.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1098.jpg: 480x640 4 LegoBricks, 153.5ms
    Speed: 5.0ms preprocess, 153.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1099.jpg: 480x640 1 LegoBrick, 156.6ms
    Speed: 6.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_110.jpg: 480x640 3 LegoBricks, 191.5ms
    Speed: 56.8ms preprocess, 191.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1100.jpg: 480x640 3 LegoBricks, 333.1ms
    Speed: 12.0ms preprocess, 333.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1101.jpg: 480x640 3 LegoBricks, 171.5ms
    Speed: 6.6ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1102.jpg: 480x640 3 LegoBricks, 172.5ms
    Speed: 6.0ms preprocess, 172.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1103.jpg: 480x640 1 LegoBrick, 190.5ms
    Speed: 7.0ms preprocess, 190.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1106.jpg: 480x640 5 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1108.jpg: 480x640 1 LegoBrick, 152.6ms
    Speed: 6.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1109.jpg: 480x640 1 LegoBrick, 149.6ms
    Speed: 7.0ms preprocess, 149.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_111.jpg: 480x640 5 LegoBricks, 154.6ms
    Speed: 5.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1111.jpg: 480x640 2 LegoBricks, 159.6ms
    Speed: 5.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1112.jpg: 480x640 3 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1113.jpg: 480x640 9 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1114.jpg: 480x640 5 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1115.jpg: 480x640 4 LegoBricks, 152.9ms
    Speed: 5.0ms preprocess, 152.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1116.jpg: 480x640 10 LegoBricks, 131.5ms
    Speed: 5.0ms preprocess, 131.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1117.jpg: 480x640 2 LegoBricks, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1118.jpg: 480x640 1 LegoBrick, 325.0ms
    Speed: 31.9ms preprocess, 325.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_112.jpg: 480x640 6 LegoBricks, 169.5ms
    Speed: 12.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1120.jpg: 480x640 7 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1121.jpg: 480x640 5 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1122.jpg: 480x640 4 LegoBricks, 190.5ms
    Speed: 6.0ms preprocess, 190.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1123.jpg: 480x640 3 LegoBricks, 356.0ms
    Speed: 7.0ms preprocess, 356.0ms inference, 24.9ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1124.jpg: 480x640 3 LegoBricks, 250.3ms
    Speed: 75.8ms preprocess, 250.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1125.jpg: 480x640 2 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1126.jpg: 480x640 4 LegoBricks, 236.4ms
    Speed: 6.0ms preprocess, 236.4ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1127.jpg: 480x640 1 LegoBrick, 176.5ms
    Speed: 6.0ms preprocess, 176.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1128.jpg: 480x640 5 LegoBricks, 176.5ms
    Speed: 6.0ms preprocess, 176.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1129.jpg: 480x640 5 LegoBricks, 171.5ms
    Speed: 6.0ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_113.jpg: 480x640 3 LegoBricks, 170.5ms
    Speed: 5.0ms preprocess, 170.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1130.jpg: 480x640 2 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1131.jpg: 480x640 4 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1132.jpg: 480x640 4 LegoBricks, 227.4ms
    Speed: 6.0ms preprocess, 227.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1133.jpg: 480x640 3 LegoBricks, 212.6ms
    Speed: 8.0ms preprocess, 212.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1134.jpg: 480x640 3 LegoBricks, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1135.jpg: 480x640 2 LegoBricks, 198.5ms
    Speed: 7.0ms preprocess, 198.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1136.jpg: 480x640 2 LegoBricks, 194.5ms
    Speed: 5.0ms preprocess, 194.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1137.jpg: 480x640 2 LegoBricks, 186.5ms
    Speed: 8.0ms preprocess, 186.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1138.jpg: 480x640 1 LegoBrick, 171.5ms
    Speed: 6.0ms preprocess, 171.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1139.jpg: 480x640 3 LegoBricks, 166.6ms
    Speed: 7.0ms preprocess, 166.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_114.jpg: 480x640 11 LegoBricks, 160.6ms
    Speed: 7.0ms preprocess, 160.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1140.jpg: 480x640 3 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1141.jpg: 480x640 2 LegoBricks, 167.6ms
    Speed: 5.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1142.jpg: 480x640 2 LegoBricks, 201.5ms
    Speed: 67.8ms preprocess, 201.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1143.jpg: 480x640 2 LegoBricks, 160.6ms
    Speed: 6.0ms preprocess, 160.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1144.jpg: 480x640 3 LegoBricks, 194.5ms
    Speed: 6.0ms preprocess, 194.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1145.jpg: 480x640 2 LegoBricks, 311.2ms
    Speed: 12.0ms preprocess, 311.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1146.jpg: 480x640 3 LegoBricks, 210.4ms
    Speed: 18.0ms preprocess, 210.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1147.jpg: 480x640 3 LegoBricks, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1148.jpg: 480x640 2 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1149.jpg: 480x640 3 LegoBricks, 156.6ms
    Speed: 6.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_115.jpg: 480x640 3 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1150.jpg: 480x640 2 LegoBricks, 162.6ms
    Speed: 5.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1151.jpg: 480x640 3 LegoBricks, 141.4ms
    Speed: 4.0ms preprocess, 141.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1152.jpg: 480x640 2 LegoBricks, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1153.jpg: 480x640 3 LegoBricks, 140.6ms
    Speed: 4.0ms preprocess, 140.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1154.jpg: 480x640 4 LegoBricks, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1155.jpg: 480x640 4 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 2.7ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1156.jpg: 480x640 3 LegoBricks, 144.6ms
    Speed: 6.0ms preprocess, 144.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1157.jpg: 480x640 3 LegoBricks, 169.5ms
    Speed: 6.0ms preprocess, 169.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1158.jpg: 480x640 2 LegoBricks, 167.6ms
    Speed: 5.0ms preprocess, 167.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1159.jpg: 480x640 3 LegoBricks, 165.6ms
    Speed: 5.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_116.jpg: 480x640 5 LegoBricks, 318.1ms
    Speed: 4.0ms preprocess, 318.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1160.jpg: 480x640 1 LegoBrick, 195.9ms
    Speed: 5.0ms preprocess, 195.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1161.jpg: 480x640 4 LegoBricks, 284.2ms
    Speed: 6.0ms preprocess, 284.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1162.jpg: 480x640 7 LegoBricks, 181.5ms
    Speed: 7.0ms preprocess, 181.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1163.jpg: 480x640 3 LegoBricks, 182.5ms
    Speed: 5.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1164.jpg: 480x640 2 LegoBricks, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1165.jpg: 480x640 4 LegoBricks, 155.6ms
    Speed: 6.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1166.jpg: 480x640 5 LegoBricks, 164.6ms
    Speed: 5.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1167.jpg: 480x640 2 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1168.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1169.jpg: 480x640 3 LegoBricks, 151.9ms
    Speed: 6.0ms preprocess, 151.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_117.jpg: 480x640 3 LegoBricks, 146.1ms
    Speed: 4.0ms preprocess, 146.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1170.jpg: 480x640 2 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1171.jpg: 480x640 3 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1172.jpg: 480x640 2 LegoBricks, 190.5ms
    Speed: 6.0ms preprocess, 190.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1173.jpg: 480x640 2 LegoBricks, 379.9ms
    Speed: 12.0ms preprocess, 379.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1174.jpg: 480x640 3 LegoBricks, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1175.jpg: 480x640 4 LegoBricks, 141.1ms
    Speed: 5.0ms preprocess, 141.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1176.jpg: 480x640 2 LegoBricks, 144.8ms
    Speed: 7.0ms preprocess, 144.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1177.jpg: 480x640 5 LegoBricks, 152.6ms
    Speed: 12.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1178.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1179.jpg: 480x640 7 LegoBricks, 149.0ms
    Speed: 6.0ms preprocess, 149.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_118.jpg: 480x640 4 LegoBricks, 142.8ms
    Speed: 6.0ms preprocess, 142.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1180.jpg: 480x640 4 LegoBricks, 144.5ms
    Speed: 7.0ms preprocess, 144.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1181.jpg: 480x640 3 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1182.jpg: 480x640 7 LegoBricks, 169.5ms
    Speed: 37.9ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1183.jpg: 480x640 6 LegoBricks, 135.6ms
    Speed: 5.0ms preprocess, 135.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1185.jpg: 480x640 5 LegoBricks, 134.6ms
    Speed: 6.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1186.jpg: 480x640 3 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1187.jpg: 480x640 2 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1188.jpg: 480x640 2 LegoBricks, 150.8ms
    Speed: 6.0ms preprocess, 150.8ms inference, 1.8ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1189.jpg: 480x640 3 LegoBricks, 157.6ms
    Speed: 7.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_119.jpg: 480x640 3 LegoBricks, 134.6ms
    Speed: 6.0ms preprocess, 134.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1190.jpg: 480x640 5 LegoBricks, 138.4ms
    Speed: 5.0ms preprocess, 138.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1191.jpg: 480x640 6 LegoBricks, 209.5ms
    Speed: 5.0ms preprocess, 209.5ms inference, 5.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1192.jpg: 480x640 3 LegoBricks, 143.6ms
    Speed: 6.0ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1193.jpg: 480x640 11 LegoBricks, 165.9ms
    Speed: 6.0ms preprocess, 165.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1194.jpg: 480x640 3 LegoBricks, 208.4ms
    Speed: 5.0ms preprocess, 208.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1195.jpg: 480x640 5 LegoBricks, 213.4ms
    Speed: 7.0ms preprocess, 213.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1196.jpg: 480x640 3 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1197.jpg: 480x640 4 LegoBricks, 149.6ms
    Speed: 7.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1198.jpg: 480x640 3 LegoBricks, 127.5ms
    Speed: 6.0ms preprocess, 127.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1199.jpg: 480x640 2 LegoBricks, 117.6ms
    Speed: 6.0ms preprocess, 117.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_12.jpg: 640x480 2 LegoBricks, 116.6ms
    Speed: 6.1ms preprocess, 116.6ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1200.jpg: 480x640 3 LegoBricks, 128.7ms
    Speed: 5.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1201.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1202.jpg: 480x640 1 LegoBrick, 223.4ms
    Speed: 6.0ms preprocess, 223.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1203.jpg: 480x640 3 LegoBricks, 148.8ms
    Speed: 6.0ms preprocess, 148.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1204.jpg: 480x640 3 LegoBricks, 160.6ms
    Speed: 6.0ms preprocess, 160.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1205.jpg: 480x640 4 LegoBricks, 216.3ms
    Speed: 6.0ms preprocess, 216.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1206.jpg: 480x640 3 LegoBricks, 234.4ms
    Speed: 29.9ms preprocess, 234.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1208.jpg: 480x640 1 LegoBrick, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1209.jpg: 480x640 1 LegoBrick, 117.6ms
    Speed: 5.0ms preprocess, 117.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_121.jpg: 480x640 2 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1210.jpg: 480x640 1 LegoBrick, 127.7ms
    Speed: 6.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1211.jpg: 480x640 4 LegoBricks, 124.2ms
    Speed: 5.0ms preprocess, 124.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1212.jpg: 480x640 5 LegoBricks, 106.6ms
    Speed: 5.0ms preprocess, 106.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1213.jpg: 480x640 2 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1214.jpg: 480x640 5 LegoBricks, 133.6ms
    Speed: 6.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1215.jpg: 480x640 4 LegoBricks, 143.6ms
    Speed: 6.0ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1216.jpg: 480x640 2 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1218.jpg: 480x640 2 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1219.jpg: 480x640 4 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_122.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.2ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1220.jpg: 480x640 2 LegoBricks, 120.3ms
    Speed: 5.0ms preprocess, 120.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1222.jpg: 480x640 2 LegoBricks, 120.5ms
    Speed: 5.0ms preprocess, 120.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1223.jpg: 480x640 2 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1226.jpg: 480x640 3 LegoBricks, 282.7ms
    Speed: 48.9ms preprocess, 282.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1227.jpg: 480x640 2 LegoBricks, 212.4ms
    Speed: 11.0ms preprocess, 212.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1228.jpg: 480x640 4 LegoBricks, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1229.jpg: 480x640 3 LegoBricks, 155.0ms
    Speed: 5.0ms preprocess, 155.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_123.jpg: 480x640 3 LegoBricks, 182.5ms
    Speed: 7.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1230.jpg: 480x640 4 LegoBricks, 168.5ms
    Speed: 6.0ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1231.jpg: 480x640 7 LegoBricks, 157.3ms
    Speed: 5.0ms preprocess, 157.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1232.jpg: 480x640 5 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1233.jpg: 480x640 4 LegoBricks, 181.5ms
    Speed: 6.0ms preprocess, 181.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1234.jpg: 480x640 6 LegoBricks, 167.6ms
    Speed: 5.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1235.jpg: 480x640 1 LegoBrick, 145.6ms
    Speed: 5.0ms preprocess, 145.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1236.jpg: 480x640 1 LegoBrick, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1238.jpg: 480x640 1 LegoBrick, 112.7ms
    Speed: 6.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1239.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_124.jpg: 480x640 1 LegoBrick, 115.7ms
    Speed: 4.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1240.jpg: 480x640 3 LegoBricks, 128.7ms
    Speed: 4.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1241.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 6.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1242.jpg: 480x640 2 LegoBricks, 118.2ms
    Speed: 4.0ms preprocess, 118.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1243.jpg: 480x640 4 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1245.jpg: 480x640 5 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1248.jpg: 480x640 1 LegoBrick, 224.0ms
    Speed: 5.0ms preprocess, 224.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1249.jpg: 480x640 2 LegoBricks, 263.2ms
    Speed: 12.0ms preprocess, 263.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_125.jpg: 480x640 3 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1250.jpg: 480x640 3 LegoBricks, 123.0ms
    Speed: 5.0ms preprocess, 123.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1252.jpg: 480x640 4 LegoBricks, 151.7ms
    Speed: 6.0ms preprocess, 151.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1256.jpg: 480x640 3 LegoBricks, 163.9ms
    Speed: 6.0ms preprocess, 163.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1258.jpg: 480x640 5 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1259.jpg: 480x640 8 LegoBricks, 152.6ms
    Speed: 5.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_126.jpg: 480x640 3 LegoBricks, 169.5ms
    Speed: 7.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1260.jpg: 480x640 7 LegoBricks, 159.7ms
    Speed: 6.0ms preprocess, 159.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1261.jpg: 480x640 3 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1263.jpg: 480x640 4 LegoBricks, 131.9ms
    Speed: 5.0ms preprocess, 131.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1264.jpg: 480x640 4 LegoBricks, 119.5ms
    Speed: 6.0ms preprocess, 119.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1265.jpg: 480x640 4 LegoBricks, 121.6ms
    Speed: 6.0ms preprocess, 121.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1266.jpg: 480x640 4 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1267.jpg: 480x640 4 LegoBricks, 129.9ms
    Speed: 5.0ms preprocess, 129.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1268.jpg: 480x640 3 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1269.jpg: 480x640 3 LegoBricks, 119.5ms
    Speed: 5.0ms preprocess, 119.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_127.jpg: 480x640 4 LegoBricks, 128.2ms
    Speed: 4.0ms preprocess, 128.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1270.jpg: 480x640 1 LegoBrick, 188.5ms
    Speed: 5.0ms preprocess, 188.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1271.jpg: 480x640 2 LegoBricks, 295.3ms
    Speed: 6.0ms preprocess, 295.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1272.jpg: 480x640 3 LegoBricks, 177.5ms
    Speed: 5.0ms preprocess, 177.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1273.jpg: 480x640 4 LegoBricks, 120.7ms
    Speed: 6.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1274.jpg: 480x640 1 LegoBrick, 187.5ms
    Speed: 5.0ms preprocess, 187.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1275.jpg: 480x640 1 LegoBrick, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1276.jpg: 480x640 2 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1277.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 8.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1278.jpg: 480x640 5 LegoBricks, 120.8ms
    Speed: 5.9ms preprocess, 120.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1279.jpg: 480x640 3 LegoBricks, 117.1ms
    Speed: 6.0ms preprocess, 117.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_128.jpg: 480x640 1 LegoBrick, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1280.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1281.jpg: 480x640 4 LegoBricks, 116.7ms
    Speed: 7.2ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1282.jpg: 480x640 2 LegoBricks, 135.3ms
    Speed: 7.0ms preprocess, 135.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1283.jpg: 480x640 3 LegoBricks, 124.9ms
    Speed: 6.0ms preprocess, 124.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1284.jpg: 480x640 3 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1285.jpg: 480x640 2 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1286.jpg: 480x640 2 LegoBricks, 157.6ms
    Speed: 6.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1288.jpg: 480x640 1 LegoBrick, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1289.jpg: 480x640 2 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_129.jpg: 480x640 2 LegoBricks, 136.6ms
    Speed: 6.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1290.jpg: 480x640 3 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1291.jpg: 480x640 3 LegoBricks, 320.1ms
    Speed: 7.0ms preprocess, 320.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1292.jpg: 480x640 2 LegoBricks, 138.5ms
    Speed: 5.0ms preprocess, 138.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1293.jpg: 480x640 4 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1295.jpg: 480x640 3 LegoBricks, 173.6ms
    Speed: 5.0ms preprocess, 173.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1296.jpg: 480x640 4 LegoBricks, 139.6ms
    Speed: 18.9ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1297.jpg: 480x640 6 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1298.jpg: 480x640 3 LegoBricks, 156.6ms
    Speed: 6.0ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1299.jpg: 480x640 2 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_13.jpg: 640x480 2 LegoBricks, 130.7ms
    Speed: 6.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_130.jpg: 480x640 3 LegoBricks, 109.9ms
    Speed: 5.0ms preprocess, 109.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1300.jpg: 480x640 1 LegoBrick, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1301.jpg: 480x640 1 LegoBrick, 98.2ms
    Speed: 3.0ms preprocess, 98.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1302.jpg: 480x640 2 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1303.jpg: 480x640 2 LegoBricks, 129.7ms
    Speed: 5.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1304.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1306.jpg: 480x640 2 LegoBricks, 106.7ms
    Speed: 6.0ms preprocess, 106.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1307.jpg: 480x640 4 LegoBricks, 131.2ms
    Speed: 5.0ms preprocess, 131.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1308.jpg: 480x640 2 LegoBricks, 185.6ms
    Speed: 6.0ms preprocess, 185.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1309.jpg: 480x640 1 LegoBrick, 279.1ms
    Speed: 6.0ms preprocess, 279.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_131.jpg: 480x640 1 LegoBrick, 146.0ms
    Speed: 6.0ms preprocess, 146.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1310.jpg: 480x640 1 LegoBrick, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1311.jpg: 480x640 3 LegoBricks, 104.7ms
    Speed: 5.0ms preprocess, 104.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1312.jpg: 480x640 1 LegoBrick, 109.3ms
    Speed: 6.0ms preprocess, 109.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1313.jpg: 480x640 1 LegoBrick, 122.0ms
    Speed: 6.0ms preprocess, 122.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1314.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 7.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1315.jpg: 480x640 5 LegoBricks, 112.9ms
    Speed: 6.0ms preprocess, 112.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1317.jpg: 480x640 3 LegoBricks, 164.8ms
    Speed: 4.0ms preprocess, 164.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1318.jpg: 480x640 2 LegoBricks, 112.6ms
    Speed: 4.0ms preprocess, 112.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1319.jpg: 480x640 1 LegoBrick, 108.7ms
    Speed: 4.0ms preprocess, 108.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_132.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 4.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1320.jpg: 480x640 1 LegoBrick, 105.7ms
    Speed: 5.0ms preprocess, 105.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1321.jpg: 480x640 2 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1323.jpg: 480x640 3 LegoBricks, 103.7ms
    Speed: 5.0ms preprocess, 103.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1324.jpg: 480x640 6 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1326.jpg: 480x640 4 LegoBricks, 105.7ms
    Speed: 5.0ms preprocess, 105.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1328.jpg: 480x640 4 LegoBricks, 113.0ms
    Speed: 6.0ms preprocess, 113.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1329.jpg: 480x640 4 LegoBricks, 115.0ms
    Speed: 6.0ms preprocess, 115.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_133.jpg: 480x640 2 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1330.jpg: 480x640 5 LegoBricks, 120.0ms
    Speed: 5.0ms preprocess, 120.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1331.jpg: 480x640 3 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1332.jpg: 480x640 3 LegoBricks, 187.9ms
    Speed: 6.0ms preprocess, 187.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1333.jpg: 480x640 5 LegoBricks, 243.3ms
    Speed: 18.0ms preprocess, 243.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1334.jpg: 480x640 5 LegoBricks, 159.6ms
    Speed: 6.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1335.jpg: 480x640 2 LegoBricks, 177.5ms
    Speed: 7.0ms preprocess, 177.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1336.jpg: 480x640 2 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1337.jpg: 480x640 5 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1339.jpg: 480x640 5 LegoBricks, 127.7ms
    Speed: 6.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_134.jpg: 480x640 2 LegoBricks, 117.5ms
    Speed: 4.0ms preprocess, 117.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1340.jpg: 480x640 2 LegoBricks, 133.6ms
    Speed: 6.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1341.jpg: 480x640 1 LegoBrick, 145.6ms
    Speed: 52.9ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1342.jpg: 480x640 4 LegoBricks, 114.0ms
    Speed: 5.0ms preprocess, 114.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1343.jpg: 480x640 2 LegoBricks, 111.7ms
    Speed: 6.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1344.jpg: 480x640 3 LegoBricks, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1346.jpg: 480x640 2 LegoBricks, 109.3ms
    Speed: 5.0ms preprocess, 109.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1347.jpg: 480x640 3 LegoBricks, 136.6ms
    Speed: 6.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1348.jpg: 480x640 5 LegoBricks, 128.6ms
    Speed: 5.0ms preprocess, 128.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1349.jpg: 480x640 2 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_135.jpg: 480x640 1 LegoBrick, 136.6ms
    Speed: 6.0ms preprocess, 136.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1350.jpg: 480x640 1 LegoBrick, 245.3ms
    Speed: 8.0ms preprocess, 245.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1351.jpg: 480x640 3 LegoBricks, 212.4ms
    Speed: 6.0ms preprocess, 212.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1352.jpg: 480x640 5 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1353.jpg: 480x640 3 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1354.jpg: 480x640 5 LegoBricks, 162.6ms
    Speed: 6.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1355.jpg: 480x640 4 LegoBricks, 160.6ms
    Speed: 6.0ms preprocess, 160.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1356.jpg: 480x640 1 LegoBrick, 352.1ms
    Speed: 6.0ms preprocess, 352.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1357.jpg: 480x640 1 LegoBrick, 267.3ms
    Speed: 8.0ms preprocess, 267.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1358.jpg: 480x640 2 LegoBricks, 178.5ms
    Speed: 7.0ms preprocess, 178.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1359.jpg: 480x640 2 LegoBricks, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_136.jpg: 480x640 1 LegoBrick, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1360.jpg: 480x640 1 LegoBrick, 146.6ms
    Speed: 7.0ms preprocess, 146.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1361.jpg: 480x640 2 LegoBricks, 144.6ms
    Speed: 6.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1362.jpg: 480x640 1 LegoBrick, 159.6ms
    Speed: 34.9ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1363.jpg: 480x640 4 LegoBricks, 164.6ms
    Speed: 5.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1364.jpg: 480x640 4 LegoBricks, 143.6ms
    Speed: 6.0ms preprocess, 143.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1365.jpg: 480x640 3 LegoBricks, 197.5ms
    Speed: 6.0ms preprocess, 197.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1366.jpg: 480x640 2 LegoBricks, 234.4ms
    Speed: 9.9ms preprocess, 234.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1367.jpg: 480x640 2 LegoBricks, 158.6ms
    Speed: 7.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1368.jpg: 480x640 1 LegoBrick, 154.6ms
    Speed: 7.0ms preprocess, 154.6ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1369.jpg: 480x640 2 LegoBricks, 138.6ms
    Speed: 7.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_137.jpg: 480x640 1 LegoBrick, 145.6ms
    Speed: 5.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1370.jpg: 480x640 4 LegoBricks, 159.6ms
    Speed: 5.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1371.jpg: 480x640 5 LegoBricks, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1372.jpg: 480x640 3 LegoBricks, 156.6ms
    Speed: 7.0ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1373.jpg: 480x640 5 LegoBricks, 169.5ms
    Speed: 6.0ms preprocess, 169.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1374.jpg: 480x640 6 LegoBricks, 157.6ms
    Speed: 6.0ms preprocess, 157.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1375.jpg: 480x640 1 LegoBrick, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1376.jpg: 480x640 1 LegoBrick, 149.6ms
    Speed: 6.0ms preprocess, 149.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1377.jpg: 480x640 2 LegoBricks, 161.6ms
    Speed: 6.0ms preprocess, 161.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1378.jpg: 480x640 3 LegoBricks, 172.5ms
    Speed: 8.0ms preprocess, 172.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1379.jpg: 480x640 4 LegoBricks, 146.9ms
    Speed: 7.0ms preprocess, 146.9ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_138.jpg: 480x640 2 LegoBricks, 206.4ms
    Speed: 5.0ms preprocess, 206.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1380.jpg: 480x640 4 LegoBricks, 297.2ms
    Speed: 5.0ms preprocess, 297.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1381.jpg: 480x640 3 LegoBricks, 215.4ms
    Speed: 6.0ms preprocess, 215.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1382.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 5.6ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1383.jpg: 480x640 3 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1384.jpg: 480x640 4 LegoBricks, 156.6ms
    Speed: 5.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1385.jpg: 480x640 4 LegoBricks, 228.4ms
    Speed: 6.0ms preprocess, 228.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1386.jpg: 480x640 1 LegoBrick, 183.5ms
    Speed: 7.0ms preprocess, 183.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1387.jpg: 480x640 5 LegoBricks, 133.6ms
    Speed: 6.0ms preprocess, 133.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1388.jpg: 480x640 5 LegoBricks, 148.6ms
    Speed: 5.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1389.jpg: 480x640 6 LegoBricks, 165.6ms
    Speed: 6.0ms preprocess, 165.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_139.jpg: 480x640 2 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1390.jpg: 480x640 2 LegoBricks, 137.2ms
    Speed: 5.0ms preprocess, 137.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1391.jpg: 480x640 3 LegoBricks, 117.9ms
    Speed: 6.0ms preprocess, 117.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1392.jpg: 480x640 3 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1393.jpg: 480x640 5 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1394.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1395.jpg: 480x640 3 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1396.jpg: 480x640 2 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1397.jpg: 480x640 3 LegoBricks, 322.1ms
    Speed: 5.6ms preprocess, 322.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1398.jpg: 480x640 3 LegoBricks, 293.2ms
    Speed: 13.0ms preprocess, 293.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1399.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 6.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_14.jpg: 640x480 3 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_140.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 4.0ms preprocess, 147.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1400.jpg: 480x640 3 LegoBricks, 225.3ms
    Speed: 5.0ms preprocess, 225.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1401.jpg: 480x640 3 LegoBricks, 170.5ms
    Speed: 5.0ms preprocess, 170.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1402.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1403.jpg: 480x640 5 LegoBricks, 156.6ms
    Speed: 5.0ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1404.jpg: 480x640 1 LegoBrick, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1405.jpg: 480x640 1 LegoBrick, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1406.jpg: 480x640 4 LegoBricks, 104.3ms
    Speed: 5.0ms preprocess, 104.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1407.jpg: 480x640 2 LegoBricks, 131.6ms
    Speed: 6.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1408.jpg: 480x640 4 LegoBricks, 124.7ms
    Speed: 4.8ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1409.jpg: 480x640 4 LegoBricks, 108.6ms
    Speed: 5.0ms preprocess, 108.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_141.jpg: 480x640 7 LegoBricks, 111.6ms
    Speed: 3.0ms preprocess, 111.6ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1410.jpg: 480x640 3 LegoBricks, 103.7ms
    Speed: 3.0ms preprocess, 103.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1411.jpg: 480x640 5 LegoBricks, 99.7ms
    Speed: 5.0ms preprocess, 99.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1412.jpg: 480x640 4 LegoBricks, 117.7ms
    Speed: 5.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1413.jpg: 480x640 4 LegoBricks, 118.6ms
    Speed: 6.0ms preprocess, 118.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1414.jpg: 480x640 3 LegoBricks, 264.4ms
    Speed: 5.0ms preprocess, 264.4ms inference, 6.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1415.jpg: 480x640 3 LegoBricks, 181.5ms
    Speed: 5.0ms preprocess, 181.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1417.jpg: 480x640 3 LegoBricks, 118.7ms
    Speed: 6.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1418.jpg: 480x640 1 LegoBrick, 116.7ms
    Speed: 6.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1419.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_142.jpg: 480x640 5 LegoBricks, 218.4ms
    Speed: 7.0ms preprocess, 218.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1420.jpg: 480x640 5 LegoBricks, 131.8ms
    Speed: 6.0ms preprocess, 131.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1421.jpg: 480x640 7 LegoBricks, 336.8ms
    Speed: 5.0ms preprocess, 336.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1422.jpg: 480x640 6 LegoBricks, 177.1ms
    Speed: 6.0ms preprocess, 177.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1423.jpg: 480x640 2 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1424.jpg: 480x640 2 LegoBricks, 114.4ms
    Speed: 5.0ms preprocess, 114.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1425.jpg: 480x640 2 LegoBricks, 116.7ms
    Speed: 7.0ms preprocess, 116.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1426.jpg: 480x640 3 LegoBricks, 109.2ms
    Speed: 5.0ms preprocess, 109.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1427.jpg: 480x640 1 LegoBrick, 114.6ms
    Speed: 5.0ms preprocess, 114.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1428.jpg: 480x640 7 LegoBricks, 90.8ms
    Speed: 5.0ms preprocess, 90.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1429.jpg: 480x640 8 LegoBricks, 91.8ms
    Speed: 4.0ms preprocess, 91.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_143.jpg: 480x640 4 LegoBricks, 110.7ms
    Speed: 4.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1430.jpg: 480x640 7 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1431.jpg: 480x640 2 LegoBricks, 114.3ms
    Speed: 5.0ms preprocess, 114.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1432.jpg: 480x640 1 LegoBrick, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1433.jpg: 480x640 2 LegoBricks, 129.7ms
    Speed: 4.8ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1434.jpg: 480x640 2 LegoBricks, 218.1ms
    Speed: 13.0ms preprocess, 218.1ms inference, 9.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1435.jpg: 480x640 6 LegoBricks, 192.3ms
    Speed: 6.1ms preprocess, 192.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1436.jpg: 480x640 1 LegoBrick, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1437.jpg: 480x640 2 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1438.jpg: 480x640 3 LegoBricks, 134.6ms
    Speed: 6.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1439.jpg: 480x640 3 LegoBricks, 267.0ms
    Speed: 31.9ms preprocess, 267.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_144.jpg: 480x640 5 LegoBricks, 209.1ms
    Speed: 21.9ms preprocess, 209.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1440.jpg: 480x640 3 LegoBricks, 183.4ms
    Speed: 6.0ms preprocess, 183.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1441.jpg: 480x640 3 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1442.jpg: 480x640 3 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1443.jpg: 480x640 2 LegoBricks, 267.5ms
    Speed: 7.0ms preprocess, 267.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1444.jpg: 480x640 3 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1445.jpg: 480x640 5 LegoBricks, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1446.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 3.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1447.jpg: 480x640 5 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1448.jpg: 480x640 9 LegoBricks, 116.0ms
    Speed: 6.0ms preprocess, 116.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1449.jpg: 480x640 7 LegoBricks, 111.6ms
    Speed: 5.0ms preprocess, 111.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_145.jpg: 480x640 3 LegoBricks, 98.7ms
    Speed: 5.0ms preprocess, 98.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1450.jpg: 480x640 5 LegoBricks, 90.8ms
    Speed: 4.0ms preprocess, 90.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1451.jpg: 480x640 2 LegoBricks, 106.7ms
    Speed: 5.0ms preprocess, 106.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1452.jpg: 480x640 5 LegoBricks, 100.4ms
    Speed: 5.0ms preprocess, 100.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1453.jpg: 480x640 3 LegoBricks, 101.6ms
    Speed: 4.0ms preprocess, 101.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1454.jpg: 480x640 1 LegoBrick, 111.7ms
    Speed: 57.8ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1455.jpg: 480x640 2 LegoBricks, 110.9ms
    Speed: 5.0ms preprocess, 110.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1456.jpg: 480x640 1 LegoBrick, 106.7ms
    Speed: 5.3ms preprocess, 106.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1457.jpg: 480x640 4 LegoBricks, 116.8ms
    Speed: 5.0ms preprocess, 116.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1458.jpg: 480x640 5 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1459.jpg: 480x640 6 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_146.jpg: 480x640 4 LegoBricks, 197.5ms
    Speed: 5.0ms preprocess, 197.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1460.jpg: 480x640 7 LegoBricks, 182.5ms
    Speed: 14.9ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1461.jpg: 480x640 6 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1462.jpg: 480x640 3 LegoBricks, 114.7ms
    Speed: 6.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1463.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1464.jpg: 480x640 4 LegoBricks, 134.5ms
    Speed: 7.0ms preprocess, 134.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1465.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 7.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1466.jpg: 480x640 3 LegoBricks, 130.0ms
    Speed: 7.0ms preprocess, 130.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1467.jpg: 480x640 4 LegoBricks, 148.6ms
    Speed: 5.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1468.jpg: 480x640 7 LegoBricks, 154.6ms
    Speed: 5.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1469.jpg: 480x640 3 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_147.jpg: 480x640 5 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1470.jpg: 480x640 6 LegoBricks, 124.0ms
    Speed: 5.0ms preprocess, 124.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1472.jpg: 480x640 5 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1473.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1474.jpg: 480x640 4 LegoBricks, 114.7ms
    Speed: 4.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1475.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1476.jpg: 480x640 6 LegoBricks, 168.0ms
    Speed: 4.0ms preprocess, 168.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1477.jpg: 480x640 5 LegoBricks, 381.9ms
    Speed: 6.0ms preprocess, 381.9ms inference, 1.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1478.jpg: 480x640 3 LegoBricks, 282.2ms
    Speed: 7.0ms preprocess, 282.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_148.jpg: 480x640 1 LegoBrick, 232.4ms
    Speed: 27.9ms preprocess, 232.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1480.jpg: 480x640 5 LegoBricks, 131.5ms
    Speed: 9.0ms preprocess, 131.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1481.jpg: 480x640 2 LegoBricks, 116.9ms
    Speed: 5.0ms preprocess, 116.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1482.jpg: 480x640 1 LegoBrick, 105.7ms
    Speed: 6.0ms preprocess, 105.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1483.jpg: 480x640 1 LegoBrick, 106.7ms
    Speed: 5.0ms preprocess, 106.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1484.jpg: 480x640 2 LegoBricks, 120.5ms
    Speed: 6.0ms preprocess, 120.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1485.jpg: 480x640 4 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1486.jpg: 480x640 4 LegoBricks, 127.7ms
    Speed: 5.0ms preprocess, 127.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1487.jpg: 480x640 3 LegoBricks, 174.8ms
    Speed: 5.0ms preprocess, 174.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1488.jpg: 480x640 4 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1489.jpg: 480x640 3 LegoBricks, 113.4ms
    Speed: 6.0ms preprocess, 113.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_149.jpg: 480x640 3 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1490.jpg: 480x640 3 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1492.jpg: 480x640 1 LegoBrick, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1493.jpg: 480x640 1 LegoBrick, 125.0ms
    Speed: 46.9ms preprocess, 125.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1494.jpg: 480x640 2 LegoBricks, 120.5ms
    Speed: 5.0ms preprocess, 120.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1496.jpg: 480x640 3 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1497.jpg: 480x640 3 LegoBricks, 241.4ms
    Speed: 5.0ms preprocess, 241.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1498.jpg: 480x640 3 LegoBricks, 170.5ms
    Speed: 4.0ms preprocess, 170.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1499.jpg: 480x640 3 LegoBricks, 132.2ms
    Speed: 5.0ms preprocess, 132.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_15.jpg: 640x480 6 LegoBricks, 104.5ms
    Speed: 4.0ms preprocess, 104.5ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_150.jpg: 480x640 3 LegoBricks, 98.7ms
    Speed: 4.0ms preprocess, 98.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1500.jpg: 480x640 2 LegoBricks, 103.5ms
    Speed: 4.0ms preprocess, 103.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1501.jpg: 480x640 1 LegoBrick, 110.7ms
    Speed: 7.0ms preprocess, 110.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1502.jpg: 480x640 2 LegoBricks, 105.7ms
    Speed: 6.0ms preprocess, 105.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1503.jpg: 480x640 1 LegoBrick, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1504.jpg: 480x640 3 LegoBricks, 117.7ms
    Speed: 5.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1505.jpg: 480x640 7 LegoBricks, 118.7ms
    Speed: 6.1ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1506.jpg: 480x640 5 LegoBricks, 128.6ms
    Speed: 5.0ms preprocess, 128.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1507.jpg: 480x640 3 LegoBricks, 130.5ms
    Speed: 5.0ms preprocess, 130.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1508.jpg: 480x640 3 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1509.jpg: 480x640 1 LegoBrick, 112.1ms
    Speed: 5.0ms preprocess, 112.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_151.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1510.jpg: 480x640 5 LegoBricks, 111.7ms
    Speed: 41.9ms preprocess, 111.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1511.jpg: 480x640 7 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1512.jpg: 480x640 7 LegoBricks, 122.9ms
    Speed: 5.0ms preprocess, 122.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1513.jpg: 480x640 2 LegoBricks, 122.7ms
    Speed: 7.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1514.jpg: 480x640 4 LegoBricks, 256.0ms
    Speed: 7.0ms preprocess, 256.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1515.jpg: 480x640 5 LegoBricks, 207.4ms
    Speed: 7.0ms preprocess, 207.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1516.jpg: 480x640 5 LegoBricks, 145.8ms
    Speed: 5.0ms preprocess, 145.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1517.jpg: 480x640 8 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1518.jpg: 480x640 4 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1519.jpg: 480x640 4 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_152.jpg: 480x640 4 LegoBricks, 127.4ms
    Speed: 6.0ms preprocess, 127.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1520.jpg: 480x640 4 LegoBricks, 116.2ms
    Speed: 5.0ms preprocess, 116.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1521.jpg: 480x640 2 LegoBricks, 111.7ms
    Speed: 4.0ms preprocess, 111.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1522.jpg: 480x640 2 LegoBricks, 113.9ms
    Speed: 5.0ms preprocess, 113.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1523.jpg: 480x640 2 LegoBricks, 141.4ms
    Speed: 5.0ms preprocess, 141.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1524.jpg: 480x640 1 LegoBrick, 158.6ms
    Speed: 5.0ms preprocess, 158.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1525.jpg: 480x640 1 LegoBrick, 132.1ms
    Speed: 5.0ms preprocess, 132.1ms inference, 1.5ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1526.jpg: 480x640 1 LegoBrick, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1527.jpg: 480x640 5 LegoBricks, 137.1ms
    Speed: 5.0ms preprocess, 137.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1528.jpg: 480x640 9 LegoBricks, 137.6ms
    Speed: 11.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1529.jpg: 480x640 10 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_153.jpg: 480x640 3 LegoBricks, 226.4ms
    Speed: 20.9ms preprocess, 226.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1530.jpg: 480x640 3 LegoBricks, 162.6ms
    Speed: 13.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1531.jpg: 480x640 3 LegoBricks, 140.5ms
    Speed: 5.0ms preprocess, 140.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1532.jpg: 480x640 4 LegoBricks, 125.5ms
    Speed: 6.0ms preprocess, 125.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1533.jpg: 480x640 1 LegoBrick, 117.7ms
    Speed: 6.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1534.jpg: 480x640 2 LegoBricks, 154.8ms
    Speed: 7.0ms preprocess, 154.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1536.jpg: 480x640 5 LegoBricks, 124.2ms
    Speed: 6.0ms preprocess, 124.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1537.jpg: 480x640 7 LegoBricks, 112.4ms
    Speed: 6.0ms preprocess, 112.4ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1538.jpg: 480x640 7 LegoBricks, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1539.jpg: 480x640 8 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1540.jpg: 480x640 7 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1541.jpg: 480x640 5 LegoBricks, 116.0ms
    Speed: 6.0ms preprocess, 116.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1542.jpg: 480x640 6 LegoBricks, 121.6ms
    Speed: 5.0ms preprocess, 121.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1543.jpg: 480x640 2 LegoBricks, 188.1ms
    Speed: 5.0ms preprocess, 188.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1544.jpg: 480x640 3 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1545.jpg: 480x640 3 LegoBricks, 99.7ms
    Speed: 5.0ms preprocess, 99.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1546.jpg: 480x640 3 LegoBricks, 106.7ms
    Speed: 4.0ms preprocess, 106.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1547.jpg: 480x640 8 LegoBricks, 135.0ms
    Speed: 4.0ms preprocess, 135.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1548.jpg: 480x640 3 LegoBricks, 193.4ms
    Speed: 4.0ms preprocess, 193.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1549.jpg: 480x640 4 LegoBricks, 133.4ms
    Speed: 5.0ms preprocess, 133.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_155.jpg: 480x640 5 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1550.jpg: 480x640 3 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 12.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1551.jpg: 480x640 4 LegoBricks, 279.4ms
    Speed: 7.0ms preprocess, 279.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1552.jpg: 480x640 4 LegoBricks, 141.6ms
    Speed: 5.0ms preprocess, 141.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1553.jpg: 480x640 3 LegoBricks, 106.2ms
    Speed: 6.0ms preprocess, 106.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1554.jpg: 480x640 4 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1555.jpg: 480x640 2 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1556.jpg: 480x640 6 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1557.jpg: 480x640 7 LegoBricks, 118.7ms
    Speed: 21.9ms preprocess, 118.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1558.jpg: 480x640 8 LegoBricks, 103.7ms
    Speed: 5.0ms preprocess, 103.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1559.jpg: 480x640 4 LegoBricks, 141.2ms
    Speed: 4.0ms preprocess, 141.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_156.jpg: 480x640 3 LegoBricks, 123.9ms
    Speed: 6.0ms preprocess, 123.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1560.jpg: 480x640 4 LegoBricks, 146.4ms
    Speed: 6.0ms preprocess, 146.4ms inference, 1.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1561.jpg: 480x640 6 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1562.jpg: 480x640 6 LegoBricks, 128.7ms
    Speed: 6.0ms preprocess, 128.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1563.jpg: 480x640 4 LegoBricks, 219.7ms
    Speed: 18.0ms preprocess, 219.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1564.jpg: 480x640 5 LegoBricks, 229.9ms
    Speed: 6.0ms preprocess, 229.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1565.jpg: 480x640 9 LegoBricks, 167.1ms
    Speed: 7.0ms preprocess, 167.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1566.jpg: 480x640 9 LegoBricks, 312.4ms
    Speed: 5.0ms preprocess, 312.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1568.jpg: 480x640 9 LegoBricks, 207.0ms
    Speed: 21.9ms preprocess, 207.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1569.jpg: 480x640 3 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1570.jpg: 480x640 8 LegoBricks, 140.6ms
    Speed: 5.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1571.jpg: 480x640 9 LegoBricks, 126.8ms
    Speed: 5.0ms preprocess, 126.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1572.jpg: 480x640 6 LegoBricks, 121.7ms
    Speed: 6.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1573.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1574.jpg: 480x640 8 LegoBricks, 155.6ms
    Speed: 6.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1575.jpg: 480x640 8 LegoBricks, 148.1ms
    Speed: 6.0ms preprocess, 148.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1577.jpg: 480x640 11 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1578.jpg: 480x640 8 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1579.jpg: 480x640 5 LegoBricks, 188.5ms
    Speed: 6.0ms preprocess, 188.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_158.jpg: 480x640 4 LegoBricks, 234.3ms
    Speed: 95.7ms preprocess, 234.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1580.jpg: 480x640 5 LegoBricks, 157.6ms
    Speed: 7.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1581.jpg: 480x640 7 LegoBricks, 136.6ms
    Speed: 7.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1582.jpg: 480x640 8 LegoBricks, 134.6ms
    Speed: 6.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1583.jpg: 480x640 6 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1584.jpg: 480x640 8 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1585.jpg: 480x640 8 LegoBricks, 130.7ms
    Speed: 5.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1586.jpg: 480x640 7 LegoBricks, 139.6ms
    Speed: 6.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1587.jpg: 480x640 7 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1588.jpg: 480x640 5 LegoBricks, 126.7ms
    Speed: 6.0ms preprocess, 126.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1589.jpg: 480x640 6 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_159.jpg: 480x640 4 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1590.jpg: 480x640 2 LegoBricks, 115.7ms
    Speed: 4.0ms preprocess, 115.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1591.jpg: 480x640 7 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1592.jpg: 480x640 9 LegoBricks, 126.7ms
    Speed: 6.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1593.jpg: 480x640 10 LegoBricks, 128.7ms
    Speed: 5.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1594.jpg: 480x640 5 LegoBricks, 265.3ms
    Speed: 6.0ms preprocess, 265.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1599.jpg: 480x640 6 LegoBricks, 225.4ms
    Speed: 6.0ms preprocess, 225.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_16.jpg: 640x480 6 LegoBricks, 230.3ms
    Speed: 103.7ms preprocess, 230.3ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1600.jpg: 480x640 7 LegoBricks, 145.3ms
    Speed: 5.0ms preprocess, 145.3ms inference, 1.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1601.jpg: 480x640 6 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1602.jpg: 480x640 6 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1603.jpg: 480x640 6 LegoBricks, 120.7ms
    Speed: 6.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1605.jpg: 480x640 9 LegoBricks, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1606.jpg: 480x640 7 LegoBricks, 169.5ms
    Speed: 7.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1607.jpg: 480x640 7 LegoBricks, 173.0ms
    Speed: 6.0ms preprocess, 173.0ms inference, 2.8ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1609.jpg: 480x640 6 LegoBricks, 175.5ms
    Speed: 7.0ms preprocess, 175.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_161.jpg: 480x640 2 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1610.jpg: 480x640 4 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1611.jpg: 480x640 8 LegoBricks, 123.8ms
    Speed: 6.0ms preprocess, 123.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1612.jpg: 480x640 8 LegoBricks, 122.2ms
    Speed: 5.0ms preprocess, 122.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1613.jpg: 480x640 12 LegoBricks, 125.6ms
    Speed: 6.0ms preprocess, 125.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1614.jpg: 480x640 8 LegoBricks, 187.5ms
    Speed: 7.0ms preprocess, 187.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1615.jpg: 480x640 8 LegoBricks, 292.2ms
    Speed: 29.9ms preprocess, 292.2ms inference, 5.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1616.jpg: 480x640 6 LegoBricks, 341.4ms
    Speed: 7.0ms preprocess, 341.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1617.jpg: 480x640 7 LegoBricks, 290.1ms
    Speed: 9.0ms preprocess, 290.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1618.jpg: 480x640 3 LegoBricks, 601.9ms
    Speed: 5.0ms preprocess, 601.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1619.jpg: 480x640 8 LegoBricks, 247.3ms
    Speed: 23.9ms preprocess, 247.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_162.jpg: 480x640 2 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1620.jpg: 480x640 8 LegoBricks, 205.4ms
    Speed: 6.0ms preprocess, 205.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1621.jpg: 480x640 7 LegoBricks, 167.6ms
    Speed: 6.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1622.jpg: 480x640 9 LegoBricks, 211.5ms
    Speed: 7.1ms preprocess, 211.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1623.jpg: 480x640 13 LegoBricks, 139.0ms
    Speed: 6.0ms preprocess, 139.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1624.jpg: 480x640 6 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1625.jpg: 480x640 17 LegoBricks, 153.5ms
    Speed: 5.0ms preprocess, 153.5ms inference, 15.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1626.jpg: 480x640 3 LegoBricks, 170.4ms
    Speed: 6.1ms preprocess, 170.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1627.jpg: 480x640 8 LegoBricks, 186.5ms
    Speed: 7.0ms preprocess, 186.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1628.jpg: 480x640 7 LegoBricks, 271.3ms
    Speed: 5.0ms preprocess, 271.3ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1629.jpg: 480x640 7 LegoBricks, 367.9ms
    Speed: 11.0ms preprocess, 367.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_163.jpg: 480x640 3 LegoBricks, 189.5ms
    Speed: 5.0ms preprocess, 189.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1630.jpg: 480x640 9 LegoBricks, 149.5ms
    Speed: 6.0ms preprocess, 149.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1631.jpg: 480x640 8 LegoBricks, 504.1ms
    Speed: 14.0ms preprocess, 504.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1632.jpg: 480x640 4 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1633.jpg: 480x640 7 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1634.jpg: 480x640 8 LegoBricks, 123.9ms
    Speed: 5.0ms preprocess, 123.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1635.jpg: 480x640 8 LegoBricks, 117.5ms
    Speed: 6.0ms preprocess, 117.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1636.jpg: 480x640 9 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1637.jpg: 480x640 10 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1638.jpg: 480x640 9 LegoBricks, 155.8ms
    Speed: 5.0ms preprocess, 155.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1639.jpg: 480x640 7 LegoBricks, 203.4ms
    Speed: 6.0ms preprocess, 203.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1640.jpg: 480x640 7 LegoBricks, 234.4ms
    Speed: 7.0ms preprocess, 234.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1641.jpg: 480x640 6 LegoBricks, 157.6ms
    Speed: 7.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1642.jpg: 480x640 10 LegoBricks, 516.6ms
    Speed: 5.0ms preprocess, 516.6ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1644.jpg: 480x640 6 LegoBricks, 387.6ms
    Speed: 4.0ms preprocess, 387.6ms inference, 9.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1645.jpg: 480x640 6 LegoBricks, 236.5ms
    Speed: 9.0ms preprocess, 236.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1646.jpg: 480x640 5 LegoBricks, 236.0ms
    Speed: 25.9ms preprocess, 236.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1647.jpg: 480x640 6 LegoBricks, 242.8ms
    Speed: 14.0ms preprocess, 242.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1648.jpg: 480x640 4 LegoBricks, 206.4ms
    Speed: 8.0ms preprocess, 206.4ms inference, 3.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1649.jpg: 480x640 6 LegoBricks, 241.4ms
    Speed: 9.0ms preprocess, 241.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_165.jpg: 480x640 1 LegoBrick, 167.5ms
    Speed: 9.0ms preprocess, 167.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1651.jpg: 480x640 4 LegoBricks, 159.6ms
    Speed: 5.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1652.jpg: 480x640 5 LegoBricks, 133.6ms
    Speed: 6.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1653.jpg: 480x640 8 LegoBricks, 137.1ms
    Speed: 6.0ms preprocess, 137.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1654.jpg: 480x640 4 LegoBricks, 158.6ms
    Speed: 6.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1655.jpg: 480x640 5 LegoBricks, 157.6ms
    Speed: 6.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1656.jpg: 480x640 6 LegoBricks, 161.5ms
    Speed: 22.9ms preprocess, 161.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1657.jpg: 480x640 5 LegoBricks, 521.3ms
    Speed: 6.0ms preprocess, 521.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1658.jpg: 480x640 6 LegoBricks, 153.6ms
    Speed: 7.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1659.jpg: 480x640 7 LegoBricks, 131.9ms
    Speed: 6.0ms preprocess, 131.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_166.jpg: 480x640 3 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1660.jpg: 480x640 7 LegoBricks, 156.9ms
    Speed: 5.0ms preprocess, 156.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1661.jpg: 480x640 4 LegoBricks, 156.1ms
    Speed: 6.0ms preprocess, 156.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1662.jpg: 480x640 4 LegoBricks, 219.4ms
    Speed: 7.0ms preprocess, 219.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1663.jpg: 480x640 2 LegoBricks, 174.9ms
    Speed: 6.0ms preprocess, 174.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1664.jpg: 480x640 2 LegoBricks, 102.7ms
    Speed: 5.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1665.jpg: 480x640 3 LegoBricks, 97.4ms
    Speed: 5.0ms preprocess, 97.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1666.jpg: 480x640 3 LegoBricks, 161.0ms
    Speed: 5.0ms preprocess, 161.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1667.jpg: 480x640 5 LegoBricks, 111.7ms
    Speed: 6.0ms preprocess, 111.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1668.jpg: 480x640 3 LegoBricks, 112.7ms
    Speed: 5.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1669.jpg: 480x640 6 LegoBricks, 112.7ms
    Speed: 6.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_167.jpg: 480x640 4 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1670.jpg: 480x640 4 LegoBricks, 129.7ms
    Speed: 5.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1671.jpg: 480x640 4 LegoBricks, 117.6ms
    Speed: 5.0ms preprocess, 117.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1672.jpg: 480x640 4 LegoBricks, 102.7ms
    Speed: 5.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1673.jpg: 480x640 5 LegoBricks, 105.7ms
    Speed: 5.1ms preprocess, 105.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1674.jpg: 480x640 8 LegoBricks, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 5.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1675.jpg: 480x640 6 LegoBricks, 300.2ms
    Speed: 7.0ms preprocess, 300.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1676.jpg: 480x640 5 LegoBricks, 169.0ms
    Speed: 7.0ms preprocess, 169.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1677.jpg: 480x640 6 LegoBricks, 121.6ms
    Speed: 6.0ms preprocess, 121.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1678.jpg: 480x640 8 LegoBricks, 348.1ms
    Speed: 8.0ms preprocess, 348.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1679.jpg: 480x640 7 LegoBricks, 190.5ms
    Speed: 7.0ms preprocess, 190.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_168.jpg: 480x640 3 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1680.jpg: 480x640 8 LegoBricks, 117.3ms
    Speed: 5.0ms preprocess, 117.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1681.jpg: 480x640 8 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1682.jpg: 480x640 11 LegoBricks, 123.7ms
    Speed: 6.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1683.jpg: 480x640 5 LegoBricks, 117.7ms
    Speed: 5.0ms preprocess, 117.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1684.jpg: 480x640 5 LegoBricks, 118.7ms
    Speed: 4.5ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1685.jpg: 480x640 7 LegoBricks, 129.7ms
    Speed: 10.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1686.jpg: 480x640 8 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1687.jpg: 480x640 7 LegoBricks, 101.2ms
    Speed: 5.0ms preprocess, 101.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1688.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1689.jpg: 480x640 4 LegoBricks, 178.5ms
    Speed: 5.0ms preprocess, 178.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1690.jpg: 480x640 7 LegoBricks, 342.1ms
    Speed: 6.0ms preprocess, 342.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1691.jpg: 480x640 5 LegoBricks, 165.6ms
    Speed: 6.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1692.jpg: 480x640 6 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1693.jpg: 480x640 6 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1694.jpg: 480x640 5 LegoBricks, 163.5ms
    Speed: 6.0ms preprocess, 163.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1695.jpg: 480x640 3 LegoBricks, 147.2ms
    Speed: 6.0ms preprocess, 147.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1696.jpg: 480x640 6 LegoBricks, 146.5ms
    Speed: 6.1ms preprocess, 146.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1697.jpg: 480x640 9 LegoBricks, 146.5ms
    Speed: 6.0ms preprocess, 146.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1698.jpg: 480x640 5 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1699.jpg: 480x640 3 LegoBricks, 173.5ms
    Speed: 5.0ms preprocess, 173.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_17.jpg: 640x480 2 LegoBricks, 145.6ms
    Speed: 6.4ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1700.jpg: 480x640 5 LegoBricks, 139.4ms
    Speed: 6.0ms preprocess, 139.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1701.jpg: 480x640 9 LegoBricks, 130.7ms
    Speed: 4.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1702.jpg: 480x640 5 LegoBricks, 180.8ms
    Speed: 5.0ms preprocess, 180.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1703.jpg: 480x640 5 LegoBricks, 148.6ms
    Speed: 5.0ms preprocess, 148.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1704.jpg: 480x640 7 LegoBricks, 287.2ms
    Speed: 6.0ms preprocess, 287.2ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1705.jpg: 480x640 8 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1706.jpg: 480x640 5 LegoBricks, 117.7ms
    Speed: 6.0ms preprocess, 117.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1707.jpg: 480x640 3 LegoBricks, 237.1ms
    Speed: 19.9ms preprocess, 237.1ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1708.jpg: 480x640 8 LegoBricks, 165.5ms
    Speed: 14.0ms preprocess, 165.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1709.jpg: 480x640 6 LegoBricks, 108.1ms
    Speed: 4.0ms preprocess, 108.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_171.jpg: 480x640 1 LegoBrick, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1711.jpg: 480x640 4 LegoBricks, 128.7ms
    Speed: 4.9ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1712.jpg: 480x640 3 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1713.jpg: 480x640 3 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1714.jpg: 480x640 5 LegoBricks, 100.6ms
    Speed: 6.0ms preprocess, 100.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1715.jpg: 480x640 6 LegoBricks, 113.4ms
    Speed: 5.0ms preprocess, 113.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1716.jpg: 480x640 10 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1718.jpg: 480x640 3 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1719.jpg: 480x640 8 LegoBricks, 106.9ms
    Speed: 5.0ms preprocess, 106.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_172.jpg: 480x640 5 LegoBricks, 104.7ms
    Speed: 5.0ms preprocess, 104.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1720.jpg: 480x640 4 LegoBricks, 117.7ms
    Speed: 6.0ms preprocess, 117.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1721.jpg: 480x640 2 LegoBricks, 499.5ms
    Speed: 37.9ms preprocess, 499.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1722.jpg: 480x640 5 LegoBricks, 139.6ms
    Speed: 6.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1723.jpg: 480x640 3 LegoBricks, 118.8ms
    Speed: 5.0ms preprocess, 118.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1724.jpg: 480x640 4 LegoBricks, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1725.jpg: 480x640 5 LegoBricks, 106.7ms
    Speed: 5.0ms preprocess, 106.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1726.jpg: 480x640 5 LegoBricks, 123.7ms
    Speed: 6.4ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1727.jpg: 480x640 7 LegoBricks, 120.4ms
    Speed: 5.0ms preprocess, 120.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1728.jpg: 480x640 7 LegoBricks, 120.7ms
    Speed: 6.0ms preprocess, 120.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1729.jpg: 480x640 3 LegoBricks, 289.7ms
    Speed: 5.0ms preprocess, 289.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_173.jpg: 480x640 4 LegoBricks, 172.3ms
    Speed: 5.0ms preprocess, 172.3ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1730.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1731.jpg: 480x640 5 LegoBricks, 100.0ms
    Speed: 5.0ms preprocess, 100.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1732.jpg: 480x640 6 LegoBricks, 113.6ms
    Speed: 5.0ms preprocess, 113.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1733.jpg: 480x640 2 LegoBricks, 105.8ms
    Speed: 6.0ms preprocess, 105.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1734.jpg: 480x640 3 LegoBricks, 125.5ms
    Speed: 4.0ms preprocess, 125.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1735.jpg: 480x640 2 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1736.jpg: 480x640 7 LegoBricks, 139.6ms
    Speed: 7.0ms preprocess, 139.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1737.jpg: 480x640 3 LegoBricks, 124.9ms
    Speed: 6.0ms preprocess, 124.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1738.jpg: 480x640 5 LegoBricks, 144.6ms
    Speed: 27.9ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_174.jpg: 480x640 4 LegoBricks, 154.0ms
    Speed: 6.0ms preprocess, 154.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1740.jpg: 480x640 4 LegoBricks, 165.6ms
    Speed: 6.0ms preprocess, 165.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1741.jpg: 480x640 5 LegoBricks, 298.9ms
    Speed: 25.9ms preprocess, 298.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1742.jpg: 480x640 5 LegoBricks, 215.4ms
    Speed: 6.0ms preprocess, 215.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1743.jpg: 480x640 3 LegoBricks, 179.6ms
    Speed: 9.0ms preprocess, 179.6ms inference, 2.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1744.jpg: 480x640 4 LegoBricks, 136.6ms
    Speed: 5.9ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1745.jpg: 480x640 4 LegoBricks, 142.1ms
    Speed: 6.0ms preprocess, 142.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1746.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1747.jpg: 480x640 3 LegoBricks, 152.6ms
    Speed: 6.0ms preprocess, 152.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1748.jpg: 480x640 5 LegoBricks, 150.6ms
    Speed: 6.0ms preprocess, 150.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1749.jpg: 480x640 3 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_175.jpg: 480x640 5 LegoBricks, 118.9ms
    Speed: 5.0ms preprocess, 118.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1751.jpg: 480x640 2 LegoBricks, 135.6ms
    Speed: 5.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1752.jpg: 480x640 6 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1753.jpg: 480x640 2 LegoBricks, 102.5ms
    Speed: 2.0ms preprocess, 102.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1754.jpg: 480x640 3 LegoBricks, 95.7ms
    Speed: 5.0ms preprocess, 95.7ms inference, 2.2ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1755.jpg: 480x640 6 LegoBricks, 151.6ms
    Speed: 4.0ms preprocess, 151.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1756.jpg: 480x640 4 LegoBricks, 100.7ms
    Speed: 3.0ms preprocess, 100.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1757.jpg: 480x640 4 LegoBricks, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1758.jpg: 480x640 4 LegoBricks, 304.6ms
    Speed: 29.9ms preprocess, 304.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1759.jpg: 480x640 3 LegoBricks, 184.7ms
    Speed: 9.0ms preprocess, 184.7ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_176.jpg: 480x640 2 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1760.jpg: 480x640 3 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1761.jpg: 480x640 4 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1762.jpg: 480x640 4 LegoBricks, 106.7ms
    Speed: 6.0ms preprocess, 106.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1763.jpg: 480x640 4 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1764.jpg: 480x640 4 LegoBricks, 106.0ms
    Speed: 4.0ms preprocess, 106.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1765.jpg: 480x640 5 LegoBricks, 104.7ms
    Speed: 3.0ms preprocess, 104.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1766.jpg: 480x640 2 LegoBricks, 112.4ms
    Speed: 5.0ms preprocess, 112.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1767.jpg: 480x640 3 LegoBricks, 196.5ms
    Speed: 5.0ms preprocess, 196.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1768.jpg: 480x640 4 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1769.jpg: 480x640 3 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_177.jpg: 480x640 2 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1770.jpg: 480x640 2 LegoBricks, 171.5ms
    Speed: 6.0ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1771.jpg: 480x640 4 LegoBricks, 103.7ms
    Speed: 6.0ms preprocess, 103.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1772.jpg: 480x640 4 LegoBricks, 124.1ms
    Speed: 5.0ms preprocess, 124.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1773.jpg: 480x640 6 LegoBricks, 110.6ms
    Speed: 5.0ms preprocess, 110.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1774.jpg: 480x640 8 LegoBricks, 251.8ms
    Speed: 6.0ms preprocess, 251.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1775.jpg: 480x640 4 LegoBricks, 205.3ms
    Speed: 10.0ms preprocess, 205.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1776.jpg: 480x640 8 LegoBricks, 145.6ms
    Speed: 5.0ms preprocess, 145.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1777.jpg: 480x640 9 LegoBricks, 284.2ms
    Speed: 5.0ms preprocess, 284.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1778.jpg: 480x640 7 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1779.jpg: 480x640 3 LegoBricks, 110.6ms
    Speed: 6.0ms preprocess, 110.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_178.jpg: 480x640 4 LegoBricks, 119.3ms
    Speed: 6.0ms preprocess, 119.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1780.jpg: 480x640 3 LegoBricks, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1781.jpg: 480x640 3 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1782.jpg: 480x640 2 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1783.jpg: 480x640 4 LegoBricks, 168.5ms
    Speed: 6.0ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1784.jpg: 480x640 6 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1785.jpg: 480x640 7 LegoBricks, 135.4ms
    Speed: 7.0ms preprocess, 135.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1786.jpg: 480x640 3 LegoBricks, 126.7ms
    Speed: 6.0ms preprocess, 126.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1787.jpg: 480x640 4 LegoBricks, 138.6ms
    Speed: 42.9ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1788.jpg: 480x640 6 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1789.jpg: 480x640 5 LegoBricks, 377.7ms
    Speed: 9.0ms preprocess, 377.7ms inference, 5.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_179.jpg: 480x640 5 LegoBricks, 192.5ms
    Speed: 35.9ms preprocess, 192.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1790.jpg: 480x640 10 LegoBricks, 280.7ms
    Speed: 7.0ms preprocess, 280.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1791.jpg: 480x640 2 LegoBricks, 232.4ms
    Speed: 6.0ms preprocess, 232.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1792.jpg: 480x640 3 LegoBricks, 127.1ms
    Speed: 6.0ms preprocess, 127.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1793.jpg: 480x640 3 LegoBricks, 109.6ms
    Speed: 6.0ms preprocess, 109.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1794.jpg: 480x640 3 LegoBricks, 106.4ms
    Speed: 6.0ms preprocess, 106.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1795.jpg: 480x640 6 LegoBricks, 101.7ms
    Speed: 5.0ms preprocess, 101.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1796.jpg: 480x640 4 LegoBricks, 93.8ms
    Speed: 4.0ms preprocess, 93.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1797.jpg: 480x640 3 LegoBricks, 93.7ms
    Speed: 5.0ms preprocess, 93.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1798.jpg: 480x640 3 LegoBricks, 103.7ms
    Speed: 5.0ms preprocess, 103.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1799.jpg: 480x640 3 LegoBricks, 131.3ms
    Speed: 5.0ms preprocess, 131.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_18.jpg: 640x480 6 LegoBricks, 118.7ms
    Speed: 6.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_180.jpg: 480x640 4 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1800.jpg: 480x640 3 LegoBricks, 121.4ms
    Speed: 5.0ms preprocess, 121.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1801.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1802.jpg: 480x640 3 LegoBricks, 329.1ms
    Speed: 4.0ms preprocess, 329.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1803.jpg: 480x640 3 LegoBricks, 217.4ms
    Speed: 6.0ms preprocess, 217.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1804.jpg: 480x640 4 LegoBricks, 116.1ms
    Speed: 5.0ms preprocess, 116.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1805.jpg: 480x640 3 LegoBricks, 114.5ms
    Speed: 4.0ms preprocess, 114.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1806.jpg: 480x640 3 LegoBricks, 105.7ms
    Speed: 4.9ms preprocess, 105.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1807.jpg: 480x640 4 LegoBricks, 152.6ms
    Speed: 5.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1808.jpg: 480x640 3 LegoBricks, 196.4ms
    Speed: 7.0ms preprocess, 196.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1809.jpg: 480x640 3 LegoBricks, 360.1ms
    Speed: 14.0ms preprocess, 360.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_181.jpg: 480x640 5 LegoBricks, 137.7ms
    Speed: 5.0ms preprocess, 137.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1810.jpg: 480x640 5 LegoBricks, 178.5ms
    Speed: 5.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1811.jpg: 480x640 3 LegoBricks, 118.3ms
    Speed: 6.0ms preprocess, 118.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1813.jpg: 480x640 4 LegoBricks, 108.5ms
    Speed: 6.0ms preprocess, 108.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1814.jpg: 480x640 5 LegoBricks, 110.7ms
    Speed: 6.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1815.jpg: 480x640 3 LegoBricks, 175.5ms
    Speed: 5.0ms preprocess, 175.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1816.jpg: 480x640 3 LegoBricks, 127.8ms
    Speed: 6.0ms preprocess, 127.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1817.jpg: 480x640 3 LegoBricks, 128.6ms
    Speed: 5.0ms preprocess, 128.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1818.jpg: 480x640 4 LegoBricks, 95.9ms
    Speed: 5.0ms preprocess, 95.9ms inference, 1.5ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1819.jpg: 480x640 4 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_182.jpg: 480x640 6 LegoBricks, 96.7ms
    Speed: 4.0ms preprocess, 96.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1820.jpg: 480x640 4 LegoBricks, 178.5ms
    Speed: 4.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1821.jpg: 480x640 4 LegoBricks, 209.4ms
    Speed: 10.0ms preprocess, 209.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1822.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 6.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1823.jpg: 480x640 5 LegoBricks, 159.6ms
    Speed: 7.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1824.jpg: 480x640 5 LegoBricks, 150.2ms
    Speed: 6.0ms preprocess, 150.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1825.jpg: 480x640 6 LegoBricks, 168.0ms
    Speed: 5.0ms preprocess, 168.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1826.jpg: 480x640 4 LegoBricks, 145.6ms
    Speed: 5.0ms preprocess, 145.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1827.jpg: 480x640 5 LegoBricks, 131.6ms
    Speed: 21.9ms preprocess, 131.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1828.jpg: 480x640 4 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1829.jpg: 480x640 4 LegoBricks, 147.4ms
    Speed: 6.0ms preprocess, 147.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_183.jpg: 480x640 7 LegoBricks, 166.6ms
    Speed: 5.0ms preprocess, 166.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1830.jpg: 480x640 4 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1831.jpg: 480x640 4 LegoBricks, 141.6ms
    Speed: 6.0ms preprocess, 141.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1832.jpg: 480x640 6 LegoBricks, 165.8ms
    Speed: 5.0ms preprocess, 165.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1833.jpg: 480x640 4 LegoBricks, 158.6ms
    Speed: 7.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1834.jpg: 480x640 4 LegoBricks, 153.9ms
    Speed: 5.0ms preprocess, 153.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1835.jpg: 480x640 5 LegoBricks, 424.9ms
    Speed: 6.0ms preprocess, 424.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1836.jpg: 480x640 5 LegoBricks, 180.5ms
    Speed: 7.0ms preprocess, 180.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1837.jpg: 480x640 7 LegoBricks, 107.7ms
    Speed: 6.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1838.jpg: 480x640 4 LegoBricks, 106.4ms
    Speed: 5.0ms preprocess, 106.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1839.jpg: 480x640 5 LegoBricks, 102.7ms
    Speed: 4.0ms preprocess, 102.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_184.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1840.jpg: 480x640 5 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1841.jpg: 480x640 7 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1842.jpg: 480x640 3 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1843.jpg: 480x640 4 LegoBricks, 117.7ms
    Speed: 5.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1844.jpg: 480x640 10 LegoBricks, 104.7ms
    Speed: 6.0ms preprocess, 104.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1845.jpg: 480x640 8 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1846.jpg: 480x640 6 LegoBricks, 112.7ms
    Speed: 5.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1847.jpg: 480x640 6 LegoBricks, 109.4ms
    Speed: 5.1ms preprocess, 109.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1848.jpg: 480x640 4 LegoBricks, 107.6ms
    Speed: 6.0ms preprocess, 107.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_185.jpg: 480x640 3 LegoBricks, 105.7ms
    Speed: 6.0ms preprocess, 105.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1850.jpg: 480x640 3 LegoBricks, 134.6ms
    Speed: 4.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1851.jpg: 480x640 3 LegoBricks, 261.3ms
    Speed: 5.0ms preprocess, 261.3ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1852.jpg: 480x640 3 LegoBricks, 311.3ms
    Speed: 9.0ms preprocess, 311.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1853.jpg: 480x640 3 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1854.jpg: 480x640 6 LegoBricks, 119.7ms
    Speed: 5.8ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1855.jpg: 480x640 6 LegoBricks, 106.7ms
    Speed: 6.0ms preprocess, 106.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1857.jpg: 480x640 3 LegoBricks, 129.7ms
    Speed: 6.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1858.jpg: 480x640 4 LegoBricks, 129.1ms
    Speed: 6.0ms preprocess, 129.1ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1859.jpg: 480x640 8 LegoBricks, 302.2ms
    Speed: 5.0ms preprocess, 302.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_186.jpg: 480x640 3 LegoBricks, 191.5ms
    Speed: 6.0ms preprocess, 191.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1860.jpg: 480x640 11 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1861.jpg: 480x640 9 LegoBricks, 120.1ms
    Speed: 5.0ms preprocess, 120.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1863.jpg: 480x640 6 LegoBricks, 135.6ms
    Speed: 5.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1864.jpg: 480x640 4 LegoBricks, 126.1ms
    Speed: 6.0ms preprocess, 126.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1865.jpg: 480x640 6 LegoBricks, 128.5ms
    Speed: 5.0ms preprocess, 128.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1866.jpg: 480x640 6 LegoBricks, 108.5ms
    Speed: 6.2ms preprocess, 108.5ms inference, 58.8ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1867.jpg: 480x640 5 LegoBricks, 300.2ms
    Speed: 11.0ms preprocess, 300.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1868.jpg: 480x640 2 LegoBricks, 195.5ms
    Speed: 7.0ms preprocess, 195.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1869.jpg: 480x640 8 LegoBricks, 248.7ms
    Speed: 9.0ms preprocess, 248.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_187.jpg: 480x640 4 LegoBricks, 165.5ms
    Speed: 6.0ms preprocess, 165.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1870.jpg: 480x640 6 LegoBricks, 169.5ms
    Speed: 5.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1871.jpg: 480x640 3 LegoBricks, 400.7ms
    Speed: 34.9ms preprocess, 400.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1872.jpg: 480x640 6 LegoBricks, 178.9ms
    Speed: 7.0ms preprocess, 178.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1873.jpg: 480x640 4 LegoBricks, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1874.jpg: 480x640 4 LegoBricks, 157.3ms
    Speed: 6.0ms preprocess, 157.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1875.jpg: 480x640 7 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1876.jpg: 480x640 6 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1877.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1879.jpg: 480x640 5 LegoBricks, 149.6ms
    Speed: 5.3ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_188.jpg: 480x640 5 LegoBricks, 143.6ms
    Speed: 70.1ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1880.jpg: 480x640 5 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1881.jpg: 480x640 9 LegoBricks, 121.6ms
    Speed: 5.0ms preprocess, 121.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1882.jpg: 480x640 7 LegoBricks, 317.1ms
    Speed: 12.0ms preprocess, 317.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1883.jpg: 480x640 2 LegoBricks, 161.6ms
    Speed: 5.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1884.jpg: 480x640 3 LegoBricks, 128.7ms
    Speed: 7.8ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1885.jpg: 480x640 6 LegoBricks, 195.4ms
    Speed: 7.0ms preprocess, 195.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1886.jpg: 480x640 6 LegoBricks, 221.2ms
    Speed: 5.0ms preprocess, 221.2ms inference, 6.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1887.jpg: 480x640 6 LegoBricks, 135.3ms
    Speed: 7.0ms preprocess, 135.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1888.jpg: 480x640 4 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1889.jpg: 480x640 5 LegoBricks, 135.2ms
    Speed: 8.0ms preprocess, 135.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_189.jpg: 480x640 2 LegoBricks, 146.3ms
    Speed: 6.0ms preprocess, 146.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1890.jpg: 480x640 5 LegoBricks, 173.5ms
    Speed: 7.0ms preprocess, 173.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1891.jpg: 480x640 5 LegoBricks, 193.5ms
    Speed: 6.0ms preprocess, 193.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1892.jpg: 480x640 3 LegoBricks, 175.5ms
    Speed: 7.0ms preprocess, 175.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1893.jpg: 480x640 3 LegoBricks, 166.6ms
    Speed: 10.0ms preprocess, 166.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1894.jpg: 480x640 5 LegoBricks, 177.7ms
    Speed: 5.0ms preprocess, 177.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1895.jpg: 480x640 9 LegoBricks, 235.7ms
    Speed: 6.0ms preprocess, 235.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1896.jpg: 480x640 11 LegoBricks, 255.3ms
    Speed: 6.0ms preprocess, 255.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1897.jpg: 480x640 9 LegoBricks, 189.3ms
    Speed: 5.0ms preprocess, 189.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1898.jpg: 480x640 3 LegoBricks, 158.6ms
    Speed: 7.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1899.jpg: 480x640 2 LegoBricks, 131.8ms
    Speed: 4.0ms preprocess, 131.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_19.jpg: 640x480 9 LegoBricks, 101.7ms
    Speed: 3.0ms preprocess, 101.7ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_190.jpg: 480x640 2 LegoBricks, 99.7ms
    Speed: 5.0ms preprocess, 99.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1900.jpg: 480x640 6 LegoBricks, 112.3ms
    Speed: 5.0ms preprocess, 112.3ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1901.jpg: 480x640 6 LegoBricks, 103.7ms
    Speed: 6.0ms preprocess, 103.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1902.jpg: 480x640 9 LegoBricks, 114.7ms
    Speed: 4.0ms preprocess, 114.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1903.jpg: 480x640 7 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1904.jpg: 480x640 9 LegoBricks, 119.7ms
    Speed: 4.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1905.jpg: 480x640 6 LegoBricks, 181.1ms
    Speed: 6.0ms preprocess, 181.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1906.jpg: 480x640 15 LegoBricks, 126.9ms
    Speed: 5.0ms preprocess, 126.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1907.jpg: 480x640 1 LegoBrick, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1908.jpg: 480x640 5 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1909.jpg: 480x640 5 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_191.jpg: 480x640 3 LegoBricks, 203.8ms
    Speed: 6.0ms preprocess, 203.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1910.jpg: 480x640 2 LegoBricks, 315.4ms
    Speed: 6.0ms preprocess, 315.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1911.jpg: 480x640 5 LegoBricks, 202.1ms
    Speed: 5.0ms preprocess, 202.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1912.jpg: 480x640 5 LegoBricks, 181.8ms
    Speed: 6.7ms preprocess, 181.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1913.jpg: 480x640 6 LegoBricks, 165.6ms
    Speed: 6.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1915.jpg: 480x640 6 LegoBricks, 145.6ms
    Speed: 6.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1916.jpg: 480x640 3 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1918.jpg: 480x640 3 LegoBricks, 131.1ms
    Speed: 5.0ms preprocess, 131.1ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1919.jpg: 480x640 3 LegoBricks, 160.5ms
    Speed: 6.0ms preprocess, 160.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_192.jpg: 480x640 4 LegoBricks, 148.6ms
    Speed: 53.6ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1921.jpg: 480x640 3 LegoBricks, 136.9ms
    Speed: 6.6ms preprocess, 136.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1922.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1923.jpg: 480x640 3 LegoBricks, 119.9ms
    Speed: 7.0ms preprocess, 119.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1924.jpg: 480x640 5 LegoBricks, 146.7ms
    Speed: 5.0ms preprocess, 146.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1925.jpg: 480x640 3 LegoBricks, 139.6ms
    Speed: 6.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1926.jpg: 480x640 5 LegoBricks, 122.7ms
    Speed: 4.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1927.jpg: 480x640 4 LegoBricks, 171.6ms
    Speed: 4.0ms preprocess, 171.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1928.jpg: 480x640 6 LegoBricks, 253.3ms
    Speed: 6.0ms preprocess, 253.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1929.jpg: 480x640 4 LegoBricks, 152.6ms
    Speed: 5.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_193.jpg: 480x640 4 LegoBricks, 189.5ms
    Speed: 6.0ms preprocess, 189.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1930.jpg: 480x640 2 LegoBricks, 327.2ms
    Speed: 30.9ms preprocess, 327.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1931.jpg: 480x640 1 LegoBrick, 190.4ms
    Speed: 8.0ms preprocess, 190.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1932.jpg: 480x640 1 LegoBrick, 161.6ms
    Speed: 5.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1934.jpg: 480x640 7 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1935.jpg: 480x640 8 LegoBricks, 241.9ms
    Speed: 7.0ms preprocess, 241.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1936.jpg: 480x640 5 LegoBricks, 145.5ms
    Speed: 6.0ms preprocess, 145.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1937.jpg: 480x640 3 LegoBricks, 140.1ms
    Speed: 6.0ms preprocess, 140.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1938.jpg: 480x640 9 LegoBricks, 176.5ms
    Speed: 6.0ms preprocess, 176.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1939.jpg: 480x640 4 LegoBricks, 132.9ms
    Speed: 6.0ms preprocess, 132.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1940.jpg: 480x640 6 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1941.jpg: 480x640 1 LegoBrick, 262.7ms
    Speed: 6.0ms preprocess, 262.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1942.jpg: 480x640 3 LegoBricks, 238.4ms
    Speed: 24.9ms preprocess, 238.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1943.jpg: 480x640 4 LegoBricks, 109.3ms
    Speed: 5.0ms preprocess, 109.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1944.jpg: 480x640 4 LegoBricks, 100.7ms
    Speed: 5.0ms preprocess, 100.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1945.jpg: 480x640 4 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1947.jpg: 480x640 3 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1949.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1950.jpg: 480x640 6 LegoBricks, 313.8ms
    Speed: 5.0ms preprocess, 313.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1951.jpg: 480x640 3 LegoBricks, 273.3ms
    Speed: 5.0ms preprocess, 273.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1952.jpg: 480x640 4 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1953.jpg: 480x640 8 LegoBricks, 116.7ms
    Speed: 5.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1954.jpg: 480x640 4 LegoBricks, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1955.jpg: 480x640 3 LegoBricks, 108.1ms
    Speed: 5.0ms preprocess, 108.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1956.jpg: 480x640 2 LegoBricks, 120.4ms
    Speed: 6.0ms preprocess, 120.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1957.jpg: 480x640 1 LegoBrick, 118.3ms
    Speed: 6.0ms preprocess, 118.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1958.jpg: 480x640 4 LegoBricks, 147.7ms
    Speed: 5.0ms preprocess, 147.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1959.jpg: 480x640 4 LegoBricks, 127.8ms
    Speed: 5.0ms preprocess, 127.8ms inference, 2.9ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_196.jpg: 480x640 2 LegoBricks, 198.8ms
    Speed: 5.0ms preprocess, 198.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1960.jpg: 480x640 3 LegoBricks, 181.2ms
    Speed: 6.0ms preprocess, 181.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1961.jpg: 480x640 5 LegoBricks, 351.9ms
    Speed: 25.9ms preprocess, 351.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1962.jpg: 480x640 6 LegoBricks, 196.5ms
    Speed: 7.0ms preprocess, 196.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1963.jpg: 480x640 5 LegoBricks, 128.7ms
    Speed: 6.0ms preprocess, 128.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1964.jpg: 480x640 7 LegoBricks, 154.3ms
    Speed: 6.0ms preprocess, 154.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1965.jpg: 480x640 6 LegoBricks, 173.5ms
    Speed: 5.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1966.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1967.jpg: 480x640 2 LegoBricks, 158.6ms
    Speed: 38.9ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1968.jpg: 480x640 5 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1969.jpg: 480x640 4 LegoBricks, 130.0ms
    Speed: 6.0ms preprocess, 130.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1970.jpg: 480x640 5 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1972.jpg: 480x640 6 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1973.jpg: 480x640 3 LegoBricks, 112.7ms
    Speed: 5.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1974.jpg: 480x640 4 LegoBricks, 112.7ms
    Speed: 6.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1975.jpg: 480x640 7 LegoBricks, 101.7ms
    Speed: 5.0ms preprocess, 101.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1976.jpg: 480x640 6 LegoBricks, 107.7ms
    Speed: 6.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1977.jpg: 480x640 5 LegoBricks, 128.3ms
    Speed: 6.0ms preprocess, 128.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1978.jpg: 480x640 3 LegoBricks, 486.9ms
    Speed: 6.0ms preprocess, 486.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1979.jpg: 480x640 7 LegoBricks, 175.5ms
    Speed: 8.0ms preprocess, 175.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_198.jpg: 480x640 5 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1980.jpg: 480x640 7 LegoBricks, 169.5ms
    Speed: 9.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_1981.jpg: 480x640 5 LegoBricks, 178.5ms
    Speed: 6.0ms preprocess, 178.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_20.jpg: 640x480 5 LegoBricks, 199.9ms
    Speed: 5.0ms preprocess, 199.9ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_200.jpg: 480x640 2 LegoBricks, 362.9ms
    Speed: 9.0ms preprocess, 362.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_201.jpg: 480x640 1 LegoBrick, 234.5ms
    Speed: 7.0ms preprocess, 234.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_203.jpg: 480x640 1 LegoBrick, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_204.jpg: 480x640 2 LegoBricks, 136.0ms
    Speed: 6.0ms preprocess, 136.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_205.jpg: 480x640 5 LegoBricks, 125.2ms
    Speed: 6.0ms preprocess, 125.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_206.jpg: 480x640 5 LegoBricks, 119.0ms
    Speed: 6.0ms preprocess, 119.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_207.jpg: 480x640 5 LegoBricks, 103.7ms
    Speed: 5.0ms preprocess, 103.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_208.jpg: 480x640 5 LegoBricks, 101.2ms
    Speed: 4.2ms preprocess, 101.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_209.jpg: 480x640 4 LegoBricks, 102.7ms
    Speed: 4.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_21.jpg: 640x480 8 LegoBricks, 143.8ms
    Speed: 5.0ms preprocess, 143.8ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_210.jpg: 480x640 1 LegoBrick, 116.8ms
    Speed: 6.0ms preprocess, 116.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_211.jpg: 480x640 3 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_212.jpg: 480x640 3 LegoBricks, 233.1ms
    Speed: 5.4ms preprocess, 233.1ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_213.jpg: 480x640 3 LegoBricks, 198.7ms
    Speed: 5.0ms preprocess, 198.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_219.jpg: 480x640 1 LegoBrick, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_221.jpg: 480x640 2 LegoBricks, 114.1ms
    Speed: 6.0ms preprocess, 114.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_223.jpg: 480x640 9 LegoBricks, 123.8ms
    Speed: 5.0ms preprocess, 123.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_224.jpg: 480x640 5 LegoBricks, 126.0ms
    Speed: 6.0ms preprocess, 126.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_225.jpg: 480x640 4 LegoBricks, 206.6ms
    Speed: 5.0ms preprocess, 206.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_226.jpg: 480x640 10 LegoBricks, 178.0ms
    Speed: 5.0ms preprocess, 178.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_228.jpg: 480x640 3 LegoBricks, 227.5ms
    Speed: 5.0ms preprocess, 227.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_23.jpg: 640x480 9 LegoBricks, 167.4ms
    Speed: 6.0ms preprocess, 167.4ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_230.jpg: 480x640 1 LegoBrick, 139.0ms
    Speed: 6.0ms preprocess, 139.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_231.jpg: 480x640 2 LegoBricks, 132.6ms
    Speed: 7.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_232.jpg: 480x640 3 LegoBricks, 159.1ms
    Speed: 11.0ms preprocess, 159.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_233.jpg: 480x640 3 LegoBricks, 129.1ms
    Speed: 5.0ms preprocess, 129.1ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_235.jpg: 480x640 3 LegoBricks, 127.1ms
    Speed: 5.0ms preprocess, 127.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_237.jpg: 480x640 5 LegoBricks, 136.9ms
    Speed: 5.0ms preprocess, 136.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_238.jpg: 480x640 8 LegoBricks, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_239.jpg: 480x640 7 LegoBricks, 164.0ms
    Speed: 6.0ms preprocess, 164.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_240.jpg: 480x640 3 LegoBricks, 143.5ms
    Speed: 5.0ms preprocess, 143.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_242.jpg: 480x640 4 LegoBricks, 243.3ms
    Speed: 6.0ms preprocess, 243.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_243.jpg: 480x640 4 LegoBricks, 266.7ms
    Speed: 32.9ms preprocess, 266.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_244.jpg: 480x640 4 LegoBricks, 118.3ms
    Speed: 5.0ms preprocess, 118.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_245.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_246.jpg: 480x640 4 LegoBricks, 115.7ms
    Speed: 4.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_247.jpg: 480x640 3 LegoBricks, 128.9ms
    Speed: 5.0ms preprocess, 128.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_248.jpg: 480x640 3 LegoBricks, 123.2ms
    Speed: 5.0ms preprocess, 123.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_249.jpg: 480x640 1 LegoBrick, 134.9ms
    Speed: 5.8ms preprocess, 134.9ms inference, 1.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_25.jpg: 640x480 6 LegoBricks, 255.5ms
    Speed: 6.0ms preprocess, 255.5ms inference, 4.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_250.jpg: 480x640 2 LegoBricks, 142.6ms
    Speed: 8.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_251.jpg: 480x640 3 LegoBricks, 138.0ms
    Speed: 5.0ms preprocess, 138.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_252.jpg: 480x640 4 LegoBricks, 146.6ms
    Speed: 8.9ms preprocess, 146.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_253.jpg: 480x640 1 LegoBrick, 181.9ms
    Speed: 46.9ms preprocess, 181.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_254.jpg: 480x640 1 LegoBrick, 153.3ms
    Speed: 7.0ms preprocess, 153.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_255.jpg: 480x640 2 LegoBricks, 150.6ms
    Speed: 6.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_256.jpg: 480x640 4 LegoBricks, 178.5ms
    Speed: 7.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_257.jpg: 480x640 5 LegoBricks, 177.7ms
    Speed: 6.0ms preprocess, 177.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_258.jpg: 480x640 3 LegoBricks, 194.5ms
    Speed: 6.0ms preprocess, 194.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_259.jpg: 480x640 3 LegoBricks, 306.2ms
    Speed: 31.9ms preprocess, 306.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_26.jpg: 640x480 5 LegoBricks, 224.6ms
    Speed: 27.9ms preprocess, 224.6ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_260.jpg: 480x640 4 LegoBricks, 168.5ms
    Speed: 5.8ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_261.jpg: 480x640 2 LegoBricks, 164.6ms
    Speed: 45.9ms preprocess, 164.6ms inference, 2.9ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_262.jpg: 480x640 3 LegoBricks, 168.3ms
    Speed: 5.0ms preprocess, 168.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_263.jpg: 480x640 3 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_264.jpg: 480x640 2 LegoBricks, 158.2ms
    Speed: 5.0ms preprocess, 158.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_265.jpg: 480x640 2 LegoBricks, 156.6ms
    Speed: 5.0ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_268.jpg: 480x640 2 LegoBricks, 151.3ms
    Speed: 5.0ms preprocess, 151.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_269.jpg: 480x640 3 LegoBricks, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_270.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_271.jpg: 480x640 2 LegoBricks, 192.9ms
    Speed: 6.0ms preprocess, 192.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_272.jpg: 480x640 4 LegoBricks, 140.5ms
    Speed: 6.0ms preprocess, 140.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_274.jpg: 480x640 3 LegoBricks, 140.2ms
    Speed: 6.0ms preprocess, 140.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_275.jpg: 480x640 4 LegoBricks, 164.3ms
    Speed: 5.0ms preprocess, 164.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_276.jpg: 480x640 6 LegoBricks, 196.0ms
    Speed: 8.0ms preprocess, 196.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_277.jpg: 480x640 3 LegoBricks, 179.5ms
    Speed: 6.0ms preprocess, 179.5ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_278.jpg: 480x640 2 LegoBricks, 253.7ms
    Speed: 6.0ms preprocess, 253.7ms inference, 31.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_28.jpg: 480x640 3 LegoBricks, 248.3ms
    Speed: 7.0ms preprocess, 248.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_280.jpg: 480x640 1 LegoBrick, 170.5ms
    Speed: 33.9ms preprocess, 170.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_281.jpg: 480x640 2 LegoBricks, 145.6ms
    Speed: 6.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_282.jpg: 480x640 2 LegoBricks, 222.4ms
    Speed: 6.0ms preprocess, 222.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_285.jpg: 480x640 2 LegoBricks, 158.6ms
    Speed: 5.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_286.jpg: 480x640 4 LegoBricks, 140.6ms
    Speed: 5.0ms preprocess, 140.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_287.jpg: 480x640 2 LegoBricks, 127.6ms
    Speed: 5.0ms preprocess, 127.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_288.jpg: 480x640 1 LegoBrick, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_289.jpg: 480x640 1 LegoBrick, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_29.jpg: 480x640 6 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_290.jpg: 480x640 3 LegoBricks, 137.0ms
    Speed: 5.0ms preprocess, 137.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_291.jpg: 480x640 1 LegoBrick, 114.7ms
    Speed: 6.0ms preprocess, 114.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_292.jpg: 480x640 1 LegoBrick, 124.7ms
    Speed: 8.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_293.jpg: 480x640 3 LegoBricks, 121.7ms
    Speed: 6.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_294.jpg: 480x640 5 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_296.jpg: 480x640 3 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_297.jpg: 480x640 2 LegoBricks, 125.5ms
    Speed: 24.9ms preprocess, 125.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_298.jpg: 480x640 1 LegoBrick, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_299.jpg: 480x640 1 LegoBrick, 129.7ms
    Speed: 5.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_3.jpg: 640x480 5 LegoBricks, 131.5ms
    Speed: 5.0ms preprocess, 131.5ms inference, 3.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_30.jpg: 480x640 5 LegoBricks, 167.6ms
    Speed: 7.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_300.jpg: 480x640 2 LegoBricks, 483.0ms
    Speed: 6.0ms preprocess, 483.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_302.jpg: 480x640 3 LegoBricks, 177.9ms
    Speed: 8.4ms preprocess, 177.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_303.jpg: 480x640 6 LegoBricks, 163.5ms
    Speed: 6.0ms preprocess, 163.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_305.jpg: 480x640 4 LegoBricks, 179.1ms
    Speed: 5.0ms preprocess, 179.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_307.jpg: 480x640 4 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_308.jpg: 480x640 4 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_309.jpg: 480x640 5 LegoBricks, 128.3ms
    Speed: 5.0ms preprocess, 128.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_31.jpg: 480x640 4 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_310.jpg: 480x640 3 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_311.jpg: 480x640 3 LegoBricks, 126.1ms
    Speed: 5.0ms preprocess, 126.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_312.jpg: 480x640 5 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_313.jpg: 480x640 5 LegoBricks, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_315.jpg: 480x640 2 LegoBricks, 171.5ms
    Speed: 5.0ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_316.jpg: 480x640 5 LegoBricks, 115.6ms
    Speed: 5.0ms preprocess, 115.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_318.jpg: 480x640 5 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_319.jpg: 480x640 2 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_32.jpg: 480x640 2 LegoBricks, 112.7ms
    Speed: 6.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_320.jpg: 480x640 1 LegoBrick, 127.7ms
    Speed: 5.7ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_321.jpg: 480x640 4 LegoBricks, 110.4ms
    Speed: 5.0ms preprocess, 110.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_322.jpg: 480x640 2 LegoBricks, 239.3ms
    Speed: 6.0ms preprocess, 239.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_323.jpg: 480x640 3 LegoBricks, 187.5ms
    Speed: 7.0ms preprocess, 187.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_325.jpg: 480x640 2 LegoBricks, 212.4ms
    Speed: 5.0ms preprocess, 212.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_326.jpg: 480x640 3 LegoBricks, 151.6ms
    Speed: 7.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_327.jpg: 480x640 5 LegoBricks, 170.5ms
    Speed: 6.0ms preprocess, 170.5ms inference, 2.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_328.jpg: 480x640 1 LegoBrick, 162.6ms
    Speed: 7.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_329.jpg: 480x640 1 LegoBrick, 161.6ms
    Speed: 6.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_33.jpg: 480x640 3 LegoBricks, 186.8ms
    Speed: 48.9ms preprocess, 186.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_330.jpg: 480x640 4 LegoBricks, 149.0ms
    Speed: 5.0ms preprocess, 149.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_331.jpg: 480x640 5 LegoBricks, 175.5ms
    Speed: 5.0ms preprocess, 175.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_332.jpg: 480x640 2 LegoBricks, 155.4ms
    Speed: 6.0ms preprocess, 155.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_333.jpg: 480x640 5 LegoBricks, 153.0ms
    Speed: 6.0ms preprocess, 153.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_334.jpg: 480x640 4 LegoBricks, 116.3ms
    Speed: 6.0ms preprocess, 116.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_337.jpg: 480x640 4 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_34.jpg: 480x640 4 LegoBricks, 126.2ms
    Speed: 5.0ms preprocess, 126.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_340.jpg: 480x640 2 LegoBricks, 347.2ms
    Speed: 4.0ms preprocess, 347.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_341.jpg: 480x640 2 LegoBricks, 233.4ms
    Speed: 21.9ms preprocess, 233.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_344.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 3.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_345.jpg: 480x640 2 LegoBricks, 137.8ms
    Speed: 5.0ms preprocess, 137.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_346.jpg: 480x640 4 LegoBricks, 171.5ms
    Speed: 6.0ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_347.jpg: 480x640 4 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_348.jpg: 480x640 7 LegoBricks, 125.5ms
    Speed: 3.0ms preprocess, 125.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_35.jpg: 480x640 4 LegoBricks, 137.7ms
    Speed: 5.0ms preprocess, 137.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_350.jpg: 480x640 4 LegoBricks, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_351.jpg: 480x640 6 LegoBricks, 165.1ms
    Speed: 5.0ms preprocess, 165.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_353.jpg: 480x640 1 LegoBrick, 159.6ms
    Speed: 6.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_355.jpg: 480x640 1 LegoBrick, 145.0ms
    Speed: 6.0ms preprocess, 145.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_356.jpg: 480x640 3 LegoBricks, 161.6ms
    Speed: 6.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_357.jpg: 480x640 5 LegoBricks, 160.6ms
    Speed: 7.0ms preprocess, 160.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_358.jpg: 480x640 2 LegoBricks, 163.6ms
    Speed: 6.0ms preprocess, 163.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_359.jpg: 480x640 4 LegoBricks, 154.6ms
    Speed: 6.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_36.jpg: 480x640 2 LegoBricks, 132.2ms
    Speed: 5.0ms preprocess, 132.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_361.jpg: 480x640 5 LegoBricks, 134.0ms
    Speed: 5.0ms preprocess, 134.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_364.jpg: 480x640 1 LegoBrick, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_365.jpg: 480x640 2 LegoBricks, 223.4ms
    Speed: 5.0ms preprocess, 223.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_366.jpg: 480x640 3 LegoBricks, 228.4ms
    Speed: 12.0ms preprocess, 228.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_367.jpg: 480x640 4 LegoBricks, 172.8ms
    Speed: 31.9ms preprocess, 172.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_37.jpg: 480x640 1 LegoBrick, 160.6ms
    Speed: 6.0ms preprocess, 160.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_371.jpg: 480x640 2 LegoBricks, 156.6ms
    Speed: 7.0ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_372.jpg: 480x640 1 LegoBrick, 126.5ms
    Speed: 6.0ms preprocess, 126.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_373.jpg: 480x640 3 LegoBricks, 126.4ms
    Speed: 5.0ms preprocess, 126.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_374.jpg: 480x640 2 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_375.jpg: 480x640 3 LegoBricks, 139.6ms
    Speed: 6.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_376.jpg: 480x640 3 LegoBricks, 125.1ms
    Speed: 6.0ms preprocess, 125.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_377.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_38.jpg: 480x640 5 LegoBricks, 147.9ms
    Speed: 6.0ms preprocess, 147.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_380.jpg: 480x640 1 LegoBrick, 153.4ms
    Speed: 6.0ms preprocess, 153.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_381.jpg: 480x640 3 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_382.jpg: 480x640 7 LegoBricks, 194.5ms
    Speed: 5.0ms preprocess, 194.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_383.jpg: 480x640 5 LegoBricks, 159.7ms
    Speed: 6.0ms preprocess, 159.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_384.jpg: 480x640 6 LegoBricks, 214.4ms
    Speed: 5.0ms preprocess, 214.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_385.jpg: 480x640 7 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_386.jpg: 480x640 6 LegoBricks, 241.9ms
    Speed: 5.0ms preprocess, 241.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_387.jpg: 480x640 3 LegoBricks, 252.0ms
    Speed: 6.0ms preprocess, 252.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_388.jpg: 480x640 3 LegoBricks, 166.9ms
    Speed: 6.0ms preprocess, 166.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_389.jpg: 480x640 4 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_39.jpg: 480x640 2 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_390.jpg: 480x640 3 LegoBricks, 104.7ms
    Speed: 5.0ms preprocess, 104.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_391.jpg: 480x640 3 LegoBricks, 97.7ms
    Speed: 5.0ms preprocess, 97.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_392.jpg: 480x640 4 LegoBricks, 102.7ms
    Speed: 6.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_393.jpg: 480x640 7 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_395.jpg: 480x640 6 LegoBricks, 108.0ms
    Speed: 5.0ms preprocess, 108.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_397.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_398.jpg: 480x640 4 LegoBricks, 109.0ms
    Speed: 6.0ms preprocess, 109.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_399.jpg: 480x640 4 LegoBricks, 115.1ms
    Speed: 6.0ms preprocess, 115.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_40.jpg: 480x640 1 LegoBrick, 130.1ms
    Speed: 11.0ms preprocess, 130.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_400.jpg: 480x640 3 LegoBricks, 165.6ms
    Speed: 5.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_401.jpg: 480x640 6 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_402.jpg: 480x640 5 LegoBricks, 149.0ms
    Speed: 5.0ms preprocess, 149.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_403.jpg: 480x640 3 LegoBricks, 137.4ms
    Speed: 6.0ms preprocess, 137.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_405.jpg: 480x640 5 LegoBricks, 266.5ms
    Speed: 8.0ms preprocess, 266.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_406.jpg: 480x640 2 LegoBricks, 289.1ms
    Speed: 34.9ms preprocess, 289.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_407.jpg: 480x640 1 LegoBrick, 150.6ms
    Speed: 6.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_408.jpg: 480x640 1 LegoBrick, 164.6ms
    Speed: 7.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_409.jpg: 480x640 2 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_41.jpg: 480x640 7 LegoBricks, 154.6ms
    Speed: 5.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_410.jpg: 480x640 4 LegoBricks, 144.8ms
    Speed: 5.0ms preprocess, 144.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_411.jpg: 480x640 4 LegoBricks, 158.6ms
    Speed: 6.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_412.jpg: 480x640 3 LegoBricks, 126.7ms
    Speed: 6.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_413.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_414.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 11.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_415.jpg: 480x640 3 LegoBricks, 112.7ms
    Speed: 8.0ms preprocess, 112.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_417.jpg: 480x640 1 LegoBrick, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_418.jpg: 480x640 1 LegoBrick, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_419.jpg: 480x640 2 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_42.jpg: 480x640 5 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_421.jpg: 480x640 3 LegoBricks, 112.2ms
    Speed: 5.0ms preprocess, 112.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_422.jpg: 480x640 3 LegoBricks, 280.3ms
    Speed: 8.0ms preprocess, 280.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_423.jpg: 480x640 3 LegoBricks, 158.8ms
    Speed: 7.0ms preprocess, 158.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_424.jpg: 480x640 3 LegoBricks, 124.0ms
    Speed: 6.0ms preprocess, 124.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_425.jpg: 480x640 2 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_426.jpg: 480x640 1 LegoBrick, 122.7ms
    Speed: 5.6ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_427.jpg: 480x640 2 LegoBricks, 155.8ms
    Speed: 10.7ms preprocess, 155.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_428.jpg: 480x640 1 LegoBrick, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_429.jpg: 480x640 3 LegoBricks, 166.6ms
    Speed: 56.8ms preprocess, 166.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_43.jpg: 480x640 4 LegoBricks, 143.6ms
    Speed: 6.0ms preprocess, 143.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_430.jpg: 480x640 7 LegoBricks, 199.4ms
    Speed: 7.0ms preprocess, 199.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_431.jpg: 480x640 5 LegoBricks, 147.6ms
    Speed: 7.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_432.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_433.jpg: 480x640 3 LegoBricks, 140.0ms
    Speed: 6.0ms preprocess, 140.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_434.jpg: 480x640 1 LegoBrick, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_435.jpg: 480x640 5 LegoBricks, 128.3ms
    Speed: 5.0ms preprocess, 128.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_436.jpg: 480x640 7 LegoBricks, 178.5ms
    Speed: 7.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_437.jpg: 480x640 7 LegoBricks, 295.5ms
    Speed: 7.0ms preprocess, 295.5ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_438.jpg: 480x640 2 LegoBricks, 193.7ms
    Speed: 6.0ms preprocess, 193.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_439.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_44.jpg: 480x640 3 LegoBricks, 126.7ms
    Speed: 4.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_440.jpg: 480x640 5 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_441.jpg: 480x640 5 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_442.jpg: 480x640 8 LegoBricks, 106.6ms
    Speed: 5.0ms preprocess, 106.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_443.jpg: 480x640 4 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_444.jpg: 480x640 4 LegoBricks, 208.8ms
    Speed: 5.0ms preprocess, 208.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_445.jpg: 480x640 4 LegoBricks, 141.6ms
    Speed: 6.0ms preprocess, 141.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_446.jpg: 480x640 2 LegoBricks, 119.5ms
    Speed: 6.0ms preprocess, 119.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_447.jpg: 480x640 2 LegoBricks, 126.6ms
    Speed: 5.0ms preprocess, 126.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_448.jpg: 480x640 2 LegoBricks, 129.7ms
    Speed: 5.0ms preprocess, 129.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_449.jpg: 480x640 1 LegoBrick, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_450.jpg: 480x640 1 LegoBrick, 126.6ms
    Speed: 6.0ms preprocess, 126.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_451.jpg: 480x640 1 LegoBrick, 158.1ms
    Speed: 6.0ms preprocess, 158.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_452.jpg: 480x640 5 LegoBricks, 176.3ms
    Speed: 8.0ms preprocess, 176.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_453.jpg: 480x640 9 LegoBricks, 320.5ms
    Speed: 21.9ms preprocess, 320.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_454.jpg: 480x640 10 LegoBricks, 185.4ms
    Speed: 6.0ms preprocess, 185.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_455.jpg: 480x640 3 LegoBricks, 181.1ms
    Speed: 6.0ms preprocess, 181.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_456.jpg: 480x640 3 LegoBricks, 129.7ms
    Speed: 6.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_457.jpg: 480x640 4 LegoBricks, 201.4ms
    Speed: 7.0ms preprocess, 201.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_458.jpg: 480x640 1 LegoBrick, 132.6ms
    Speed: 5.9ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_459.jpg: 480x640 2 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_46.jpg: 480x640 2 LegoBricks, 124.7ms
    Speed: 4.0ms preprocess, 124.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_460.jpg: 480x640 1 LegoBrick, 129.6ms
    Speed: 5.0ms preprocess, 129.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_461.jpg: 480x640 5 LegoBricks, 119.7ms
    Speed: 5.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_462.jpg: 480x640 7 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_463.jpg: 480x640 7 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_464.jpg: 480x640 8 LegoBricks, 115.9ms
    Speed: 13.0ms preprocess, 115.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_465.jpg: 480x640 7 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_466.jpg: 480x640 5 LegoBricks, 145.6ms
    Speed: 7.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_467.jpg: 480x640 6 LegoBricks, 115.7ms
    Speed: 6.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_468.jpg: 480x640 2 LegoBricks, 178.5ms
    Speed: 6.0ms preprocess, 178.5ms inference, 2.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_469.jpg: 480x640 3 LegoBricks, 231.4ms
    Speed: 32.9ms preprocess, 231.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_47.jpg: 480x640 4 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_470.jpg: 480x640 3 LegoBricks, 176.5ms
    Speed: 24.9ms preprocess, 176.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_471.jpg: 480x640 5 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_472.jpg: 480x640 3 LegoBricks, 164.7ms
    Speed: 6.0ms preprocess, 164.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_473.jpg: 480x640 5 LegoBricks, 180.3ms
    Speed: 7.0ms preprocess, 180.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_474.jpg: 480x640 4 LegoBricks, 150.6ms
    Speed: 6.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_475.jpg: 480x640 1 LegoBrick, 145.6ms
    Speed: 6.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_476.jpg: 480x640 1 LegoBrick, 141.8ms
    Speed: 6.0ms preprocess, 141.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_477.jpg: 480x640 2 LegoBricks, 125.8ms
    Speed: 7.0ms preprocess, 125.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_478.jpg: 480x640 2 LegoBricks, 168.5ms
    Speed: 6.0ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_479.jpg: 480x640 1 LegoBrick, 161.6ms
    Speed: 6.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_48.jpg: 480x640 1 LegoBrick, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_480.jpg: 480x640 1 LegoBrick, 128.5ms
    Speed: 5.0ms preprocess, 128.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_481.jpg: 480x640 4 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_482.jpg: 480x640 4 LegoBricks, 376.4ms
    Speed: 5.0ms preprocess, 376.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_483.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 7.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_484.jpg: 480x640 2 LegoBricks, 128.9ms
    Speed: 5.0ms preprocess, 128.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_485.jpg: 480x640 2 LegoBricks, 113.6ms
    Speed: 5.0ms preprocess, 113.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_486.jpg: 480x640 1 LegoBrick, 125.6ms
    Speed: 6.0ms preprocess, 125.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_487.jpg: 480x640 2 LegoBricks, 154.6ms
    Speed: 6.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_488.jpg: 480x640 4 LegoBricks, 136.6ms
    Speed: 6.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_489.jpg: 480x640 5 LegoBricks, 114.5ms
    Speed: 6.0ms preprocess, 114.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_49.jpg: 480x640 5 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_490.jpg: 480x640 5 LegoBricks, 119.5ms
    Speed: 5.0ms preprocess, 119.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_491.jpg: 480x640 6 LegoBricks, 134.6ms
    Speed: 35.9ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_492.jpg: 480x640 1 LegoBrick, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_493.jpg: 480x640 1 LegoBrick, 138.3ms
    Speed: 5.0ms preprocess, 138.3ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_494.jpg: 480x640 2 LegoBricks, 170.4ms
    Speed: 7.0ms preprocess, 170.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_495.jpg: 480x640 3 LegoBricks, 175.8ms
    Speed: 7.0ms preprocess, 175.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_496.jpg: 480x640 4 LegoBricks, 167.6ms
    Speed: 6.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_497.jpg: 480x640 4 LegoBricks, 161.6ms
    Speed: 6.0ms preprocess, 161.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_498.jpg: 480x640 3 LegoBricks, 139.3ms
    Speed: 5.0ms preprocess, 139.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_499.jpg: 480x640 3 LegoBricks, 130.2ms
    Speed: 5.0ms preprocess, 130.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_50.jpg: 480x640 5 LegoBricks, 148.6ms
    Speed: 5.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_500.jpg: 480x640 4 LegoBricks, 471.7ms
    Speed: 15.0ms preprocess, 471.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_501.jpg: 480x640 4 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_502.jpg: 480x640 1 LegoBrick, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_503.jpg: 480x640 5 LegoBricks, 220.4ms
    Speed: 5.0ms preprocess, 220.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_504.jpg: 480x640 5 LegoBricks, 129.7ms
    Speed: 5.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_505.jpg: 480x640 6 LegoBricks, 123.4ms
    Speed: 6.0ms preprocess, 123.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_506.jpg: 480x640 2 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_507.jpg: 480x640 3 LegoBricks, 113.7ms
    Speed: 6.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_508.jpg: 480x640 3 LegoBricks, 110.7ms
    Speed: 6.0ms preprocess, 110.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_509.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 6.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_510.jpg: 480x640 3 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_511.jpg: 480x640 2 LegoBricks, 108.1ms
    Speed: 5.0ms preprocess, 108.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_512.jpg: 480x640 3 LegoBricks, 134.8ms
    Speed: 8.0ms preprocess, 134.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_513.jpg: 480x640 3 LegoBricks, 139.6ms
    Speed: 6.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_514.jpg: 480x640 3 LegoBricks, 129.5ms
    Speed: 5.0ms preprocess, 129.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_515.jpg: 480x640 3 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_516.jpg: 480x640 3 LegoBricks, 226.0ms
    Speed: 5.0ms preprocess, 226.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_517.jpg: 480x640 3 LegoBricks, 383.2ms
    Speed: 17.6ms preprocess, 383.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_518.jpg: 480x640 5 LegoBricks, 146.1ms
    Speed: 6.0ms preprocess, 146.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_519.jpg: 480x640 1 LegoBrick, 145.8ms
    Speed: 5.0ms preprocess, 145.8ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_52.jpg: 480x640 4 LegoBricks, 167.6ms
    Speed: 6.0ms preprocess, 167.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_520.jpg: 480x640 4 LegoBricks, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_521.jpg: 480x640 2 LegoBricks, 155.8ms
    Speed: 6.0ms preprocess, 155.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_522.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_523.jpg: 480x640 4 LegoBricks, 116.7ms
    Speed: 6.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_524.jpg: 480x640 3 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_525.jpg: 480x640 5 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_526.jpg: 480x640 4 LegoBricks, 130.7ms
    Speed: 6.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_527.jpg: 480x640 4 LegoBricks, 133.6ms
    Speed: 4.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_528.jpg: 480x640 3 LegoBricks, 135.9ms
    Speed: 5.0ms preprocess, 135.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_529.jpg: 480x640 3 LegoBricks, 197.5ms
    Speed: 5.0ms preprocess, 197.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_53.jpg: 480x640 4 LegoBricks, 129.0ms
    Speed: 5.0ms preprocess, 129.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_530.jpg: 480x640 3 LegoBricks, 153.6ms
    Speed: 5.0ms preprocess, 153.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_531.jpg: 480x640 1 LegoBrick, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_532.jpg: 480x640 3 LegoBricks, 112.7ms
    Speed: 5.0ms preprocess, 112.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_533.jpg: 480x640 5 LegoBricks, 158.6ms
    Speed: 3.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_534.jpg: 480x640 7 LegoBricks, 265.7ms
    Speed: 5.0ms preprocess, 265.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_535.jpg: 480x640 6 LegoBricks, 143.6ms
    Speed: 5.0ms preprocess, 143.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_536.jpg: 480x640 2 LegoBricks, 180.2ms
    Speed: 5.0ms preprocess, 180.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_537.jpg: 480x640 2 LegoBricks, 158.0ms
    Speed: 7.0ms preprocess, 158.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_538.jpg: 480x640 2 LegoBricks, 173.5ms
    Speed: 6.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_539.jpg: 480x640 3 LegoBricks, 141.6ms
    Speed: 6.0ms preprocess, 141.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_54.jpg: 480x640 3 LegoBricks, 162.6ms
    Speed: 7.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_540.jpg: 480x640 7 LegoBricks, 216.9ms
    Speed: 5.0ms preprocess, 216.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_541.jpg: 480x640 8 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_542.jpg: 480x640 7 LegoBricks, 146.2ms
    Speed: 6.0ms preprocess, 146.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_543.jpg: 480x640 2 LegoBricks, 135.9ms
    Speed: 5.0ms preprocess, 135.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_544.jpg: 480x640 1 LegoBrick, 136.6ms
    Speed: 6.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_545.jpg: 480x640 2 LegoBricks, 154.1ms
    Speed: 5.0ms preprocess, 154.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_546.jpg: 480x640 2 LegoBricks, 141.7ms
    Speed: 5.0ms preprocess, 141.7ms inference, 1.9ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_547.jpg: 480x640 6 LegoBricks, 142.5ms
    Speed: 5.0ms preprocess, 142.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_548.jpg: 480x640 1 LegoBrick, 215.4ms
    Speed: 6.0ms preprocess, 215.4ms inference, 25.9ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_549.jpg: 480x640 2 LegoBricks, 381.4ms
    Speed: 16.0ms preprocess, 381.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_55.jpg: 480x640 3 LegoBricks, 114.0ms
    Speed: 5.0ms preprocess, 114.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_550.jpg: 480x640 3 LegoBricks, 112.8ms
    Speed: 5.0ms preprocess, 112.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_551.jpg: 480x640 3 LegoBricks, 129.7ms
    Speed: 7.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_552.jpg: 480x640 3 LegoBricks, 148.6ms
    Speed: 42.9ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_553.jpg: 480x640 3 LegoBricks, 129.7ms
    Speed: 6.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_554.jpg: 480x640 2 LegoBricks, 127.3ms
    Speed: 5.0ms preprocess, 127.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_555.jpg: 480x640 3 LegoBricks, 114.5ms
    Speed: 6.0ms preprocess, 114.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_556.jpg: 480x640 5 LegoBricks, 109.0ms
    Speed: 6.0ms preprocess, 109.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_557.jpg: 480x640 3 LegoBricks, 151.5ms
    Speed: 5.0ms preprocess, 151.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_558.jpg: 480x640 5 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_559.jpg: 480x640 9 LegoBricks, 128.6ms
    Speed: 5.0ms preprocess, 128.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_56.jpg: 480x640 2 LegoBricks, 135.8ms
    Speed: 5.0ms preprocess, 135.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_560.jpg: 480x640 5 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_561.jpg: 480x640 2 LegoBricks, 170.5ms
    Speed: 71.8ms preprocess, 170.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_562.jpg: 480x640 5 LegoBricks, 178.5ms
    Speed: 6.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_563.jpg: 480x640 3 LegoBricks, 141.3ms
    Speed: 5.3ms preprocess, 141.3ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_564.jpg: 480x640 1 LegoBrick, 240.4ms
    Speed: 6.0ms preprocess, 240.4ms inference, 11.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_565.jpg: 480x640 2 LegoBricks, 216.4ms
    Speed: 5.0ms preprocess, 216.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_567.jpg: 480x640 4 LegoBricks, 145.6ms
    Speed: 5.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_569.jpg: 480x640 3 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_57.jpg: 480x640 2 LegoBricks, 148.8ms
    Speed: 6.0ms preprocess, 148.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_570.jpg: 480x640 8 LegoBricks, 191.5ms
    Speed: 7.0ms preprocess, 191.5ms inference, 2.2ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_571.jpg: 480x640 8 LegoBricks, 162.6ms
    Speed: 5.0ms preprocess, 162.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_572.jpg: 480x640 3 LegoBricks, 158.6ms
    Speed: 6.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_573.jpg: 480x640 5 LegoBricks, 117.4ms
    Speed: 6.1ms preprocess, 117.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_574.jpg: 480x640 7 LegoBricks, 110.5ms
    Speed: 5.0ms preprocess, 110.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_575.jpg: 480x640 6 LegoBricks, 126.9ms
    Speed: 62.8ms preprocess, 126.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_576.jpg: 480x640 12 LegoBricks, 111.3ms
    Speed: 5.0ms preprocess, 111.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_577.jpg: 480x640 7 LegoBricks, 109.6ms
    Speed: 5.0ms preprocess, 109.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_578.jpg: 480x640 9 LegoBricks, 125.7ms
    Speed: 5.9ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_579.jpg: 480x640 13 LegoBricks, 110.9ms
    Speed: 6.0ms preprocess, 110.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_58.jpg: 480x640 2 LegoBricks, 157.0ms
    Speed: 4.0ms preprocess, 157.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_580.jpg: 480x640 6 LegoBricks, 150.6ms
    Speed: 6.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_581.jpg: 480x640 17 LegoBricks, 118.7ms
    Speed: 6.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_582.jpg: 480x640 3 LegoBricks, 171.2ms
    Speed: 7.0ms preprocess, 171.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_583.jpg: 480x640 8 LegoBricks, 338.6ms
    Speed: 9.0ms preprocess, 338.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_584.jpg: 480x640 7 LegoBricks, 172.1ms
    Speed: 42.9ms preprocess, 172.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_585.jpg: 480x640 7 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_586.jpg: 480x640 9 LegoBricks, 140.6ms
    Speed: 5.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_587.jpg: 480x640 8 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_588.jpg: 480x640 4 LegoBricks, 151.2ms
    Speed: 5.0ms preprocess, 151.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_589.jpg: 480x640 7 LegoBricks, 130.7ms
    Speed: 5.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_59.jpg: 480x640 1 LegoBrick, 122.7ms
    Speed: 6.0ms preprocess, 122.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_590.jpg: 480x640 8 LegoBricks, 147.9ms
    Speed: 16.0ms preprocess, 147.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_591.jpg: 480x640 8 LegoBricks, 136.5ms
    Speed: 5.0ms preprocess, 136.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_592.jpg: 480x640 9 LegoBricks, 127.7ms
    Speed: 6.0ms preprocess, 127.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_593.jpg: 480x640 10 LegoBricks, 126.9ms
    Speed: 6.0ms preprocess, 126.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_594.jpg: 480x640 9 LegoBricks, 125.5ms
    Speed: 7.0ms preprocess, 125.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_595.jpg: 480x640 7 LegoBricks, 136.4ms
    Speed: 3.0ms preprocess, 136.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_596.jpg: 480x640 7 LegoBricks, 129.4ms
    Speed: 7.0ms preprocess, 129.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_597.jpg: 480x640 6 LegoBricks, 301.7ms
    Speed: 6.0ms preprocess, 301.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_598.jpg: 480x640 10 LegoBricks, 187.5ms
    Speed: 6.0ms preprocess, 187.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_60.jpg: 480x640 3 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_600.jpg: 480x640 6 LegoBricks, 206.4ms
    Speed: 8.0ms preprocess, 206.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_601.jpg: 480x640 6 LegoBricks, 157.4ms
    Speed: 7.0ms preprocess, 157.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_602.jpg: 480x640 5 LegoBricks, 116.7ms
    Speed: 6.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_603.jpg: 480x640 6 LegoBricks, 132.0ms
    Speed: 6.0ms preprocess, 132.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_604.jpg: 480x640 4 LegoBricks, 178.5ms
    Speed: 5.0ms preprocess, 178.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_605.jpg: 480x640 6 LegoBricks, 219.4ms
    Speed: 8.0ms preprocess, 219.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_607.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_608.jpg: 480x640 5 LegoBricks, 152.6ms
    Speed: 29.8ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_609.jpg: 480x640 8 LegoBricks, 126.7ms
    Speed: 5.0ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_61.jpg: 480x640 3 LegoBricks, 121.7ms
    Speed: 4.1ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_610.jpg: 480x640 4 LegoBricks, 157.6ms
    Speed: 5.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_611.jpg: 480x640 5 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_612.jpg: 480x640 6 LegoBricks, 123.7ms
    Speed: 8.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_613.jpg: 480x640 5 LegoBricks, 374.6ms
    Speed: 6.0ms preprocess, 374.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_614.jpg: 480x640 6 LegoBricks, 245.3ms
    Speed: 19.0ms preprocess, 245.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_615.jpg: 480x640 7 LegoBricks, 153.6ms
    Speed: 8.0ms preprocess, 153.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_616.jpg: 480x640 7 LegoBricks, 118.7ms
    Speed: 7.0ms preprocess, 118.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_617.jpg: 480x640 4 LegoBricks, 128.7ms
    Speed: 5.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_618.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_619.jpg: 480x640 2 LegoBricks, 124.7ms
    Speed: 7.0ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_62.jpg: 480x640 2 LegoBricks, 183.3ms
    Speed: 5.0ms preprocess, 183.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_620.jpg: 480x640 3 LegoBricks, 149.6ms
    Speed: 5.0ms preprocess, 149.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_621.jpg: 480x640 7 LegoBricks, 194.5ms
    Speed: 6.0ms preprocess, 194.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_622.jpg: 480x640 7 LegoBricks, 169.5ms
    Speed: 7.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_623.jpg: 480x640 9 LegoBricks, 159.6ms
    Speed: 6.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_624.jpg: 480x640 3 LegoBricks, 144.6ms
    Speed: 6.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_625.jpg: 480x640 2 LegoBricks, 149.3ms
    Speed: 5.0ms preprocess, 149.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_626.jpg: 480x640 3 LegoBricks, 157.6ms
    Speed: 6.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_627.jpg: 480x640 3 LegoBricks, 362.5ms
    Speed: 7.0ms preprocess, 362.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_628.jpg: 480x640 5 LegoBricks, 241.8ms
    Speed: 8.0ms preprocess, 241.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_629.jpg: 480x640 6 LegoBricks, 157.5ms
    Speed: 5.0ms preprocess, 157.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_63.jpg: 480x640 2 LegoBricks, 139.5ms
    Speed: 5.0ms preprocess, 139.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_630.jpg: 480x640 4 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_631.jpg: 480x640 4 LegoBricks, 156.6ms
    Speed: 36.9ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_632.jpg: 480x640 4 LegoBricks, 136.1ms
    Speed: 4.0ms preprocess, 136.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_633.jpg: 480x640 5 LegoBricks, 123.9ms
    Speed: 6.0ms preprocess, 123.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_634.jpg: 480x640 8 LegoBricks, 142.9ms
    Speed: 6.0ms preprocess, 142.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_635.jpg: 480x640 6 LegoBricks, 128.7ms
    Speed: 5.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_636.jpg: 480x640 5 LegoBricks, 165.0ms
    Speed: 5.0ms preprocess, 165.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_637.jpg: 480x640 6 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_638.jpg: 480x640 8 LegoBricks, 125.4ms
    Speed: 6.0ms preprocess, 125.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_639.jpg: 480x640 8 LegoBricks, 130.6ms
    Speed: 3.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_64.jpg: 480x640 2 LegoBricks, 114.7ms
    Speed: 5.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_640.jpg: 480x640 8 LegoBricks, 161.1ms
    Speed: 6.0ms preprocess, 161.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_641.jpg: 480x640 11 LegoBricks, 330.3ms
    Speed: 7.9ms preprocess, 330.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_642.jpg: 480x640 5 LegoBricks, 147.6ms
    Speed: 7.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_643.jpg: 480x640 5 LegoBricks, 201.8ms
    Speed: 5.0ms preprocess, 201.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_644.jpg: 480x640 7 LegoBricks, 157.6ms
    Speed: 5.0ms preprocess, 157.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_645.jpg: 480x640 8 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_646.jpg: 480x640 7 LegoBricks, 124.2ms
    Speed: 5.0ms preprocess, 124.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_647.jpg: 480x640 4 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_648.jpg: 480x640 4 LegoBricks, 156.1ms
    Speed: 5.0ms preprocess, 156.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_649.jpg: 480x640 5 LegoBricks, 141.6ms
    Speed: 7.0ms preprocess, 141.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_65.jpg: 480x640 3 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_650.jpg: 480x640 6 LegoBricks, 129.0ms
    Speed: 5.0ms preprocess, 129.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_651.jpg: 480x640 6 LegoBricks, 167.8ms
    Speed: 4.0ms preprocess, 167.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_652.jpg: 480x640 5 LegoBricks, 228.3ms
    Speed: 6.0ms preprocess, 228.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_653.jpg: 480x640 3 LegoBricks, 167.9ms
    Speed: 6.0ms preprocess, 167.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_654.jpg: 480x640 6 LegoBricks, 169.6ms
    Speed: 5.0ms preprocess, 169.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_655.jpg: 480x640 9 LegoBricks, 350.2ms
    Speed: 5.0ms preprocess, 350.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_656.jpg: 480x640 5 LegoBricks, 189.4ms
    Speed: 5.0ms preprocess, 189.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_657.jpg: 480x640 3 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_658.jpg: 480x640 5 LegoBricks, 114.7ms
    Speed: 7.0ms preprocess, 114.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_659.jpg: 480x640 5 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_66.jpg: 480x640 2 LegoBricks, 156.6ms
    Speed: 5.5ms preprocess, 156.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_660.jpg: 480x640 5 LegoBricks, 154.6ms
    Speed: 6.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_661.jpg: 480x640 7 LegoBricks, 142.3ms
    Speed: 5.0ms preprocess, 142.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_662.jpg: 480x640 8 LegoBricks, 125.7ms
    Speed: 4.0ms preprocess, 125.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_663.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 3.0ms preprocess, 113.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_664.jpg: 480x640 3 LegoBricks, 121.7ms
    Speed: 68.8ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_665.jpg: 480x640 8 LegoBricks, 111.4ms
    Speed: 5.0ms preprocess, 111.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_666.jpg: 480x640 6 LegoBricks, 115.4ms
    Speed: 4.0ms preprocess, 115.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_668.jpg: 480x640 4 LegoBricks, 126.1ms
    Speed: 6.0ms preprocess, 126.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_669.jpg: 480x640 4 LegoBricks, 154.6ms
    Speed: 5.0ms preprocess, 154.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_67.jpg: 480x640 3 LegoBricks, 224.3ms
    Speed: 5.0ms preprocess, 224.3ms inference, 6.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_670.jpg: 480x640 3 LegoBricks, 267.8ms
    Speed: 5.0ms preprocess, 267.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_671.jpg: 480x640 4 LegoBricks, 147.6ms
    Speed: 7.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_672.jpg: 480x640 4 LegoBricks, 127.7ms
    Speed: 5.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_674.jpg: 480x640 4 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_675.jpg: 480x640 2 LegoBricks, 185.5ms
    Speed: 53.9ms preprocess, 185.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_676.jpg: 480x640 6 LegoBricks, 295.4ms
    Speed: 6.0ms preprocess, 295.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_677.jpg: 480x640 7 LegoBricks, 234.5ms
    Speed: 7.0ms preprocess, 234.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_678.jpg: 480x640 4 LegoBricks, 158.6ms
    Speed: 5.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_679.jpg: 480x640 4 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_68.jpg: 480x640 3 LegoBricks, 142.2ms
    Speed: 5.0ms preprocess, 142.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_680.jpg: 480x640 6 LegoBricks, 126.6ms
    Speed: 5.0ms preprocess, 126.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_681.jpg: 480x640 6 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_682.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 7.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_683.jpg: 480x640 5 LegoBricks, 130.0ms
    Speed: 5.0ms preprocess, 130.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_684.jpg: 480x640 9 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_685.jpg: 480x640 9 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_687.jpg: 480x640 9 LegoBricks, 124.7ms
    Speed: 6.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_688.jpg: 480x640 8 LegoBricks, 273.2ms
    Speed: 12.0ms preprocess, 273.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_689.jpg: 480x640 9 LegoBricks, 164.6ms
    Speed: 7.0ms preprocess, 164.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_69.jpg: 480x640 2 LegoBricks, 237.4ms
    Speed: 5.0ms preprocess, 237.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_690.jpg: 480x640 6 LegoBricks, 119.9ms
    Speed: 6.0ms preprocess, 119.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_691.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_692.jpg: 480x640 8 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_693.jpg: 480x640 8 LegoBricks, 141.1ms
    Speed: 6.0ms preprocess, 141.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_695.jpg: 480x640 11 LegoBricks, 162.6ms
    Speed: 5.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_696.jpg: 480x640 8 LegoBricks, 170.5ms
    Speed: 6.0ms preprocess, 170.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_697.jpg: 480x640 5 LegoBricks, 149.9ms
    Speed: 6.0ms preprocess, 149.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_698.jpg: 480x640 7 LegoBricks, 135.6ms
    Speed: 5.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_699.jpg: 480x640 8 LegoBricks, 128.7ms
    Speed: 6.0ms preprocess, 128.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_7.jpg: 640x480 7 LegoBricks, 220.4ms
    Speed: 6.0ms preprocess, 220.4ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_70.jpg: 480x640 3 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_700.jpg: 480x640 6 LegoBricks, 123.9ms
    Speed: 6.0ms preprocess, 123.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_701.jpg: 480x640 8 LegoBricks, 273.3ms
    Speed: 5.0ms preprocess, 273.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_702.jpg: 480x640 8 LegoBricks, 236.1ms
    Speed: 9.0ms preprocess, 236.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_703.jpg: 480x640 7 LegoBricks, 201.5ms
    Speed: 7.0ms preprocess, 201.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_704.jpg: 480x640 7 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_705.jpg: 480x640 5 LegoBricks, 137.4ms
    Speed: 6.0ms preprocess, 137.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_706.jpg: 480x640 6 LegoBricks, 122.3ms
    Speed: 5.0ms preprocess, 122.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_707.jpg: 480x640 2 LegoBricks, 192.5ms
    Speed: 6.0ms preprocess, 192.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_708.jpg: 480x640 9 LegoBricks, 142.3ms
    Speed: 6.0ms preprocess, 142.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_709.jpg: 480x640 10 LegoBricks, 130.6ms
    Speed: 5.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_71.jpg: 480x640 2 LegoBricks, 137.6ms
    Speed: 5.0ms preprocess, 137.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_710.jpg: 480x640 5 LegoBricks, 119.7ms
    Speed: 4.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_715.jpg: 480x640 6 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.6ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_716.jpg: 480x640 7 LegoBricks, 131.6ms
    Speed: 6.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_717.jpg: 480x640 6 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_718.jpg: 480x640 6 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_72.jpg: 480x640 3 LegoBricks, 127.6ms
    Speed: 6.0ms preprocess, 127.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_720.jpg: 480x640 9 LegoBricks, 345.5ms
    Speed: 86.8ms preprocess, 345.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_721.jpg: 480x640 7 LegoBricks, 185.5ms
    Speed: 9.0ms preprocess, 185.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_722.jpg: 480x640 7 LegoBricks, 182.5ms
    Speed: 5.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_724.jpg: 480x640 6 LegoBricks, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_725.jpg: 480x640 4 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_726.jpg: 480x640 8 LegoBricks, 165.5ms
    Speed: 5.0ms preprocess, 165.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_727.jpg: 480x640 8 LegoBricks, 152.6ms
    Speed: 7.0ms preprocess, 152.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_728.jpg: 480x640 8 LegoBricks, 134.0ms
    Speed: 6.0ms preprocess, 134.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_729.jpg: 480x640 8 LegoBricks, 118.7ms
    Speed: 6.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_73.jpg: 480x640 2 LegoBricks, 203.8ms
    Speed: 6.0ms preprocess, 203.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_733.jpg: 480x640 8 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_734.jpg: 480x640 8 LegoBricks, 112.7ms
    Speed: 5.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_735.jpg: 480x640 3 LegoBricks, 121.0ms
    Speed: 5.0ms preprocess, 121.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_736.jpg: 480x640 3 LegoBricks, 118.6ms
    Speed: 6.0ms preprocess, 118.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_737.jpg: 480x640 5 LegoBricks, 110.7ms
    Speed: 6.0ms preprocess, 110.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_738.jpg: 480x640 6 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_739.jpg: 480x640 10 LegoBricks, 206.0ms
    Speed: 6.0ms preprocess, 206.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_74.jpg: 480x640 3 LegoBricks, 233.4ms
    Speed: 4.0ms preprocess, 233.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_741.jpg: 480x640 3 LegoBricks, 259.8ms
    Speed: 6.0ms preprocess, 259.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_742.jpg: 480x640 8 LegoBricks, 380.1ms
    Speed: 6.0ms preprocess, 380.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_743.jpg: 480x640 4 LegoBricks, 217.8ms
    Speed: 11.0ms preprocess, 217.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_744.jpg: 480x640 2 LegoBricks, 149.3ms
    Speed: 6.0ms preprocess, 149.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_745.jpg: 480x640 5 LegoBricks, 135.2ms
    Speed: 5.0ms preprocess, 135.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_746.jpg: 480x640 3 LegoBricks, 162.6ms
    Speed: 5.0ms preprocess, 162.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_747.jpg: 480x640 4 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_748.jpg: 480x640 5 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_749.jpg: 480x640 5 LegoBricks, 175.5ms
    Speed: 5.0ms preprocess, 175.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_75.jpg: 480x640 4 LegoBricks, 276.1ms
    Speed: 8.0ms preprocess, 276.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_750.jpg: 480x640 7 LegoBricks, 152.0ms
    Speed: 6.0ms preprocess, 152.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_751.jpg: 480x640 7 LegoBricks, 185.4ms
    Speed: 6.0ms preprocess, 185.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_752.jpg: 480x640 3 LegoBricks, 216.0ms
    Speed: 19.0ms preprocess, 216.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_753.jpg: 480x640 4 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_754.jpg: 480x640 5 LegoBricks, 141.7ms
    Speed: 7.0ms preprocess, 141.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_755.jpg: 480x640 6 LegoBricks, 260.5ms
    Speed: 29.9ms preprocess, 260.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_756.jpg: 480x640 2 LegoBricks, 180.9ms
    Speed: 8.0ms preprocess, 180.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_757.jpg: 480x640 3 LegoBricks, 106.7ms
    Speed: 5.0ms preprocess, 106.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_758.jpg: 480x640 2 LegoBricks, 199.2ms
    Speed: 5.0ms preprocess, 199.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_759.jpg: 480x640 7 LegoBricks, 127.0ms
    Speed: 6.0ms preprocess, 127.0ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_76.jpg: 480x640 4 LegoBricks, 127.9ms
    Speed: 5.0ms preprocess, 127.9ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_760.jpg: 480x640 3 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_761.jpg: 480x640 5 LegoBricks, 137.6ms
    Speed: 5.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_763.jpg: 480x640 4 LegoBricks, 129.6ms
    Speed: 5.0ms preprocess, 129.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_764.jpg: 480x640 5 LegoBricks, 148.6ms
    Speed: 9.0ms preprocess, 148.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_765.jpg: 480x640 5 LegoBricks, 156.6ms
    Speed: 6.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_766.jpg: 480x640 3 LegoBricks, 174.5ms
    Speed: 6.0ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_767.jpg: 480x640 4 LegoBricks, 441.1ms
    Speed: 12.0ms preprocess, 441.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_768.jpg: 480x640 4 LegoBricks, 168.3ms
    Speed: 6.0ms preprocess, 168.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_769.jpg: 480x640 4 LegoBricks, 164.3ms
    Speed: 12.0ms preprocess, 164.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_77.jpg: 480x640 3 LegoBricks, 147.6ms
    Speed: 6.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_770.jpg: 480x640 3 LegoBricks, 154.6ms
    Speed: 6.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_771.jpg: 480x640 5 LegoBricks, 157.9ms
    Speed: 6.0ms preprocess, 157.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_772.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_774.jpg: 480x640 2 LegoBricks, 125.7ms
    Speed: 6.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_775.jpg: 480x640 6 LegoBricks, 118.7ms
    Speed: 6.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_776.jpg: 480x640 2 LegoBricks, 113.7ms
    Speed: 6.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_777.jpg: 480x640 3 LegoBricks, 131.6ms
    Speed: 6.0ms preprocess, 131.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_778.jpg: 480x640 6 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_779.jpg: 480x640 4 LegoBricks, 110.7ms
    Speed: 5.0ms preprocess, 110.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_78.jpg: 480x640 3 LegoBricks, 100.1ms
    Speed: 5.0ms preprocess, 100.1ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_780.jpg: 480x640 4 LegoBricks, 106.9ms
    Speed: 4.5ms preprocess, 106.9ms inference, 1.5ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_781.jpg: 480x640 4 LegoBricks, 102.7ms
    Speed: 5.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_782.jpg: 480x640 3 LegoBricks, 159.5ms
    Speed: 5.0ms preprocess, 159.5ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_783.jpg: 480x640 3 LegoBricks, 334.1ms
    Speed: 7.0ms preprocess, 334.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_784.jpg: 480x640 4 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_785.jpg: 480x640 4 LegoBricks, 135.0ms
    Speed: 5.0ms preprocess, 135.0ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_786.jpg: 480x640 4 LegoBricks, 128.7ms
    Speed: 6.0ms preprocess, 128.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_787.jpg: 480x640 4 LegoBricks, 136.8ms
    Speed: 5.9ms preprocess, 136.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_788.jpg: 480x640 5 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_789.jpg: 480x640 2 LegoBricks, 207.4ms
    Speed: 5.0ms preprocess, 207.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_79.jpg: 480x640 2 LegoBricks, 152.8ms
    Speed: 6.0ms preprocess, 152.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_790.jpg: 480x640 3 LegoBricks, 188.5ms
    Speed: 5.0ms preprocess, 188.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_791.jpg: 480x640 4 LegoBricks, 173.6ms
    Speed: 7.0ms preprocess, 173.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_792.jpg: 480x640 3 LegoBricks, 140.5ms
    Speed: 7.0ms preprocess, 140.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_793.jpg: 480x640 2 LegoBricks, 133.5ms
    Speed: 5.0ms preprocess, 133.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_794.jpg: 480x640 4 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_795.jpg: 480x640 4 LegoBricks, 143.2ms
    Speed: 5.0ms preprocess, 143.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_796.jpg: 480x640 6 LegoBricks, 124.1ms
    Speed: 5.0ms preprocess, 124.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_797.jpg: 480x640 8 LegoBricks, 272.2ms
    Speed: 72.8ms preprocess, 272.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_798.jpg: 480x640 4 LegoBricks, 187.9ms
    Speed: 5.0ms preprocess, 187.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_799.jpg: 480x640 8 LegoBricks, 136.5ms
    Speed: 6.0ms preprocess, 136.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_80.jpg: 480x640 3 LegoBricks, 111.7ms
    Speed: 3.0ms preprocess, 111.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_800.jpg: 480x640 9 LegoBricks, 109.6ms
    Speed: 5.0ms preprocess, 109.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_801.jpg: 480x640 7 LegoBricks, 102.7ms
    Speed: 3.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_802.jpg: 480x640 3 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_803.jpg: 480x640 3 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_804.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_805.jpg: 480x640 2 LegoBricks, 205.5ms
    Speed: 5.0ms preprocess, 205.5ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_806.jpg: 480x640 4 LegoBricks, 155.6ms
    Speed: 6.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_807.jpg: 480x640 6 LegoBricks, 160.6ms
    Speed: 8.0ms preprocess, 160.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_808.jpg: 480x640 7 LegoBricks, 159.6ms
    Speed: 6.0ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_809.jpg: 480x640 3 LegoBricks, 137.3ms
    Speed: 6.1ms preprocess, 137.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_81.jpg: 480x640 1 LegoBrick, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_810.jpg: 480x640 4 LegoBricks, 144.6ms
    Speed: 6.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_811.jpg: 480x640 6 LegoBricks, 137.6ms
    Speed: 8.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_812.jpg: 480x640 5 LegoBricks, 578.4ms
    Speed: 6.0ms preprocess, 578.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_813.jpg: 480x640 10 LegoBricks, 262.8ms
    Speed: 7.0ms preprocess, 262.8ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_814.jpg: 480x640 2 LegoBricks, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_815.jpg: 480x640 3 LegoBricks, 134.6ms
    Speed: 6.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_816.jpg: 480x640 3 LegoBricks, 125.6ms
    Speed: 6.0ms preprocess, 125.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_817.jpg: 480x640 5 LegoBricks, 127.7ms
    Speed: 5.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_818.jpg: 480x640 6 LegoBricks, 164.6ms
    Speed: 6.0ms preprocess, 164.6ms inference, 3.2ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_819.jpg: 480x640 5 LegoBricks, 303.9ms
    Speed: 6.0ms preprocess, 303.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_82.jpg: 480x640 4 LegoBricks, 177.5ms
    Speed: 15.0ms preprocess, 177.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_820.jpg: 480x640 6 LegoBricks, 183.9ms
    Speed: 5.1ms preprocess, 183.9ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_821.jpg: 480x640 4 LegoBricks, 171.5ms
    Speed: 7.0ms preprocess, 171.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_822.jpg: 480x640 8 LegoBricks, 138.6ms
    Speed: 35.9ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_823.jpg: 480x640 5 LegoBricks, 134.6ms
    Speed: 5.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_824.jpg: 480x640 5 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_825.jpg: 480x640 5 LegoBricks, 231.0ms
    Speed: 5.0ms preprocess, 231.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_826.jpg: 480x640 9 LegoBricks, 165.6ms
    Speed: 5.0ms preprocess, 165.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_827.jpg: 480x640 11 LegoBricks, 132.9ms
    Speed: 4.0ms preprocess, 132.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_828.jpg: 480x640 9 LegoBricks, 252.3ms
    Speed: 7.0ms preprocess, 252.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_829.jpg: 480x640 3 LegoBricks, 387.3ms
    Speed: 53.9ms preprocess, 387.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_83.jpg: 480x640 7 LegoBricks, 198.5ms
    Speed: 8.0ms preprocess, 198.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_830.jpg: 480x640 2 LegoBricks, 189.1ms
    Speed: 6.0ms preprocess, 189.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_831.jpg: 480x640 6 LegoBricks, 202.5ms
    Speed: 12.0ms preprocess, 202.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_832.jpg: 480x640 6 LegoBricks, 191.4ms
    Speed: 7.0ms preprocess, 191.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_833.jpg: 480x640 9 LegoBricks, 168.5ms
    Speed: 6.0ms preprocess, 168.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_834.jpg: 480x640 7 LegoBricks, 167.6ms
    Speed: 6.0ms preprocess, 167.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_835.jpg: 480x640 9 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_836.jpg: 480x640 6 LegoBricks, 102.7ms
    Speed: 5.0ms preprocess, 102.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_837.jpg: 480x640 15 LegoBricks, 120.7ms
    Speed: 5.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_838.jpg: 480x640 1 LegoBrick, 105.7ms
    Speed: 5.0ms preprocess, 105.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_839.jpg: 480x640 5 LegoBricks, 109.5ms
    Speed: 6.0ms preprocess, 109.5ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_84.jpg: 480x640 3 LegoBricks, 119.7ms
    Speed: 6.0ms preprocess, 119.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_840.jpg: 480x640 5 LegoBricks, 124.3ms
    Speed: 61.0ms preprocess, 124.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_841.jpg: 480x640 2 LegoBricks, 151.6ms
    Speed: 5.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_842.jpg: 480x640 5 LegoBricks, 346.1ms
    Speed: 5.0ms preprocess, 346.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_843.jpg: 480x640 5 LegoBricks, 169.5ms
    Speed: 5.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_844.jpg: 480x640 6 LegoBricks, 142.6ms
    Speed: 6.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_846.jpg: 480x640 6 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_847.jpg: 480x640 3 LegoBricks, 205.3ms
    Speed: 6.0ms preprocess, 205.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_849.jpg: 480x640 3 LegoBricks, 614.2ms
    Speed: 42.9ms preprocess, 614.2ms inference, 6.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_85.jpg: 480x640 2 LegoBricks, 400.9ms
    Speed: 41.9ms preprocess, 400.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_852.jpg: 480x640 3 LegoBricks, 154.6ms
    Speed: 8.0ms preprocess, 154.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_853.jpg: 480x640 3 LegoBricks, 130.7ms
    Speed: 6.0ms preprocess, 130.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_854.jpg: 480x640 3 LegoBricks, 135.6ms
    Speed: 6.0ms preprocess, 135.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_855.jpg: 480x640 5 LegoBricks, 169.5ms
    Speed: 5.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_856.jpg: 480x640 3 LegoBricks, 158.6ms
    Speed: 17.0ms preprocess, 158.6ms inference, 2.2ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_857.jpg: 480x640 5 LegoBricks, 148.2ms
    Speed: 5.0ms preprocess, 148.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_858.jpg: 480x640 4 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_859.jpg: 480x640 6 LegoBricks, 139.6ms
    Speed: 7.0ms preprocess, 139.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_86.jpg: 480x640 4 LegoBricks, 368.0ms
    Speed: 6.0ms preprocess, 368.0ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_860.jpg: 480x640 4 LegoBricks, 199.5ms
    Speed: 13.0ms preprocess, 199.5ms inference, 10.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_861.jpg: 480x640 2 LegoBricks, 199.7ms
    Speed: 6.0ms preprocess, 199.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_862.jpg: 480x640 1 LegoBrick, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_863.jpg: 480x640 1 LegoBrick, 158.6ms
    Speed: 6.0ms preprocess, 158.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_865.jpg: 480x640 7 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_866.jpg: 480x640 8 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_867.jpg: 480x640 5 LegoBricks, 151.6ms
    Speed: 6.0ms preprocess, 151.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_868.jpg: 480x640 3 LegoBricks, 153.6ms
    Speed: 6.0ms preprocess, 153.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_869.jpg: 480x640 9 LegoBricks, 255.2ms
    Speed: 6.0ms preprocess, 255.2ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_87.jpg: 480x640 5 LegoBricks, 153.6ms
    Speed: 7.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_870.jpg: 480x640 4 LegoBricks, 137.0ms
    Speed: 6.0ms preprocess, 137.0ms inference, 2.1ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_871.jpg: 480x640 6 LegoBricks, 129.7ms
    Speed: 6.0ms preprocess, 129.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_872.jpg: 480x640 1 LegoBrick, 127.7ms
    Speed: 6.0ms preprocess, 127.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_873.jpg: 480x640 3 LegoBricks, 151.5ms
    Speed: 6.0ms preprocess, 151.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_874.jpg: 480x640 4 LegoBricks, 143.1ms
    Speed: 6.0ms preprocess, 143.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_875.jpg: 480x640 4 LegoBricks, 226.4ms
    Speed: 7.0ms preprocess, 226.4ms inference, 4.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_876.jpg: 480x640 4 LegoBricks, 443.1ms
    Speed: 46.9ms preprocess, 443.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_878.jpg: 480x640 3 LegoBricks, 184.3ms
    Speed: 4.0ms preprocess, 184.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_88.jpg: 480x640 2 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_880.jpg: 480x640 4 LegoBricks, 110.0ms
    Speed: 5.0ms preprocess, 110.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_881.jpg: 480x640 6 LegoBricks, 104.7ms
    Speed: 5.0ms preprocess, 104.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_882.jpg: 480x640 3 LegoBricks, 108.7ms
    Speed: 5.0ms preprocess, 108.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_883.jpg: 480x640 4 LegoBricks, 115.7ms
    Speed: 5.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_884.jpg: 480x640 8 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_885.jpg: 480x640 4 LegoBricks, 123.7ms
    Speed: 5.0ms preprocess, 123.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_886.jpg: 480x640 3 LegoBricks, 121.7ms
    Speed: 5.0ms preprocess, 121.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_887.jpg: 480x640 2 LegoBricks, 184.4ms
    Speed: 6.0ms preprocess, 184.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_888.jpg: 480x640 1 LegoBrick, 117.3ms
    Speed: 5.0ms preprocess, 117.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_889.jpg: 480x640 4 LegoBricks, 169.5ms
    Speed: 7.0ms preprocess, 169.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_89.jpg: 480x640 4 LegoBricks, 169.4ms
    Speed: 7.0ms preprocess, 169.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_890.jpg: 480x640 4 LegoBricks, 133.6ms
    Speed: 7.0ms preprocess, 133.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_891.jpg: 480x640 3 LegoBricks, 193.5ms
    Speed: 5.0ms preprocess, 193.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_892.jpg: 480x640 5 LegoBricks, 409.4ms
    Speed: 18.9ms preprocess, 409.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_893.jpg: 480x640 6 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_894.jpg: 480x640 5 LegoBricks, 217.3ms
    Speed: 5.0ms preprocess, 217.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_895.jpg: 480x640 7 LegoBricks, 328.1ms
    Speed: 6.0ms preprocess, 328.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_896.jpg: 480x640 6 LegoBricks, 240.1ms
    Speed: 30.9ms preprocess, 240.1ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_897.jpg: 480x640 3 LegoBricks, 142.6ms
    Speed: 7.0ms preprocess, 142.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_898.jpg: 480x640 2 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_899.jpg: 480x640 5 LegoBricks, 192.4ms
    Speed: 4.0ms preprocess, 192.4ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_9.jpg: 640x480 1 LegoBrick, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_90.jpg: 480x640 3 LegoBricks, 140.3ms
    Speed: 6.0ms preprocess, 140.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_900.jpg: 480x640 4 LegoBricks, 116.7ms
    Speed: 6.0ms preprocess, 116.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_901.jpg: 480x640 5 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_903.jpg: 480x640 6 LegoBricks, 118.7ms
    Speed: 5.0ms preprocess, 118.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_904.jpg: 480x640 3 LegoBricks, 112.7ms
    Speed: 6.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_905.jpg: 480x640 4 LegoBricks, 107.7ms
    Speed: 5.0ms preprocess, 107.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_906.jpg: 480x640 7 LegoBricks, 124.7ms
    Speed: 5.0ms preprocess, 124.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_907.jpg: 480x640 6 LegoBricks, 173.5ms
    Speed: 5.0ms preprocess, 173.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_908.jpg: 480x640 5 LegoBricks, 125.7ms
    Speed: 5.0ms preprocess, 125.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_909.jpg: 480x640 3 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_91.jpg: 480x640 2 LegoBricks, 177.8ms
    Speed: 6.0ms preprocess, 177.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_910.jpg: 480x640 3 LegoBricks, 494.7ms
    Speed: 8.0ms preprocess, 494.7ms inference, 2.7ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_911.jpg: 480x640 7 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_912.jpg: 480x640 7 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_913.jpg: 480x640 5 LegoBricks, 156.6ms
    Speed: 6.0ms preprocess, 156.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_914.jpg: 480x640 3 LegoBricks, 236.4ms
    Speed: 6.0ms preprocess, 236.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_915.jpg: 480x640 6 LegoBricks, 338.0ms
    Speed: 73.8ms preprocess, 338.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_916.jpg: 480x640 4 LegoBricks, 131.6ms
    Speed: 6.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_917.jpg: 480x640 3 LegoBricks, 146.6ms
    Speed: 6.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_918.jpg: 480x640 3 LegoBricks, 114.7ms
    Speed: 4.0ms preprocess, 114.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_919.jpg: 480x640 3 LegoBricks, 123.7ms
    Speed: 4.0ms preprocess, 123.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_92.jpg: 480x640 3 LegoBricks, 183.3ms
    Speed: 6.0ms preprocess, 183.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_920.jpg: 480x640 3 LegoBricks, 120.7ms
    Speed: 6.0ms preprocess, 120.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_921.jpg: 480x640 4 LegoBricks, 117.7ms
    Speed: 4.0ms preprocess, 117.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_923.jpg: 480x640 4 LegoBricks, 238.4ms
    Speed: 7.0ms preprocess, 238.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_924.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_925.jpg: 480x640 3 LegoBricks, 202.5ms
    Speed: 6.0ms preprocess, 202.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_926.jpg: 480x640 4 LegoBricks, 229.0ms
    Speed: 7.0ms preprocess, 229.0ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_927.jpg: 480x640 3 LegoBricks, 210.4ms
    Speed: 6.0ms preprocess, 210.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_928.jpg: 480x640 3 LegoBricks, 140.6ms
    Speed: 5.0ms preprocess, 140.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_929.jpg: 480x640 5 LegoBricks, 113.7ms
    Speed: 6.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_93.jpg: 480x640 2 LegoBricks, 182.5ms
    Speed: 66.8ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_930.jpg: 480x640 3 LegoBricks, 167.5ms
    Speed: 6.0ms preprocess, 167.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_932.jpg: 480x640 4 LegoBricks, 176.5ms
    Speed: 6.0ms preprocess, 176.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_933.jpg: 480x640 3 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_934.jpg: 480x640 3 LegoBricks, 144.6ms
    Speed: 5.0ms preprocess, 144.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_935.jpg: 480x640 3 LegoBricks, 122.7ms
    Speed: 6.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_936.jpg: 480x640 4 LegoBricks, 142.6ms
    Speed: 5.0ms preprocess, 142.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_937.jpg: 480x640 4 LegoBricks, 148.6ms
    Speed: 6.0ms preprocess, 148.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_938.jpg: 480x640 4 LegoBricks, 150.6ms
    Speed: 5.0ms preprocess, 150.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_939.jpg: 480x640 4 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_94.jpg: 480x640 2 LegoBricks, 125.5ms
    Speed: 59.8ms preprocess, 125.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_940.jpg: 480x640 4 LegoBricks, 230.4ms
    Speed: 5.0ms preprocess, 230.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_941.jpg: 480x640 5 LegoBricks, 318.1ms
    Speed: 9.0ms preprocess, 318.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_942.jpg: 480x640 5 LegoBricks, 139.6ms
    Speed: 10.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_943.jpg: 480x640 4 LegoBricks, 138.6ms
    Speed: 5.0ms preprocess, 138.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_944.jpg: 480x640 5 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_945.jpg: 480x640 4 LegoBricks, 113.7ms
    Speed: 5.0ms preprocess, 113.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_946.jpg: 480x640 4 LegoBricks, 126.7ms
    Speed: 4.9ms preprocess, 126.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_947.jpg: 480x640 4 LegoBricks, 111.7ms
    Speed: 7.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_949.jpg: 480x640 6 LegoBricks, 180.5ms
    Speed: 5.0ms preprocess, 180.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_95.jpg: 480x640 3 LegoBricks, 145.6ms
    Speed: 6.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_950.jpg: 480x640 4 LegoBricks, 121.7ms
    Speed: 6.0ms preprocess, 121.7ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_951.jpg: 480x640 4 LegoBricks, 137.6ms
    Speed: 6.0ms preprocess, 137.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_952.jpg: 480x640 5 LegoBricks, 147.6ms
    Speed: 5.0ms preprocess, 147.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_953.jpg: 480x640 7 LegoBricks, 215.4ms
    Speed: 6.0ms preprocess, 215.4ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_954.jpg: 480x640 4 LegoBricks, 130.6ms
    Speed: 6.0ms preprocess, 130.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_955.jpg: 480x640 5 LegoBricks, 140.6ms
    Speed: 6.0ms preprocess, 140.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_956.jpg: 480x640 5 LegoBricks, 414.8ms
    Speed: 5.0ms preprocess, 414.8ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_957.jpg: 480x640 7 LegoBricks, 334.1ms
    Speed: 29.9ms preprocess, 334.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_958.jpg: 480x640 3 LegoBricks, 144.6ms
    Speed: 6.0ms preprocess, 144.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_959.jpg: 480x640 4 LegoBricks, 138.6ms
    Speed: 6.0ms preprocess, 138.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_96.jpg: 480x640 4 LegoBricks, 115.7ms
    Speed: 4.0ms preprocess, 115.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_960.jpg: 480x640 10 LegoBricks, 112.7ms
    Speed: 4.0ms preprocess, 112.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_961.jpg: 480x640 8 LegoBricks, 122.7ms
    Speed: 5.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_962.jpg: 480x640 6 LegoBricks, 152.6ms
    Speed: 6.0ms preprocess, 152.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_963.jpg: 480x640 4 LegoBricks, 141.6ms
    Speed: 6.0ms preprocess, 141.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_965.jpg: 480x640 3 LegoBricks, 145.6ms
    Speed: 6.0ms preprocess, 145.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_966.jpg: 480x640 3 LegoBricks, 132.6ms
    Speed: 5.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_967.jpg: 480x640 3 LegoBricks, 133.6ms
    Speed: 5.0ms preprocess, 133.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_968.jpg: 480x640 3 LegoBricks, 172.4ms
    Speed: 5.0ms preprocess, 172.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_969.jpg: 480x640 6 LegoBricks, 111.7ms
    Speed: 5.0ms preprocess, 111.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_97.jpg: 480x640 2 LegoBricks, 132.6ms
    Speed: 7.0ms preprocess, 132.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_970.jpg: 480x640 6 LegoBricks, 109.7ms
    Speed: 5.0ms preprocess, 109.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_972.jpg: 480x640 3 LegoBricks, 254.3ms
    Speed: 6.0ms preprocess, 254.3ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_973.jpg: 480x640 8 LegoBricks, 334.3ms
    Speed: 6.7ms preprocess, 334.3ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_974.jpg: 480x640 11 LegoBricks, 238.4ms
    Speed: 9.0ms preprocess, 238.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_975.jpg: 480x640 9 LegoBricks, 172.5ms
    Speed: 5.0ms preprocess, 172.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_977.jpg: 480x640 6 LegoBricks, 134.6ms
    Speed: 7.0ms preprocess, 134.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_978.jpg: 480x640 4 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_979.jpg: 480x640 6 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_98.jpg: 480x640 5 LegoBricks, 167.2ms
    Speed: 7.0ms preprocess, 167.2ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_980.jpg: 480x640 6 LegoBricks, 131.6ms
    Speed: 5.0ms preprocess, 131.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_981.jpg: 480x640 5 LegoBricks, 139.6ms
    Speed: 5.0ms preprocess, 139.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_982.jpg: 480x640 2 LegoBricks, 153.6ms
    Speed: 15.0ms preprocess, 153.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_983.jpg: 480x640 6 LegoBricks, 159.6ms
    Speed: 4.9ms preprocess, 159.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_984.jpg: 480x640 3 LegoBricks, 155.6ms
    Speed: 5.0ms preprocess, 155.6ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_985.jpg: 480x640 6 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_986.jpg: 480x640 4 LegoBricks, 130.7ms
    Speed: 5.0ms preprocess, 130.7ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_987.jpg: 480x640 4 LegoBricks, 214.4ms
    Speed: 6.0ms preprocess, 214.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_988.jpg: 480x640 7 LegoBricks, 170.9ms
    Speed: 5.0ms preprocess, 170.9ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_989.jpg: 480x640 6 LegoBricks, 250.2ms
    Speed: 6.0ms preprocess, 250.2ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_99.jpg: 480x640 4 LegoBricks, 182.5ms
    Speed: 6.0ms preprocess, 182.5ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_990.jpg: 480x640 3 LegoBricks, 166.6ms
    Speed: 6.0ms preprocess, 166.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_992.jpg: 480x640 5 LegoBricks, 338.7ms
    Speed: 9.0ms preprocess, 338.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_993.jpg: 480x640 9 LegoBricks, 216.4ms
    Speed: 9.0ms preprocess, 216.4ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_994.jpg: 480x640 7 LegoBricks, 132.6ms
    Speed: 6.0ms preprocess, 132.6ms inference, 3.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_995.jpg: 480x640 2 LegoBricks, 122.7ms
    Speed: 6.0ms preprocess, 122.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_996.jpg: 480x640 3 LegoBricks, 136.6ms
    Speed: 5.0ms preprocess, 136.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_997.jpg: 480x640 6 LegoBricks, 212.1ms
    Speed: 6.0ms preprocess, 212.1ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_998.jpg: 480x640 6 LegoBricks, 146.6ms
    Speed: 5.0ms preprocess, 146.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Raw_images\Resized\image_999.jpg: 480x640 6 LegoBricks, 158.6ms
    Speed: 7.0ms preprocess, 158.6ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    


```python
showcase_dataset_images(cropped_bricks_folder)
```


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_29_0.png)
    


# Second Data Collection

After successfully detecting Lego pieces in cluttered images, I needed a more refined dataset to classify each piece by its unique dimensions accurately. This required a systematic approach to data collection and gave me an opportunity to solve problems creatively.

I built a setup using a spinning base to capture each Lego piece from multiple angles. The goal was to create a structured dataset where each brick could be thoroughly analyzed. The spinning base enabled consistent 360-degree views of each piece, ensuring comprehensive coverage for all classes. This approach allowed me to gather detailed images essential for building a more robust classification model.
The cropping script I previously developed was highly valuable during this phase. Using YOLO, I detected the Lego pieces in the images, and the cropping script allowed me to isolate each piece effectively. Working in batches made it easy to organize the cropped pieces by class, making the dataset more manageable and efficient.

I also applied my creativity during this stage by improvising a homemade setup with accessible materials. Using a rotating platform, consistent lighting, and a fixed camera position ensured high-quality, standardized captures. This combination of creative setup and automated tools highlighted my ability to innovate while staying resourceful, reinforcing my practical skills in data collection and preprocessing.



# Failed Classification Attempts

After completing the initial detection and cropping stages, I moved on to the classification task. The goal was to classify each Lego piece based on its unique dimensions and features. However, my early attempts at classification met with limited success due to several challenges inherent in the dataset.

First, although the dataset I created was diverse, it was ultimately too small to support accurate classification across all 26 classes. Some classes were significantly underrepresented, making it difficult for the model to learn effectively. This imbalance led to overfitting for certain classes, while the model struggled to generalize to others.

Additionally, the level of fine detail required to differentiate between certain Lego pieces posed a significant challenge. Many pieces had subtle differences in dimensions and features that were not sufficiently captured by the available data. The fine-grained nature of the classification required a level of representation that my dataset simply could not provide.

Ultimately, the combination of a small dataset, class imbalance, and insufficient feature detail led to the failure of the classification task. This experience underscored the importance of having a well-balanced dataset and ensuring that all classes are adequately represented, especially when dealing with subtle, fine-grained differences. Despite the challenges, this experience provided valuable insights into the difficulties of image-based classification and highlighted the limitations of my initial approach, which will inform future iterations of the project.



# Adaptation: Stud Detection for Dimensional Classification

After encountering challenges with the initial classification attempts, I decided to adapt my approach to determine the dimensions of each Lego piece. Instead of attempting to classify the entire piece directly, I focused on a key defining feature: the studs on top of each Lego brick. By training a model to detect these studs, I could use their coordinates to algorithmically determine the dimensions of each piece.

To achieve this, I used YOLO once again, as it had already proven effective for object detection earlier in the project. I annotated bounding boxes specifically around the studs on top of the Lego pieces, creating a new set of labeled images for training. The goal was to leverage the detected stud coordinates to infer important characteristics, such as the number of studs, which directly correlated with the dimensions of the brick.

With this adapted approach, YOLO was trained to detect studs reliably, and the resulting coordinates were used in an algorithmic process to classify the pieces based on their dimensions. By narrowing the focus to a specific feature of each piece, I simplified the classification task and achieved a more accurate estimation of the brick dimensions. This adaptation not only made the process more efficient but also underscored the importance of breaking down complex problems into smaller, more manageable components.





```python
# !labelme
```


```python
def convert_points_to_bounding_boxes(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all JSON files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            json_path = os.path.join(input_folder, file_name)

            # Load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Load corresponding image to get dimensions
            image_path = os.path.join(input_folder, data["imagePath"])
            with Image.open(image_path) as img:
                image_width, image_height = img.size
                image_area = image_width * image_height

            # Calculate the total area for the bounding boxes (1/5 of the image area)
            total_box_area = image_area / 5

            # Count the number of points in the JSON file
            points = [shape for shape in data['shapes'] if shape['shape_type'] == 'point']
            num_points = len(points)

            if num_points == 0:
                continue

            # Calculate the area for each box and determine the side length (boxes are squares)
            box_area = total_box_area / num_points
            box_side_length = math.sqrt(box_area)

            # Create bounding boxes centered around each point
            new_shapes = []
            for point in points:
                x, y = point['points'][0]
                half_side = box_side_length / 2

                # Calculate the coordinates of the bounding box
                x_min = max(0, x - half_side)
                y_min = max(0, y - half_side)
                x_max = min(image_width, x + half_side)
                y_max = min(image_height, y + half_side)

                # Create a new shape entry for the bounding box
                new_shape = {
                    "label": point["label"],
                    "points": [[x_min, y_min], [x_max, y_max]],
                    "group_id": point["group_id"],
                    "description": point["description"],
                    "shape_type": "rectangle",
                    "flags": point["flags"]
                }
                new_shapes.append(new_shape)

            # Replace the old shapes with the new bounding boxes
            data['shapes'] = new_shapes

            # Save the modified JSON to the output folder
            output_path = os.path.join(output_folder, file_name)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

            #copy the corresponding image to the output folder
            shutil.copy(image_path, output_folder)

            

# Example usage
input_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Cropped_bricks"  # Replace with the path to your input folder
output_folder = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Cropped_with_boxes_test"  # Replace with the path to your output folder
convert_points_to_bounding_boxes(input_folder, output_folder)
```


```python
labelme_jsons_to_yolos(output_folder)
```


```python
def create_missing_txt_files(folder_path):
    """
    This function iterates through all .jpg files in a given folder and checks if there is a corresponding .txt file.
    If no such .txt file exists, it creates an empty .txt file. This is useful when preparing datasets for training
    object detection models like YOLO, where each image must have a paired annotation file.

    Including images with empty .txt files during YOLO training indicates that these images do not contain any
    objects of interest, helping the model learn background patterns and reducing false positives.

    Args:
        folder_path (str): The path to the folder containing .jpg images.
    """
    # Iterate over all .jpg files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            # Construct the expected .txt file name
            txt_file_name = file_name.replace('.jpg', '.txt')
            txt_file_path = os.path.join(folder_path, txt_file_name)

            # Check if the .txt file exists, if not create an empty one
            if not os.path.exists(txt_file_path):
                with open(txt_file_path, 'w') as f:
                    pass  # Create an empty .txt file
```


```python
create_missing_txt_files(output_folder)
```


```python
visualize_yolo_annotated_images(output_folder, output_folder, num_images=6, class_names=["Stud"])
```


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_38_0.png)
    



```python
Stud_Yolo_data_path = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\YOLO_studs"

yaml_stud_detectron_path = prepare_yolo_dataset(output_folder, Stud_Yolo_data_path, class_names=["Stud"])
```

    
    YOLO Dataset Preparation Summary:
    - Training set: 1740 images and labels
    - Validation set: 436 images and labels
    - Dataset YAML file created at: C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\YOLO_studs\data\dataset.yaml
    


```python
# givena  folder. return a dataframe with the images paths, images heights, images widths, and the number of bounding boxes in the image.

def get_image_info(folder_path):

    # Create an empty list to store image information
    image_info = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            width, height = image.size

            # Count the number of bounding boxes in the corresponding .txt file
            txt_file_name = file_name.replace('.jpg', '.txt')
            txt_file_path = os.path.join(folder_path, txt_file_name)
            num_boxes = 0
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    num_boxes = len(f.readlines())

            # Append the image information to the list
            image_info.append({
                'image_path': image_path,
                'width': width,
                'height': height,
                'num_boxes': num_boxes
            })

    # Create a DataFrame from the list of image information
    df = pd.DataFrame(image_info)

    return df
```


```python
image_data = get_image_info(output_folder)
```


```python
# using image_data plot a histogram  of the number of images againts their width and height

def plot_image_info_histograms(image_data):

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), facecolor='black')

    # Plot histograms for image width, height, and number of bounding boxes
    for i, col in enumerate(['width', 'height']):
        ax = axes[i]  # Corrected indexing
        ax.hist(image_data[col], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of Image {col.capitalize()}', color='white')
        ax.set_xlabel(col.capitalize(), color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


plot_image_info_histograms(image_data)
```


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_42_0.png)
    



```python
# Stud_detectron_model = YOLO_oneLabel_train(yaml_stud_detectron_path, Imgsz = 128, label = "Stud", epochs = 150)
Stud_detectron_model_path = r"C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Models\YOLOdetectron_Stud_yolov8n_20241030_151507.pt"
Stud_detectron_model = YOLO(Stud_detectron_model_path)
```


```python
output_folder
```




    'C:\\Users\\User\\Desktop\\Final_Streamlit_Portfolio_Projects\\Brick_detectron_folder\\Cropped_with_boxes_test'




```python
image_files = [f for f in os.listdir(output_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
random_image = random.choice(image_files)

predict_and_plot(os.path.join(output_folder, random_image), Stud_detectron_model, class_names=["Stud"], conf_threshold=0.20)
```

    
    image 1/1 C:\Users\User\Desktop\Final_Streamlit_Portfolio_Projects\Brick_detectron_folder\Cropped_with_boxes_test\image_623_LegoBrick_4_c76.jpg: 640x576 2 LegoBricks, 122.3ms
    Speed: 5.0ms preprocess, 122.3ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 576)
    


    
![png](LEGO_Bricks_Project_Miguel_Di_Lalla_files/LEGO_Bricks_Project_Miguel_Di_Lalla_45_1.png)
    


# Conclusions, Reflections, and Lessons Learned

Reflecting on this project, I encountered numerous challenges, adaptations, and valuable lessons in tackling Lego piece detection and classification. Each phase—from data collection to model training and refining my approach—provided insights into both the technical aspects and the problem-solving mindset crucial for success in machine learning.

One key lesson was the importance of a high-quality dataset. Early classification attempts faltered due to a small dataset size, class imbalance, and insufficient feature representation. This experience underscored how vital data quality is for model performance and showed that investing time in preparing balanced, comprehensive data can save significant effort during later stages.

Adaptability proved essential. When the initial model struggled, I adapted by breaking the problem down and focusing on a simpler feature—the studs on each brick—which made the task more manageable. This highlighted that simplifying complex problems and iterating after setbacks often results in effective learning and solutions.

The project also demonstrated the strengths and limitations of different machine learning tools. YOLO was highly efficient for object detection but had limitations when it came to fine-grained classification. Understanding the appropriate context for each model's use is a critical skill, and this project helped me refine that understanding.

In conclusion, this project taught me valuable technical skills in computer vision, dataset preparation, and model adaptation, while also deepening my appreciation for the iterative nature of machine learning. Facing and overcoming challenges through creative problem-solving leads to growth, and I am excited to carry these lessons into future projects.

