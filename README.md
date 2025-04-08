![Hero Banner](https://migueldilalla.github.io/assets/branding-elements/brickssifier-herobanner.jpg)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1+-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-red.svg)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Albumentations](https://img.shields.io/badge/Albumentations-2.0+-brightgreen.svg)](https://albumentations.ai)
[![Rich](https://img.shields.io/badge/Rich-13.0+-9933CC.svg)](https://github.com/Textualize/rich)
[![Click](https://img.shields.io/badge/Click-8.1+-yellow.svg)](https://click.palletsprojects.com)

# 🧱 Project: Brickssifier_Studwise

> *It classifies bricks. Imperfectly. Passionately.*

> *AI is learning from us. But what are we teaching it?*

## 📌 The Question:

> *"If I can recognize most LEGO bricks at a glance, can I teach a machine vision algorithm to do the same?"*

This project is a compact but ambitious attempt to answer that question — by building a real machine learning pipeline that classifies LEGO bricks based on top-view images. It integrates two finetuned YOLOv8 models and a geometric post-processing algorithm for dimension inference.

While powerful commercial systems like [Brickognize](https://brickognize.com/) or [BrickIt](https://brickit.app/) achieve impressive results across thousands of parts, they also operate at a scale backed by teams, servers, and datasets orders of magnitude larger than mine. Brickssifier is a personal engineering exercise: built from scratch, trained on ~2000 images, and deployed with curiosity and care.

---

## 🧪 My Approach: When ML Meets Geometry

After failing to train a reliable multiclass classifier, I narrowed my goal to a **reduced set of 14 basic brick classes**, and restructured the task into a pipeline of modular steps.

### 🚀 Brick Dimension Classification Pipeline
```python
1. Detect LEGO bricks in an image with YOLOv8 (model 1)
2. For each detected box:
   a. Crop the region from the original image (NumPy array)
   b. Detect studs within the crop using YOLOv8 (model 2)
   c. Extract (x, y) center coordinates of all studs
   d. Pass the points to a regression-based geometric algorithm
   e. Predict top-view dimension (e.g., 2x4 or 1x8)
```
This hybrid strategy — combining detection models and classic geometry — allows disambiguation of brick types that share the same stud count.

---

## 🧱 Dataset 1: Brick Detection (Raw & Labeled)

This dataset contains 2000+ images of individual LEGO bricks captured under natural lighting with varied backgrounds.

- 📸 Annotated using **LabelMe**
- 🔁 Converted to **YOLO format** using custom scripts
- 🧩 Contains 14 base classes (e.g., 2x2, 1x4, 2x4...)

### 📦 Dataset Availability

The full annotated dataset is available for download and use in your own projects.

> 🧷 **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/migueldilalla/spiled-lego-bricks)**

### 📷 Example Grid (Annotated Images)
> *(Images will load from the `assets/dataset_bricks/` folder)*

```
![Grid Placeholder - Brick Detection Examples](#TODO)
```

---

## 🔍 Dataset 2: Stud Detection (Cropped Inputs)

Each cropped brick image is relabeled with visible **stud positions**:
- Keypoints marked manually
- Transformed into **bounding boxes**
- Converted to **YOLO keypoint format** using helper scripts

> 🔧 Conversion tools available in `utils/label_conversion.py`
> 📦 **[Download Stud Dataset from Kaggle](#TODO)**

### 📷 Example Grid (Cropped Brick + Stud Annotations)
> *(Images will load from `assets/dataset_studs/` folder)*

```
![Grid Placeholder - Stud Detection Examples](#TODO)
```

---

## 📐 Stud Geometry Classifier: Algorithm Summary

When a stud count maps to multiple possible dimensions, spatial logic is used to disambiguate the brick type.

### 🎓 Regression-Based Pattern Analysis

- Extract stud center points: `(x1, y1), (x2, y2), ...`
- Fit a linear regression line to the points
- Measure deviation of each point from the line
- If deviation < threshold → studs aligned → shape is linear (e.g., 1x8)
- Else → grid pattern → 2D shape (e.g., 2x4)

> 🧠 Diagram Placeholder: Stud Geometry Classifier Logic → [#TODO: Insert visual explanation]

---

## 🧪 Model Training Notebooks

All models were trained on **Kaggle Notebooks**, using Ultralytics YOLOv8 (`yolov8n.pt`) with extensive data augmentation.

- 📄 Single-class brick detection model
- 📄 Stud detection model

> 🔗 **[Kaggle Notebook: Brick Detection Finetuning](#TODO)**
> 🔗 **[Kaggle Notebook: Stud Detection Finetuning](#TODO)**

---

## 🖼️ Streamlit App Demo

You can test the full pipeline interactively on the web. Upload an image, get the result with annotations, metadata, and brick prediction.

- 🔧 Built with Streamlit + OpenCV + EXIF
- ⚙️ Includes a metadata fingerprint per inference

> 🌐 **[Try the App on Streamlit](#TODO)**

---

## 💬 Notes, Reflections & Agradecimientos

- This project was built as a **learning milestone** — not a commercial product.
- I worked solo with minimal resources, making every step an exercise in creativity, debugging, and clear thinking.
- I learned about: image labeling, ML model finetuning, keypoint detection, CLI/UX design, and metadata engineering.

> 🙏 Thanks to the open-source community and the tools that made this possible: YOLOv8, LabelMe, Albumentations, Rich, Streamlit.

📬 Feel free to explore the repo, fork it, or reach out!

🔗 [My Portfolio](https://migueldilalla.github.io/)  
💼 [My LinkedIn](https://www.linkedin.com/in/MiguelDiLalla/)  
📦 [Project Repository](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)

---

© Miguel Di Lalla — LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project.

