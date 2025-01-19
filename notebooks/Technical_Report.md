## LEGO Bricks Identification Project: A Technical Report

### 1. Introduction and Motivation 🌟

This project leverages the **`lego-bricks-ml-vision`** package to implement an efficient pipeline for detecting and classifying LEGO bricks. Inspired by the challenge of identifying specific pieces within a cluttered set of LEGO bricks, this project combines computer vision, machine learning, and dataset management to achieve scalable and replicable results.

The key objectives include:

1. Designing a pipeline for object detection using YOLOv8.
2. Documenting the process to ensure reproducibility and scalability.
3. Providing tools for visualization and analysis to showcase the model’s performance.

### 2. Dataset Creation 📝

The dataset creation process is streamlined using the **`lego-bricks-ml-vision`** package. This package provides commands for downloading datasets, preprocessing images, and converting annotations.

#### 2.1 Dataset Overview 

The dataset used for this project is hosted on Kaggle:
- **Dataset Name**: [Spiled LEGO Bricks](https://www.kaggle.com/datasets/migueldilalla/spiled-lego-bricks)
- **Contents**:
  - **Images**: 1803 images (600×800 resolution)
  - **Annotations**: LabelMe-compatible `.txt` files with bounding box data.

#### 2.2 Setting Up the Environment 

Install the **`lego-bricks-ml-vision`** package from PyPI:

```python
!pip install lego-bricks-ml-vision
```

Configure the pipeline:

```python
from lego_bricks_ml_vision import setup_environment
setup_environment()
```

This command ensures that all dependencies are installed and that your environment is ready.

#### 2.3 Downloading the Dataset 

The dataset can be directly downloaded and extracted using the following commands:

```python
!run-pipeline download-dataset \
    --kaggle-dataset "migueldilalla/spiled-lego-bricks" \
    --output-dir "datasets"
```

#### 2.4 Preprocessing Images 

Resize the images to a consistent size (e.g., 256x256) for model training:

```python
!run-pipeline preprocess-images \
    --input-dir "datasets/Images_600x800" \
    --output-dir "datasets/processed_images" \
    --target-size 256
```

#### 2.5 Converting Annotations 

Convert annotations from LabelMe format to YOLO format:

```python
!run-pipeline labelme-to-yolo \
    --input-folder "datasets/LabelMe_txt_bricks" \
    --output-folder "datasets/annotations"
```

By automating these tasks, the package ensures consistency and reduces manual effort.

---

### 3. Model Training 🧬

The project uses YOLOv8 for LEGO brick detection. Training is performed using the preprocessed dataset and YOLO-compatible annotations.

#### 3.1 Training the YOLO Model

The `train_yolo_pipeline` function enables straightforward model training:

```python
!run-pipeline train-yolo \
    --dataset-path "datasets" \
    --epochs 50 \
    --img-size 256
```

#### 3.2 Validating the Model

Evaluate the trained model on test images:

```python
!run-pipeline test-model \
    --model-path "YOLO_Lego_Detection/best.pt" \
    --test-images-dir "test_images" \
    --output-dir "results"
```

---

### 4. Visualization and Results 🎨

#### 4.1 Visualizing Results

Visualize predictions and annotations using:

```python
!run-visualize annotate-results \
    --model-path "YOLO_Lego_Detection/best.pt" \
    --input-folder "datasets/processed_images" \
    --output-folder "presentation/model_results"
```

#### 4.2 Generating Comparison Grids

Compare predictions against ground truth:

```python
!run-visualize comparison-grid \
    --model-path "YOLO_Lego_Detection/best.pt" \
    --input-folder "datasets/processed_images" \
    --output-folder "presentation/comparison"
```

#### 4.3 Creating Presentation Grids

Summarize dataset samples in a grid format:

```python
!run-visualize create-grid \
    --input-folder "datasets/processed_images" \
    --output-folder "presentation/dataset_samples"
```

---

### 5. Reflection and Future Work 🔄

This project demonstrates the effectiveness of modular pipelines for scalable machine learning workflows. Key insights include:

- **Data Quality**: High-quality, annotated datasets significantly enhance model performance.
- **Modularity**: Breaking down the pipeline into distinct stages improves reproducibility.
- **Visualization**: Effective visual tools aid in debugging and communicating results.

#### Future Improvements:
1. Expanding the dataset to include more LEGO pieces.
2. Integrating semi-automated annotation tools.
3. Developing an interactive interface for real-time predictions.

---

For more details, refer to the [LEGO Bricks ML Vision Documentation](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision).

