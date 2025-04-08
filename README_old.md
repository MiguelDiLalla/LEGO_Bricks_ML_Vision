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

A portfolio case study on teaching machines to identify LEGO bricks — blending deep learning with playful engineering, and showcasing rapid technical growth.

---

## 📌 Overview

This project is a self-contained demonstration of end-to-end ML development, focusing on LEGO brick detection and classification. It combines:

- Object detection (YOLOv8)
- Keypoint detection (studs)
- Custom pattern-based classification
- Rich CLI and Streamlit deployment

It's also a personal milestone: a hands-on transition from hospitality to tech, executed with minimal resources and maximum curiosity.

---

## ⚡ Quick Links

- 🧠 [Technical Utility API Guide](docs/Util_API_guide.md)
- 💻 [Command Line Interface Guide](docs/Lego_CLI_Guide.md)
- 📖 [Project Story & Development Timeline](docs/Storyline_Project_Storytell.md)
- 📊 **[Streamlit DEMO (try it!)](https://placeholder.streamlit.app)**
  - ⚠️ May take up to 1–2 minutes to load (free-tier cold start)
- 🧪 **[Kaggle Project Page (coming soon)](https://kaggle.com/placeholder)**
  - Includes: notebooks for YOLOv8 training, evaluation, and the 3 datasets used.

---

## 🎯 Motivation

This project was born from a single question:

> *"If I can instinctively recognize a LEGO brick, can I teach a machine to do it too?"*

The answer led to the creation of a hybrid detection pipeline, extensive CLI tooling, and a public demo. It also serves as my first full-stack ML project — built from scratch, piece by piece.

---

## 📁 Repository Structure (Key Elements)

```
├── lego_cli.py            # Main CLI interface
├── utils/                 # Core utility modules
│   ├── detection_utils.py     # ML inference logic
│   ├── classification_utils.py # Stud-based dimension inference
│   └── ...
├── docs/                 # Project documentation
├── notebooks/            # YOLOv8 training notebooks (to be uploaded)
├── presentation/         # Hero image, visuals, QR frames
├── results/              # Annotated inference outputs
└── README.md
```

> 🗂️ Full reference in the [CLI Guide](docs/Lego_CLI_Guide.md) and [Utils API Guide](docs/Util_API_guide.md)

---

## 🧠 How It Works (Pipeline Summary)

1. Detect bricks in the image using a YOLOv8 model
2. Crop and analyze individual brick regions
3. Detect studs using a second YOLOv8 model
4. Analyze stud pattern with geometric logic
5. Predict brick dimensions and embed metadata

Includes optional EXIF tagging and visual output framing.

---

## 🧪 Training and Data

Model training and dataset curation were done on **Kaggle Notebooks**, using:
- 3 custom LEGO datasets (bricks, studs, and synthetic composites)
- Data augmentation with Albumentations
- Fine-tuning pretrained YOLOv8n models

📌 *Links to the notebooks and datasets will be added to the Kaggle project page.*

---

## 🖥️ Using the Project

### 🔧 CLI (Recommended)
```bash
# Detect a brick from image
python lego_cli.py infer --image input.jpg --save-annotated

# Run batch processing
python lego_cli.py batch-inference --input folder/ --output results/
```
More in [CLI Guide](docs/Lego_CLI_Guide.md).

### 🖼️ Streamlit App
Try it online: **[Streamlit DEMO](https://placeholder.streamlit.app)**

---

## 🧩 Development Timeline (Summary)

| Phase | Highlights |
|-------|-----------|
| **1. Ideation** | Inspired by childhood LEGO play & engineering instinct |
| **2. Dataset Prep** | 27 brick classes, ~2000 images, manual annotation |
| **3. Detection** | YOLOv8 fine-tuning for bricks and studs |
| **4. Classification** | Custom geometry algorithm based on stud layout |
| **5. CLI Design** | Fully documented with Rich-based UX |
| **6. Demo Deployment** | Streamlit frontend + EXIF metadata tagging |

Full story: [Project Story](docs/Storyline_Project_Storytell.md)

---

## 🌱 Antecedents, Inspiration & Future Aspirations

### Antecedents: Standing on Giants' Shoulders

This project is inspired by several notable LEGO recognition systems developed over the years. From hardware-based solutions like Piqabrick to comprehensive AI systems like Brickognize, each has contributed to advancing the field of LEGO part recognition.

After researching existing solutions, I was particularly drawn to the possibility of combining deep learning with domain-specific knowledge about LEGO bricks. The stud pattern approach was inspired by how humans naturally identify bricks - we often count studs to determine dimensions.

### Comparative Landscape

| Project | Developer | Parts Coverage | Technology | Unique Approach | Scale |
|---------|-----------|----------------|------------|-----------------|-------|
| **Brickssifier** | Miguel Di Lalla (self-taught) | ~50 basic bricks | YOLOv8 + stud keypoints | Stud pattern classification | Individual portfolio project |
| Brickognize | Tramacsoft (company) | ~85,000 parts & sets | Mask R-CNN + image search | Synthetic data generation | Commercial product with team |
| BrickIt | BrickIt App (startup) | Est. thousands | Mobile CNN | Real-time pile analysis | Venture-backed mobile app |
| RebrickNet | Rebrickable (platform) | ~300 parts | CNN object detector | Community-sourced data | Platform feature with user base |

### My Approach & Honest Scale

As a self-taught developer transitioning from hospitality to tech, I designed Brickssifier with two goals:

1. **Learning showcase**: Demonstrate end-to-end ML development skills from data to deployment
2. **Novel contribution**: The stud-detection approach offers high accuracy for basic bricks while requiring less training data

While commercial solutions like Brickognize cover nearly the entire LEGO catalog with teams of engineers, Brickssifier focuses on a more modest but thoroughly executed scope. This deliberate limitation allowed me to implement a complete system with polish and documentation that would be challenging for a solo developer to achieve at larger scale.

### Future Aspirations

This project represents my first stepping stone in ML development, with several planned growth paths:

1. **Technical expansion**: 
   - Extend coverage to non-studded parts (tiles, slopes, Technic)
   - Implement multi-part detection in cluttered scenes
   - Experiment with on-device optimization for mobile deployment

2. **Integration possibilities**:
   - Create plugins for inventory management systems
   - Develop an open API for community applications
   - Explore collaboration with other LEGO ML projects

3. **Knowledge sharing**:
   - Document the complete development journey to help other self-taught developers
   - Contribute to open datasets and benchmarks
   - Package components as reusable libraries

My goal is to continue evolving both this project and my technical skills, using each iteration as a stepping stone toward professional growth in data science and ML engineering.

> *"Every model is a draft of a better model to come."*

---

## ❤️ Thank You for Reading!

This project marks a transition point in my career and a personal victory in learning how to think like an engineer.

> If you enjoyed exploring this project or see potential in my work, don't hesitate to reach out.

🔗 [Visit my website](https://migueldilalla.github.io/)  
💼 [Connect on LinkedIn](https://www.linkedin.com/in/MiguelDiLalla/)  
📨 Or open an issue / star the repo if you'd like to collaborate!

---

© Miguel Di Lalla — LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project.

