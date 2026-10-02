# Dental Fracture Detection using Deep Learning

A deep learning and computer vision system for detecting and localizing mandibular fractures from dental X-ray images.

The project explores binary image classification, transfer learning, hybrid machine learning, fracture localization, explainable AI, and web-based model deployment.

> **Note:** This is an academic/research prototype and is not intended for clinical diagnosis or medical decision-making.

---

## Overview

The goal of this project is to develop a computer vision pipeline capable of identifying fractures in dental X-ray images and localizing suspected fracture regions.

The project was developed through multiple stages, beginning with a custom CNN and progressing through transfer learning, CNN feature extraction with traditional machine learning classifiers, object detection, explainability, and web deployment.

The final system uses a two-stage deep learning pipeline:

1. **EfficientNetV2-S** for fracture classification
2. **YOLOv8** for fracture localization

Eigen-CAM is also used to visualize feature activations from the classification model.

---

## Application

![DentalScan Application](images/dental-fracture-detection-full.png)

The trained models were integrated into a Flask-based web application that allows users to upload dental X-ray images and perform model inference.

---

## Final System Pipeline

The final inference system follows a two-stage classification and localization approach.

```text
Dental X-ray
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNetV2-S
     │
     ├───────────────┐
     ▼               ▼
  Normal          Fracture
     │               │
     ▼               ▼
  Normal          YOLOv8
                     │
                     ▼
             Fracture Localization
                     │
                     ▼
                 Eigen-CAM
                     │
                     ▼
              Web-based Inference