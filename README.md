# Dental Fracture Detection using Deep Learning

A deep learning and computer vision system for detecting mandibular fractures from dental X-ray images.

The project explores image classification, transfer learning, hybrid machine learning, object detection, model explainability, and web-based deployment as part of an end-to-end medical image analysis pipeline.

> **Note:** This is an academic/research prototype and is not intended for clinical diagnosis or medical decision-making.

---

## Overview

Dental fractures can be difficult to identify from X-ray images because fracture regions may be small, subtle, and affected by image quality.

This project investigates whether deep learning models can automatically distinguish between:

- **Fractured X-rays**
- **Non-fractured X-rays**

The system was developed through multiple stages, starting with a custom CNN and progressing to transfer learning, hybrid machine learning, fracture localization, and explainability.

---

## System Pipeline

```text
                  Dental X-ray Image
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
              Data Augmentation
                         │
                         ▼
              Deep Learning Model
                         │
              ┌──────────┴──────────┐
              │                     │
          Fracture              No Fracture
              │
              ▼
        Feature Extraction
              │
              ▼
     Hybrid ML Classification
              │
              ▼
      Fracture Localization
              │
              ▼
       YOLOv8 / DETR
              │
              ▼
       Model Explainability
       Grad-CAM / Eigen-CAM