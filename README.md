# Explainable AI for Brain Tumor Detection — Mini Project

A small educational project that classifies brain MRI images into four image-folder classes and uses Grad-CAM to visualize image regions that influenced a prediction.

> **Important:** This is a learning prototype, not a medical device. It has not been clinically validated. Do not use it to diagnose, rule out, or treat any medical condition.

## Features

- Four-class MRI image classification: `glioma`, `meningioma`, `no_tumor`, `pituitary`
- A compact CNN built with PyTorch
- Grad-CAM heatmap for a visual explanation
- Streamlit interface for image upload and results
- Training and validation accuracy/loss printed each epoch

## Project structure

```text
explainable_ai_brain_tumor_mini_project/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── gradcam.py
├── data/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   └── val/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
└── models/
    └── brain_tumor_cnn.pth  (created after training)
```

## 1. Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset

Use a brain MRI dataset that you are permitted to use. Arrange images into the following folders. Keep the folder names exactly as shown because `ImageFolder` assigns labels alphabetically.

```text
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── val/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

Put training images in `train` and held-out validation images in `val`. Do not put the same image or near-duplicate images in both splits. Check the dataset's license and documentation before using it.

## 3. Train the model

```bash
python train.py
```

The script trains a small CNN for eight epochs and saves its weights to `models/brain_tumor_cnn.pth`. You can change `EPOCHS` and `BATCH_SIZE` in `train.py`. Results depend strongly on dataset quality, size, and split.

## 4. Launch the app

```bash
streamlit run app.py
```

Upload a JPG or PNG MRI image. The app displays class probabilities and a Grad-CAM overlay.

## How it works

1. **Preprocessing:** resize the image to 224 × 224, convert it to a tensor, and normalize it.
2. **Classification:** a CNN learns image features and outputs four class scores.
3. **Probability:** softmax converts the scores into normalized class probabilities.
4. **Explainability:** Grad-CAM uses gradients from the last convolutional layer to create a heatmap showing regions that contributed to the selected class score.

## Suggested evaluation for a college report

Report validation accuracy, per-class precision/recall/F1, and a confusion matrix. Evaluate on a separate test set if available. Do not report training accuracy as model performance on unseen data. Discuss class imbalance, image-source bias, data leakage, and the limits of Grad-CAM.

## Limitations

- A small CNN may perform poorly without sufficient, balanced, correctly labelled data.
- Image-level classification does not segment or outline a tumor.
- Grad-CAM is an approximate explanation, not a causal explanation or clinical localization.
- Confidence scores are not calibrated probabilities of disease.
- MRI interpretation requires qualified medical professionals.
