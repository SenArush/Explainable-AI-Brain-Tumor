
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.model import BrainTumorCNN, CLASS_NAMES


# 1. Configuration
DATA_DIR = Path("data")
MODEL_PATH = Path("models/brain_tumor_cnn.pth")
IMAGE_SIZE = 224
BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# 2. Load validation dataset
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", val_dataset.classes)
print("Total validation images:", len(val_dataset))


# 3. Load trained model
model = BrainTumorCNN(
    num_classes=len(CLASS_NAMES)
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully!")


# 4. Make predictions
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# 5. Calculate accuracy
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nValidation Accuracy:", round(accuracy * 100, 2), "%")


# 6. Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# 7. Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Brain Tumor Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion matrix saved successfully!")
