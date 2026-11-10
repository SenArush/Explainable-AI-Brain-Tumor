from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import BrainTumorCNN, CLASS_NAMES

DATA_DIR = Path("data")
BATCH_SIZE = 32
EPOCHS = 8
IMAGE_SIZE = 224

def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Dataset folders missing. Create data/train and data/val, "
            "each containing one subfolder per class. See README.md."
        )

    tfms = {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
    }
    datasets_by_split = {
        split: datasets.ImageFolder(DATA_DIR / split, transform=tfms[split])
        for split in ("train", "val")
    }
    for split, ds in datasets_by_split.items():
        if ds.classes != CLASS_NAMES:
            raise SystemExit(
                f"{split} class folders must be alphabetically named exactly: {CLASS_NAMES}. "
                f"Found: {ds.classes}"
            )
    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"), num_workers=0)
        for split, ds in datasets_by_split.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorCNN(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in loaders[phase]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            print(
                f"Epoch {epoch+1}/{EPOCHS} | {phase} "
                f"loss={running_loss/max(total,1):.4f} "
                f"accuracy={correct/max(total,1):.3f}"
            )

    Path("models").mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), "models/brain_tumor_cnn.pth")
    print("Saved model to models/brain_tumor_cnn.pth")

if __name__ == "__main__":
    main()
