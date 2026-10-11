
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SOURCE_DIR = Path.home() / "Downloads" / "archive"
MODEL_PATH = Path("models/truthlens_pilot_resnet18.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_IMAGES = 400


def main():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    dataset = datasets.ImageFolder(
        SOURCE_DIR / "test",
        transform=transform,
    )

    # Select up to 200 images from each class.
    selected = []
    for class_index in range(len(dataset.classes)):
        class_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_index
        ]
        selected.extend(class_indices[:MAX_IMAGES // 2])

    if not selected:
        raise RuntimeError("No test images found.")

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = [0] * len(dataset.classes)
    total = [0] * len(dataset.classes)

    print(f"Evaluating {len(selected)} images on CPU...", flush=True)

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, start=1):
            predictions = model(images).argmax(dim=1)

            for label, prediction in zip(labels, predictions):
                label = label.item()
                total[label] += 1
                if prediction.item() == label:
                    correct[label] += 1

            print(
                f"Processed {min(batch_number * BATCH_SIZE, len(selected))}"
                f"/{len(selected)} images",
                flush=True,
            )

    for i, name in enumerate(dataset.classes):
        accuracy = 100 * correct[i] / total[i] if total[i] else 0
        print(
            f"{name}: {correct[i]}/{total[i]} correct "
            f"({accuracy:.2f}%)"
        )

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
