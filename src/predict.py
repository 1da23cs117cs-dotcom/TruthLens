import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_PATH = Path("models/truthlens_finetuned_resnet18.pth")
IMAGE_SIZE = 224


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print("Model not found. Train it first with:")
        print("python src/train.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = probabilities.argmax().item()

    label = class_names[index]
    confidence = probabilities[index].item() * 100

    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
