
from pathlib import Path
from torchvision import datasets

root = Path.home() / "Downloads" / "archive"

train = datasets.ImageFolder(root / "train")
test = datasets.ImageFolder(root / "test")

print("Classes:", train.classes)
print("Training images:", len(train))
print("Test images:", len(test))
