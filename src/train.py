
from pathlib import Path
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Dataset paths
SOURCE_DIR = Path.home() / "Downloads" / "archive"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "truthlens_finetuned_resnet18.pth"

# Training settings
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
IMAGE_SIZE = 224
SEED = 42
VAL_RATIO = 0.10


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SOURCE_DIR / "train"
    test_path = SOURCE_DIR / "test"

    for folder in [
        train_path / "FAKE",
        train_path / "REAL",
        test_path / "FAKE",
        test_path / "REAL",
    ]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load ALL training images.
    # Two dataset objects let us use separate transforms
    # for training and validation.
    train_aug = datasets.ImageFolder(
        train_path, transform=train_transform
    )
    train_eval = datasets.ImageFolder(
        train_path, transform=eval_transform
    )
    test_ds = datasets.ImageFolder(
        test_path, transform=eval_transform
    )

    if train_aug.classes != ["FAKE", "REAL"]:
        raise ValueError(
            f"Unexpected class names: {train_aug.classes}"
        )

    if test_ds.classes != train_aug.classes:
        raise ValueError("Train/test classes do not match.")

    # Reproducible random split; validation uses the same
    # source-file indices but evaluation transforms.
    generator = torch.Generator().manual_seed(SEED)
    total = len(train_aug)
    val_count = max(1, int(total * VAL_RATIO))
    train_count = total - val_count

    train_ds, _ = random_split(
        train_aug,
        [train_count, val_count],
        generator=generator,
    )

    _, val_indices = random_split(
        range(total),
        [train_count, val_count],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Apply identical indices to the evaluation-transform dataset.
    val_ds = torch.utils.data.Subset(
        train_eval, val_indices.indices
    )

    print("Classes:", train_aug.classes)
    print("Total training images:", total)
    print("Images used for training:", len(train_ds))
    print("Validation images:", len(val_ds))
    print("Test images:", len(test_ds))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # num_workers=0 is compatible with Windows.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Loading pretrained ResNet18...")
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Fine-tune the final residual block and classifier.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_acc = -1.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total_seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            running_loss += loss.item() * batch_count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_seen += batch_count

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / max(total_seen, 1)
        train_acc = correct / max(total_seen, 1)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
            f" | Loss: {train_loss:.4f}"
            f" | Train accuracy: {train_acc:.2%}"
            f" | Validation accuracy: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_aug.classes,
                },
                MODEL_PATH,
            )

            print(f"Best model saved to: {MODEL_PATH}")

    # Evaluate the best checkpoint once on the test set.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc = evaluate(model, test_loader, device)

    print("\nFinal evaluation")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Model saved at: {MODEL_PATH}")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc="Evaluating", leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


if __name__ == "__main__":
    main()