
from pathlib import Path
import random
import shutil

SOURCE = Path.home() / "Downloads" / "archive"
DEST = Path("dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]


def main():
    # Verify the source folders first.
    for split in ("train", "test"):
        for label in ("FAKE", "REAL"):
            folder = SOURCE / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")

    # Copy test images only if they are not already present.
    for label in ("FAKE", "REAL"):
        files = get_images(SOURCE / "test" / label)
        if not files:
            raise RuntimeError(f"No images in test/{label}")

        target = DEST / "test" / label.lower()
        target.mkdir(parents=True, exist_ok=True)

        for file in files:
            destination = target / file.name
            if not destination.exists():
                shutil.copyfile(file, destination)

        print(f"Test {label}: {len(files)} source images", flush=True)

    # Split training images into train and validation.
    for label in ("FAKE", "REAL"):
        files = get_images(SOURCE / "train" / label)
        if len(files) < 2:
            raise RuntimeError(f"Not enough images in train/{label}")

        random.shuffle(files)
        val_count = max(1, int(len(files) * 0.1))

        for split, selected in (
            ("val", files[:val_count]),
            ("train", files[val_count:]),
        ):
            target = DEST / split / label.lower()
            target.mkdir(parents=True, exist_ok=True)

            for file in selected:
                destination = target / file.name
                if not destination.exists():
                    shutil.copyfile(file, destination)

            print(
                f"{split}/{label.lower()}: {len(selected)} source images",
                flush=True,
            )

    print("\nDataset preparation finished.", flush=True)


if __name__ == "__main__":
    main()
