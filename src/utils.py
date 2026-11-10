from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def check_dataset(root="dataset"):
    root = Path(root)
    required = [
        root / "train" / "real",
        root / "train" / "fake",
        root / "val" / "real",
        root / "val" / "fake",
        root / "test" / "real",
        root / "test" / "fake",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset folders:\n" + "\n".join(missing)
        )

    counts = {}
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            folder = root / split / label
            counts[f"{split}/{label}"] = sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    return counts
