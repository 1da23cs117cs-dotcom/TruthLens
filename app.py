
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


# --------------------------------------------------
# 1. Flask application configuration
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "truthlens_finetuned_resnet18.pth"
)

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# 2. Device configuration
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None
class_names = None


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model():
    global model, class_names

    if model is not None:
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model_instance = models.resnet18(weights=None)

    model_instance.fc = nn.Linear(
        model_instance.fc.in_features,
        len(class_names),
    )

    model_instance.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model_instance = model_instance.to(device)
    model_instance.eval()

    model = model_instance

    print("TruthLens model loaded successfully.")
    print("Classes:", class_names)
    print("Device:", device)


# --------------------------------------------------
# 4. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 5. Predict whether an image is REAL or FAKE
# --------------------------------------------------

def predict_image(image_path):
    load_model()

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = probabilities.argmax().item()

    prediction = class_names[predicted_index]
    confidence = round(
        probabilities[predicted_index].item() * 100,
        2,
    )

    return prediction, confidence


# --------------------------------------------------
# 6. Home page and image upload
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please select an image to analyze."

        elif (
            Path(uploaded_file.filename).suffix.lower()
            not in ALLOWED_EXTENSIONS
        ):
            error = (
                "Unsupported image format. "
                "Upload a JPG, JPEG, PNG, or WEBP image."
            )

        else:
            filename = secure_filename(uploaded_file.filename)

            if not filename:
                error = "Invalid filename. Please try another image."

            else:
                save_path = UPLOAD_DIR / filename

                try:
                    uploaded_file.save(save_path)

                    prediction, confidence = predict_image(save_path)

                    image_url = (
                        f"/static/uploads/{filename}"
                    )

                except Exception:
                    app.logger.exception(
                        "Image prediction failed."
                    )

                    error = (
                        "Unable to analyze this image. "
                        "Check that the trained model exists "
                        "and the uploaded file is a valid image."
                    )

                    if save_path.exists():
                        save_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error,
    )


# --------------------------------------------------
# 7. Handle oversized uploads
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        image_url=None,
        error="Image is too large. Maximum upload size is 10 MB.",
    ), 413


# --------------------------------------------------
# 8. Start the Flask server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
