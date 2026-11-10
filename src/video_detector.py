import sys
from pathlib import Path

import torch


# ==================================================
# Locate the AEGIS video detection project
# ==================================================

CURRENT_FILE = Path(__file__).resolve()

# CURRENT_FILE:
# D:\TruthLens-main\TruthLens-main\src\video_detector.py
#
# parents[0] -> src
# parents[1] -> TruthLens-main
# parents[2] -> D:\TruthLens-main

OUTER_PROJECT_DIR = CURRENT_FILE.parents[2]

AEGIS_BRANCHES_DIR = (
    OUTER_PROJECT_DIR
    / "aegis"
    / "src"
    / "branches"
)

AEGIS_CHECKPOINT = (
    OUTER_PROJECT_DIR
    / "aegis"
    / "checkpoints"
    / "phase2"
    / "checkpoint_best.pt"
)


# Add AEGIS folders to Python import path
if str(AEGIS_BRANCHES_DIR) not in sys.path:
    sys.path.insert(0, str(AEGIS_BRANCHES_DIR))


# ==================================================
# Import the working AEGIS detector
# ==================================================

from inference import load_model, analyze_video


# ==================================================
# Global model variables
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

video_model = None


# ==================================================
# Load model only once
# ==================================================

def load_video_model():
    global video_model

    if video_model is not None:
        return video_model

    if not AEGIS_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Video detection checkpoint not found at:\n"
            f"{AEGIS_CHECKPOINT}"
        )

    print("\nLoading TruthLens video detection model...")
    print(f"Checkpoint: {AEGIS_CHECKPOINT}")
    print(f"Device: {device}")

    video_model = load_model(
        str(AEGIS_CHECKPOINT),
        device,
    )

    print("Video detection model loaded successfully.\n")

    return video_model


# ==================================================
# Predict video
# ==================================================

def predict_video(video_path):
    """
    Analyze a video using the AEGIS deepfake detector.

    Returns:
        prediction: REAL or FAKE
        confidence: percentage
        details: dictionary containing branch results
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    model = load_video_model()

    try:
        result, outputs = analyze_video(
            str(video_path),
            model,
            device,
        )

        ai_probability = result["ai_probability"]

        # Same decision threshold used during testing
        prediction = (
            "FAKE"
            if ai_probability > 0.5
            else "REAL"
        )

        # Confidence represents probability of the predicted class
        if prediction == "FAKE":
            confidence = ai_probability * 100
        else:
            confidence = (1 - ai_probability) * 100

        confidence = round(confidence, 2)

        details = {
            "ai_probability": round(
                ai_probability * 100,
                2,
            ),
            "pixel_probability": round(
                result["pixel_prob"] * 100,
                2,
            ),
            "motion_probability": round(
                result["motion_prob"] * 100,
                2,
            ),
            "disagreement": round(
                result["disagreement"] * 100,
                2,
            ),
            "smoothness": round(
                result["smoothness"] * 100,
                2,
            ),
            "processing_time": round(
                result["elapsed_sec"],
                2,
            ),
        }

        print("\n========== VIDEO RESULT ==========")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence}%")
        print(f"AI probability: {details['ai_probability']}%")
        print(f"Pixel probability: {details['pixel_probability']}%")
        print(f"Motion probability: {details['motion_probability']}%")
        print(f"Processing time: {details['processing_time']} sec")
        print("==================================\n")

        return prediction, confidence, details

    except Exception as e:
        raise RuntimeError(
            f"Video prediction failed: {str(e)}"
        )