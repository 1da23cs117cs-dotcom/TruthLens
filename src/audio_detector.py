
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"
SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def load_audio_model():
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    return extractor, model


def predict_audio(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    extractor, model = load_audio_model()

    # Convert audio to mono, 16 kHz WAV using FFmpeg.
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "converted.wav"

        subprocess.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-y",
                "-i", str(audio_path),
                "-vn",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-t", "30",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        audio, _ = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            mono=True,
        )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    print("Model ID:", MODEL_ID, flush=True)
    print("Label mapping:", model.config.id2label, flush=True)
    print("Raw logits:", logits[0].tolist(), flush=True)
    print("Probabilities:", probabilities.tolist(), flush=True)  

    index = int(torch.argmax(probabilities))
    label = model.config.id2label[index]
    confidence = round(float(probabilities[index]) * 100, 2)

    return label, confidence
