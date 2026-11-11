
import cv2
import numpy as np
import librosa
import csv
import os
import sys


def analyze_sync(video_path):

    if not os.path.exists(video_path):
        print("Video file not found!")
        return

    print("\nExtracting audio information...")

    # Load audio
    audio, sr = librosa.load(
        video_path,
        sr=16000,
        mono=True
    )

    # Calculate audio energy
    hop_length = 320

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=hop_length
    )[0]

    audio_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    # Read lip measurements
    csv_file = "lip_measurements.csv"

    if not os.path.exists(csv_file):
        print("Lip measurements CSV not found!")
        return

    lip_times = []
    lip_values = []

    with open(csv_file, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            if int(row["Face_Detected"]) == 1:

                lip_times.append(
                    float(row["Time_seconds"])
                )

                lip_values.append(
                    float(row["Mouth_Opening_Ratio"])
                )

    if len(lip_values) < 10:
        print("Not enough detected lip measurements!")
        return

    lip_times = np.array(lip_times)
    lip_values = np.array(lip_values)

    # Common timeline
    duration = min(
        audio_times[-1],
        lip_times[-1]
    )

    common_times = np.arange(
        0,
        duration,
        0.02
    )

    if len(common_times) < 20:
        print("Video is too short for analysis!")
        return

    # Interpolate audio and lip signals
    audio_signal = np.interp(
        common_times,
        audio_times,
        rms
    )

    lip_signal = np.interp(
        common_times,
        lip_times,
        lip_values
    )

    # Smooth both signals
    kernel = np.ones(3) / 3

    audio_signal = np.convolve(
        audio_signal, kernel, mode="same"
    )

    lip_signal = np.convolve(
        lip_signal, kernel, mode="same"
    )

    # Normalize signals
    def normalize(signal):

        std = np.std(signal)

        if std < 1e-8:
            return None

        return (signal - np.mean(signal)) / std

    audio_signal = normalize(audio_signal)
    lip_signal = normalize(lip_signal)

    if audio_signal is None or lip_signal is None:
        print("Insufficient signal variation!")
        return

    # Compare signals at different time delays
    max_lag = int(1.5 / 0.02)

    best_score = -2
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):

        if lag > 0:
            audio_part = audio_signal[:-lag]
            lip_part = lip_signal[lag:]

        elif lag < 0:
            audio_part = audio_signal[-lag:]
            lip_part = lip_signal[:lag]

        else:
            audio_part = audio_signal
            lip_part = lip_signal

        if len(audio_part) < 20:
            continue

        score = np.corrcoef(
            audio_part,
            lip_part
        )[0, 1]

        if np.isfinite(score) and score > best_score:

            best_score = score
            best_lag = lag

    # Display results
    delay = best_lag * 0.02

    print("\n========== IMPROVED LIP SYNC ANALYSIS ==========")

    print("Video duration:", round(duration, 2), "seconds")

    print("Best correlation:", round(best_score, 3))

    print("Estimated lip delay:", round(delay, 2), "seconds")

    if best_score >= 0.6:
        print("Result: Strong signal relationship")

    elif best_score >= 0.3:
        print("Result: Moderate signal relationship")

    else:
        print("Result: Weak signal relationship")

    print("================================================")

    print("\nAnalysis completed!")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python sync_analysis.py sample_video.mp4")

    else:

        analyze_sync(sys.argv[1])
