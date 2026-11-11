
import cv2
import mediapipe as mp
import numpy as np
import csv
import sys
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Lip landmark points
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def detect_lips(video_path):

    if not os.path.exists(video_path):
        print("Error: Video file not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    output_video = "lip_detection_output.mp4"
    output_csv = "lip_measurements.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_video, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        print("Error: Cannot create output video.")
        cap.release()
        return

    frame_number = 0
    detected_faces = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        with open(output_csv, "w", newline="") as file:

            csv_writer = csv.writer(file)

            csv_writer.writerow([
                "Frame",
                "Time_seconds",
                "Mouth_Opening_Ratio",
                "Face_Detected"
            ])

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = face_mesh.process(rgb_frame)

                ratio = 0.0
                face_detected = 0

                if results.multi_face_landmarks:

                    face_detected = 1
                    detected_faces += 1

                    face_landmarks = results.multi_face_landmarks[0]

                    def get_point(index):
                        landmark = face_landmarks.landmark[index]

                        return (
                            int(landmark.x * width),
                            int(landmark.y * height)
                        )

                    upper = get_point(UPPER_LIP)
                    lower = get_point(LOWER_LIP)
                    left = get_point(LEFT_CORNER)
                    right = get_point(RIGHT_CORNER)

                    mouth_height = distance(upper, lower)
                    mouth_width = distance(left, right)

                    if mouth_width > 0:
                        ratio = mouth_height / mouth_width

                    # Draw lip landmarks
                    cv2.circle(frame, upper, 4, (0, 255, 0), -1)
                    cv2.circle(frame, lower, 4, (0, 255, 0), -1)
                    cv2.circle(frame, left, 4, (255, 0, 0), -1)
                    cv2.circle(frame, right, 4, (255, 0, 0), -1)

                    cv2.line(frame, upper, lower, (0, 255, 255), 2)

                    cv2.putText(
                        frame,
                        f"Mouth Ratio: {ratio:.3f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face Detected",
                        (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "No Face Detected",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                timestamp = frame_number / fps

                csv_writer.writerow([
                    frame_number,
                    round(timestamp, 3),
                    round(ratio, 4),
                    face_detected
                ])

                writer.write(frame)

    cap.release()
    writer.release()

    print("\nLip detection completed!")
    print("Frames processed:", frame_number)
    print("Frames with detected face:", detected_faces)
    print("Output video:", output_video)
    print("Measurements:", output_csv)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python lip_detection.py input_video.mp4")
    else:
        detect_lips(sys.argv[1])
