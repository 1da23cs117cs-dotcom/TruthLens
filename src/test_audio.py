
from audio_detector import predict_audio

audio_path = input("Enter WAV file path: ").strip().strip('"')

prediction, confidence = predict_audio(audio_path)

print("Prediction:", prediction)
print("Model score:", confidence, "%")
