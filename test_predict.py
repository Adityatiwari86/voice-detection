import librosa
from model import predict

# Path to your WAV file
audio_path = r"C:\Users\ADMIN\Desktop\789\wavs\A0064.wav"

# Load audio at 16kHz
audio, sr = librosa.load(audio_path, sr=16000)

# Run prediction
result, confidence = predict(audio)
print(result, confidence)
