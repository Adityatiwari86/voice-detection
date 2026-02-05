from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import soundfile as sf
import os
import librosa
from model import predict
from dotenv import load_dotenv

# Load environment variables from .env (optional)
load_dotenv()
API_KEY = os.environ.get("API_KEY", "hackathon-key")  # fallback key

# Initialize FastAPI
app = FastAPI(title="AI Voice Detection API",
              description="Detects whether audio is human or AI-generated using a trained ML model.",
              version="1.0")

# Request schema
class AudioRequest(BaseModel):
    audio: str  # Base64 encoded audio

@app.post("/detect")
def detect_audio(data: AudioRequest, x_api_key: str = Header(...)):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # 1️⃣ Decode base64 to bytes
        audio_bytes = base64.b64decode(data.audio)

        # 2️⃣ Read audio from bytes (support WAV/FLAC/other sound formats)
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))

        # 3️⃣ Convert stereo → mono if needed
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # 4️⃣ Resample to 16kHz (required by wav2vec2)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        # 5️⃣ Predict using the trained model
        result, confidence = predict(audio_np)

        # 6️⃣ Return JSON
        return {
            "result": result,
            "confidence": confidence
        }

    except Exception as e:
        # Catch all exceptions and return 500
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
