from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import os
import librosa
from model import predict
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.environ.get("API_KEY", "hackathon-key")

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether audio is human or AI-generated using a trained ML model.",
    version="1.0"
)

class AudioRequest(BaseModel):
    audio: str  # Base64 encoded audio

@app.post("/detect")
def detect_audio(data: AudioRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        audio_bytes = base64.b64decode(data.audio)
        audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        result, confidence = predict(audio_np)
        return {
            "result": result,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
