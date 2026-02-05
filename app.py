from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import soundfile as sf
from model import predict

API_KEY = "hackathon-key"
app = FastAPI()

class AudioRequest(BaseModel):
    audio: str

@app.post("/detect")
def detect_audio(data: AudioRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Decode base64 → bytes
        audio_bytes = base64.b64decode(data.audio)

        # Read audio directly from memory (NO FILE)
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Resample if needed
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        result, confidence = predict(audio_np)

        return {
            "result": result,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
