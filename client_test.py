import base64
import requests

API_URL = "http://127.0.0.1:8000/detect"
API_KEY = "hackathon-key"

with open(r"C:\Users\ADMIN\Desktop\789\wavs\A0064.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

payload = {
    "audio": audio_b64
}

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

r = requests.post(API_URL, json=payload, headers=headers)

print(r.status_code)
print(r.json())
