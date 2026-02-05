# AI Generated Voice Detection API

## Problem
Detect AI-generated speech from human speech.

## Solution
We use wav2vec2 embeddings and a trained neural classifier to detect synthetic voices.

## Architecture
Base64 Audio → wav2vec2 → Trained Classifier → API Response

## Endpoint
POST /detect

Headers:
x-api-key: hackathon-key

Body:
{
  "audio": "<base64>"
}

## Output
{
  "result": "AI_GENERATED | HUMAN",
  "confidence": 0-1
}

## Model Training
- Human: Mozilla Common Voice
- AI: Coqui / Google TTS
- Feature extractor: wav2vec2-base
- Classifier: MLP (trained)
