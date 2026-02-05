import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

for p in model.parameters():
    p.requires_grad = False

def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)

    return emb.squeeze().numpy()
