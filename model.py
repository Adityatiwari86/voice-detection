import torch
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
feature_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
feature_model.eval()

for p in feature_model.parameters():
    p.requires_grad = False

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),   # ✅ MUST MATCH TRAINING
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

classifier = Classifier()
classifier.load_state_dict(
    torch.load("voice_detector.pth", map_location="cpu")
)
classifier.eval()

def predict(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        emb = feature_model(**inputs).last_hidden_state.mean(dim=1)
        score = classifier(emb).item()

    if score > 0.5:
        return "AI_GENERATED", round(score, 2)
    else:
        return "HUMAN", round(1 - score, 2)
