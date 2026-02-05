import os
import torch
from torch import nn
from feature_extraction import extract_features

X, y = [], []

print("🔹 Loading human data...")
for f in os.listdir("dataset/human"):
    X.append(extract_features(f"dataset/human/{f}"))
    y.append(0)

print("🔹 Loading AI data...")
for f in os.listdir("dataset/ai"):
    X.append(extract_features(f"dataset/ai/{f}"))
    y.append(1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = Classifier()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 Training started...")
for epoch in range(20):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "voice_detector.pth")
print("✅ Model saved as voice_detector.pth")
