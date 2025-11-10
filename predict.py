# predict.py — drop in folder, run: python predict.py your_slide.jpg
import torch, torch.nn as nn, numpy as np, sys, matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

class LiverResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 7, 2, 3, bias=False),
            base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool
        )
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, 4))
    def forward(self, x): return self.head(self.features(x).flatten(1))

def preprocess(path):
    img = Image.open(path).convert('L').resize((512, 512))
    arr = np.array(img, dtype=np.float32) / 255.0
    gx = np.zeros_like(arr); gx[:,1:-1] = (arr[:,2:] - arr[:,:-2]) / 2.0
    gy = np.zeros_like(arr); gy[1:-1,:] = (arr[2:,:] - arr[:-2,:]) / 2.0
    edges = np.hypot(gx, gy)
    x = torch.from_numpy(np.stack([arr, edges])).unsqueeze(0)
    return transforms.Normalize([0.5, 0.5], [0.5, 0.5])(x)

if len(sys.argv) != 2:
    print("Usage: python predict.py <image.jpg>")
    sys.exit(1)

model = LiverResNet50()
model.load_state_dict(torch.load('SuperfineMAE.pth', map_location='cpu'))
model.eval()

x = preprocess(sys.argv[1])
with torch.no_grad():
    scores = model(x).squeeze().numpy()

print("\n" + "="*60)
print("   LIVER-JPG-AI")
print("="*60)
for name, s in zip(["STEATOSIS", "FIBROSIS", "CIRRHOSIS", "INFLAMMATION"], scores):
    grade = ["NONE", "MILD", "MODERATE", "SEVERE"][int(s)]
    print(f"{name:<12}: {s:.3f} → {grade}")
print("="*60)
print(f"OVERALL RISK: {scores.mean():.3f}/3.0")
print("="*60)
