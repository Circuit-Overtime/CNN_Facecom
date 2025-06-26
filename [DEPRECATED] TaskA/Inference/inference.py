import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


MODEL_PATH = 'models/final_gender_classifier_balanced.pt'
IMG_SIZE = 96
NUM_SAMPLES = 12  
DATA_PATHS = {
    'Male': ['Data/Task_A/train/male', 'Data/Task_A/val/male'],
    'Female': ['Data/Task_A/train/female', 'Data/Task_A/val/female']
}
SEED = 56
random.seed(SEED)
y_true = []
y_pred = []

# ========== Transform ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== Model ==========
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Linear(in_features, 1)  # No sigmoid here

    def forward(self, x):
        return self.base(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenderClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ========== Sample Images ==========
samples = []
for label, paths in DATA_PATHS.items():
    all_files = []
    for path in paths:
        all_files += [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
    selected = random.sample(all_files, NUM_SAMPLES)
    samples.extend([(img_path, label) for img_path in selected])
random.shuffle(samples)

# ========== Run Inference ==========
rows = 2
cols = NUM_SAMPLES * 2
plt.figure(figsize=(16, 6))

for idx, (img_path, true_label) in enumerate(samples):
    image = Image.open(img_path)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        pred_label = 'Male' if prob >= 0.4 else 'Female'

    y_true.append(1 if true_label == 'Male' else 0)
    y_pred.append(1 if pred_label == 'Male' else 0)

    plt.subplot(rows, cols, idx + 1)
    plt.imshow(image.convert('L'), cmap='gray')
    plt.title(f"GT: {true_label}\nPred: {pred_label}", color='green' if pred_label == true_label else 'red')
    plt.axis('off')

# ========== Print Metrics ==========
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n📊 Inference Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

plt.tight_layout()
plt.savefig("test_predictions_grid.png")
plt.show()
