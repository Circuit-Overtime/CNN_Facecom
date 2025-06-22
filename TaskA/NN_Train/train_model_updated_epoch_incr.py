import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== Configuration ==========
DATA_DIR = 'Data/Task_A'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 96
LR = 0.001
MODEL_SAVE_PATH = 'models/gender_classifier.pt'

# ========== Transformations ==========
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== Dataset ==========
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

# ========== Handle Class Imbalance ==========
labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.FloatTensor(class_weights)

sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== Model ==========
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Grayscale
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)

model = GenderClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
model.to(device)

# ========== Loss, Optimizer, Scheduler ==========
criterion = nn.BCELoss(weight=class_weights_tensor[1].to(device))
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

# ========== Evaluation Function ==========
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            preds = (outputs >= 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, precision, recall, f1

# ========== Training ==========
best_val_acc = 0
history = {"loss": [], "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": []}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc, prec, rec, f1 = evaluate(model, val_loader)
    scheduler.step(f1)  # Step scheduler on F1 score

    history["loss"].append(running_loss)
    history["val_acc"].append(acc)
    history["val_prec"].append(prec)
    history["val_rec"].append(rec)
    history["val_f1"].append(f1)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    if acc > best_val_acc:
        best_val_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Saved best model")

print("Training complete. Best validation accuracy:", best_val_acc)

# ========== Plotting Metrics ==========
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs, history["val_acc"], label="Accuracy")
plt.plot(epochs, history["val_prec"], label="Precision")
plt.plot(epochs, history["val_rec"], label="Recall")
plt.plot(epochs, history["val_f1"], label="F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gender_classifier_metrics.png")
plt.show()
