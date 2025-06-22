import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== Configuration ==========
DATA_DIR = 'Data/Task_A'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
BATCH_SIZE = 32
NUM_EPOCHS = 45
IMG_SIZE = 96
LR = 0.001
PATIENCE = 5
MODEL_SAVE_PATH = 'models/final_gender_classifier.pt'

# ========== Transformations ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== Dataset ==========
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
combined_dataset = ConcatDataset([train_dataset, val_dataset])

# ========== Handle Class Imbalance ==========
labels = [label for _, label in train_dataset] + [label for _, label in val_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.FloatTensor(class_weights)

sample_weights = [class_weights[label] for _, label in combined_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
full_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# ========== Model ==========
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.base(x)

model = GenderClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ========== Loss, Optimizer, Scheduler ==========
criterion = nn.BCELoss(weight=class_weights_tensor[1].to(device))
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# ========== Training ==========
best_loss = float('inf')
no_improve_epochs = 0
loss_history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(full_loader, desc=f"Final Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(full_loader)
    loss_history.append(avg_loss)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Saved best model")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

print("🎯 Final training complete. Best avg loss:", best_loss)

# ========== Plot ==========
plt.plot(loss_history, label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("final_training_loss.png")
plt.show()
