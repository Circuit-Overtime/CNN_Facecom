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
NUM_EPOCHS = 40
IMG_SIZE = 96
LR = 0.001
PATIENCE = 7
MODEL_SAVE_PATH = 'models/final_gender_classifier_balanced.pt'

# ========== Transformations ==========
common_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mild_female_aug = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== Custom Dataset ==========
class AugmentedGenderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, base_transform, female_transform):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.base_transform = base_transform
        self.female_transform = female_transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if label == 1:  # Female
            img = self.female_transform(img)
        else:
            img = self.base_transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

# ========== Dataset ==========
train_dataset = AugmentedGenderDataset(TRAIN_DIR, common_transform, mild_female_aug)
val_dataset = AugmentedGenderDataset(VAL_DIR, common_transform, mild_female_aug)
combined_dataset = ConcatDataset([train_dataset, val_dataset])

# ========== Handle Class Imbalance ==========
labels = []
for dataset in combined_dataset.datasets:
    labels.extend([label for _, label in dataset])

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
full_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# ========== Model ==========
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.base(x)

model = GenderClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ========== Loss, Optimizer, Scheduler ==========
pos_weight = torch.tensor([1.3]).to(device)  # Controlled bias
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

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
plt.savefig("final_training_loss_balanced.png")
plt.show()
