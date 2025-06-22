import os
import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# ========== Configuration ==========
MODEL_PATH = 'models/gender_detection.h5'
IMG_SIZE = 250
NUM_SAMPLES = 20  # per class
SEED = 42
random.seed(SEED)

TEST_PATHS = {
    'male': 'Data/Task_A/TESTING/men',
    'female': 'Data/Task_A/TESTING/women'
}

# ========== Load Model ==========
model = load_model(MODEL_PATH)

# ========== Load and Preprocess Samples ==========
data = []
labels = []

label_map = {'male': 0, 'female': 1}

for gender, path in TEST_PATHS.items():
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
    selected = random.sample(files, min(NUM_SAMPLES, len(files)))
    
    for img_path in selected:
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure consistent color ordering
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label_map[gender])

data = np.array(data)
labels = np.array(labels)

# ========== Predict ==========
preds = model.predict(data, batch_size=32, verbose=0)
preds_class = np.argmax(preds, axis=1)

# ========== Evaluate ==========
print("\n📊 Classification Report:")
print(classification_report(labels, preds_class, target_names=["Male", "Female"]))

print("✅ Accuracy:  {:.4f}".format(accuracy_score(labels, preds_class)))
print("🎯 F1 Score:  {:.4f}".format(f1_score(labels, preds_class)))
print("📈 Recall:    {:.4f}".format(recall_score(labels, preds_class)))
print("🔎 Precision: {:.4f}".format(precision_score(labels, preds_class)))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(labels, preds_class))
