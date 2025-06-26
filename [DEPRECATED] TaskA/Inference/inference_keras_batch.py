import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from glob import glob

# ========== Configuration ==========
model_path = "models/gender_classifier_final.h5"
image_size = (250, 250)
test_dirs = {
    'Male': r'Data/Task_A/TESTING/men',
    'Female': r'Data/Task_A/TESTING/women'
}

# ========== Load model ==========
model = load_model(model_path)
labels = {0: "Male", 1: "Female"}

def predict_gender(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = cv2.resize(image, image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image, verbose=0)
    predicted_class = np.argmax(preds, axis=1)[0]
    return labels[predicted_class]

# ========== Evaluate on 10 images per class ==========
total = 0
correct = 0

for true_label in test_dirs:
    image_paths = glob(os.path.join(test_dirs[true_label], '*'))[:-50]  # Take first 10
    for path in image_paths:
        predicted_label = predict_gender(path)
        match = (predicted_label == true_label)
        print(f"[{'✔' if match else '✘'}] {os.path.basename(path)} | True: {true_label} | Predicted: {predicted_label}")
        total += 1
        if match:
            correct += 1

accuracy = correct / total * 100
print(f"\n✅ Accuracy on 100 test images (50 male + 50 female): {accuracy:.2f}%")
