import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore

# ========== Configuration ==========
MODEL_PATH = r"E:\CNN_vedic\models\final_gender_model.h5"  
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Male", "Female"]

# ========== Load Model ==========
print("🔍 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ========== Predict Function ==========
def predict_gender(image_path: str) -> str:
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid: {image_path}")
    image = cv2.resize(image, IMG_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # add batch dimension

    # Predict
    preds = model.predict(image)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_class]

    return CLASS_NAMES[predicted_class], float(confidence)

# ========== Run Inference on Sample ==========
if __name__ == "__main__":
    test_image_path = r"E:\CNN_vedic\Data\Task_A\TESTING\unseen\hritik.jpg" 

    try:
        gender, conf = predict_gender(test_image_path)
        print(f"\n🧠 Prediction: {gender} ({conf * 100:.2f}% confidence)")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
