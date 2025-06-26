import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# ========== Configuration ==========
MODEL_PATH = "models/final_vgg19_gender_model.h5"
IMAGE_PATH = "Data/Task_A/TESTING/unseen/pichai1.png"  # Replace with your test image path
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Male", "Female"]  # Class label map

# ========== Load Model ==========
print("🔍 Loading model...")
model = load_model(MODEL_PATH)

# ========== Preprocess Image ==========
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ========== Predict Function ==========
def predict_gender(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0]  # Softmax output: [prob_male, prob_female]
    predicted_class = int(np.argmax(prediction))
    label = CLASS_NAMES[predicted_class]
    confidence = float(prediction[predicted_class])
    
    # Additional breakdown
    prob_male = prediction[0]
    prob_female = prediction[1]

    print(f"\n🧠 Predicted Gender: {label}")
    print(f"🔢 Confidence: {confidence * 100:.2f}%")
    print(f"📊 Probabilities: Male: {prob_male * 100:.2f}%, Female: {prob_female * 100:.2f}%")

# ========== Run ==========
if __name__ == "__main__":
    predict_gender(IMAGE_PATH)
