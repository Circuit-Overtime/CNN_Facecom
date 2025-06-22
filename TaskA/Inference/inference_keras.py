import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# ========== Configuration ==========
model_path = "models/gender_classifier_final.h5"
image_size = (250, 250)  # same as training

# ========== Load the trained model ==========
model = load_model(model_path)

# ========== Gender label mapping ==========
labels = {0: "Male", 1: "Female"}

def predict_gender(image_path: str) -> str:
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    image = cv2.resize(image, image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Make prediction
    preds = model.predict(image)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_class] * 100

    result = labels[predicted_class]
    print(f"Predicted Gender: {result} ({confidence:.2f}% confidence)")
    return result

# ========== Example usage ==========
if __name__ == "__main__":
    # Replace this with your test image path
    test_image_path = r"E:\CNN_vedic\Data\Task_A\TESTING\women\133.jpg"
    predict_gender(test_image_path)
