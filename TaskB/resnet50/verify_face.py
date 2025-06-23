import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# ======= Configuration =======
EMBEDDING_MODEL_PATH = "models/triplet/embedding_model.h5"
# THRESHOLD = 1.0870  # Obtained from compute_threshold()
THRESHOLD = 0.88  # Adjusted threshold for verification
IMG_SIZE = (224, 224)

# ======= Load Embedding Model =======
embedding_model = load_model(EMBEDDING_MODEL_PATH, compile=False)

# ======= Preprocessing Function =======
def preprocess_image(img_path, target_size=IMG_SIZE):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ======= Get Embedding =======
def get_embedding(image_path):
    image = preprocess_image(image_path)
    embedding = embedding_model.predict(image)[0]
    return embedding / np.linalg.norm(embedding)  # L2-normalization (optional if not already normalized)

# ======= Match Function =======
def is_match(reference_path, test_path, threshold=THRESHOLD):
    ref_embedding = get_embedding(reference_path)
    test_embedding = get_embedding(test_path)
    distance = np.linalg.norm(ref_embedding - test_embedding)
    print(f"🔍 Distance = {distance:.4f}")
    return distance < threshold

# ======= Example Usage =======
if __name__ == "__main__":
    reference_img = r"E:\CNN_vedic\Data\Task_A\TESTING\unseen\pichai1.png"
    test_img = r"E:\CNN_vedic\Data\Task_A\TESTING\unseen\pichai2.png"

    if is_match(reference_img, test_img):
        print("✅ MATCH: Same identity")
    else:
        print("❌ NO MATCH: Different identity")
