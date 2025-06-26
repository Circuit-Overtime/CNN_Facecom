import os
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img #type: ignore
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# ========== CONFIG ==========
GENDER_MODEL_PATH = "PRODUCTION/models/vgg19_final_epoch.h5" #AVAILABLE AT https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish102
FACE_MODEL_PATH = "PRODUCTION/models/embedding_sequel.h5" # AVAILABLE AT https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish101
IMG_SIZE_GENDER = (224, 224)
IMG_SIZE_FACE = (224, 224)
THRESHOLD = 0.945
GENDER_THRESHOLD = 0.45
# ========== FOCAL LOSS FUNCTION ==========
def focal_loss(gamma=2., alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return alpha * tf.pow(1. - p_t, gamma) * ce
    return loss

# ========== IMAGE PREPROCESSING ==========
def preprocess_image(file_storage, size, preprocess_fn):
    try:
        img_bytes = file_storage.read()  
        image = Image.open(BytesIO(img_bytes)).convert("RGB")  
        image = image.resize(size)
        image = img_to_array(image)
        image = preprocess_fn(image)
        return np.expand_dims(image, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")
# ========== ROUTES ==========

@app.route("/gender", methods=["POST"])
def predict_gender():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = request.files['image']
        processed_img = preprocess_image(image, IMG_SIZE_GENDER, vgg_preprocess)

        # Lazy load the gender model
        gender_model = load_model(GENDER_MODEL_PATH, custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.5)})
        preds = gender_model.predict(processed_img)[0]

        female_prob = preds[1]
        predicted_class = int(female_prob > GENDER_THRESHOLD)
        label = "Female" if predicted_class == 1 else "Male"
        confidence = float(female_prob if predicted_class == 1 else 1 - female_prob)

        del gender_model  # Free memory
        tf.keras.backend.clear_session()

        return jsonify({
            "gender": label,
            "confidence": round(confidence, 4)
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/face", methods=["POST"])
def verify_face():
    if 'ref_image' not in request.files or 'query_image' not in request.files:
        return jsonify({"error": "Both ref_image and query_image are required"}), 400

    try:
        ref_image = request.files['ref_image']
        query_image = request.files['query_image']

        processed_ref = preprocess_image(ref_image, IMG_SIZE_FACE, resnet_preprocess)
        processed_query = preprocess_image(query_image, IMG_SIZE_FACE, resnet_preprocess)

        # Lazy load the face embedding model
        face_model = load_model(FACE_MODEL_PATH, compile=False)

        ref_embed = face_model.predict(processed_ref)[0]
        query_embed = face_model.predict(processed_query)[0]

        # Normalize
        ref_embed = ref_embed / np.linalg.norm(ref_embed)
        query_embed = query_embed / np.linalg.norm(query_embed)

        distance = np.linalg.norm(ref_embed - query_embed)
        match = distance < THRESHOLD

        del face_model  # Free memory
        tf.keras.backend.clear_session()

        return jsonify({
            "match": bool(match),
            "distance": round(float(distance), 4),
            "threshold": THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
