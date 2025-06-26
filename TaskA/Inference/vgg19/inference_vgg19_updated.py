import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore

# ========== Config ==========
MODEL_PATH = "models/final_vgg19_gender_model.h5"
IMAGE_PATH = "Data/Task_A/TESTING/unseen/hritik2.jpg"  
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Male", "Female"]
THRESHOLD = 0.40  
# =========== FOCAL LOSS ==========
def focal_loss(gamma=2., alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return alpha * tf.pow(1. - p_t, gamma) * ce
    return loss

# ========== Load Model ==========
print("🔍 Loading model...")
model = load_model(MODEL_PATH, custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.5)})

# ========== Preprocess ==========
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0), image_path

# ========== Grad-CAM ==========
def gradcam_visualize(image_path, model, layer_name="block5_conv4", class_index=None):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(rgb_image, (224, 224))
    x = preprocess_input(np.expand_dims(input_image.astype(np.float32), axis=0))

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        if class_index is None:
            class_index = int(predictions[0][1] > THRESHOLD)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb_image, 0.6, colored, 0.4, 0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM Overlay: {CLASS_NAMES[class_index]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ========== Predict ==========
def predict_gender(image_path):
    image, _ = preprocess_image(image_path)
    preds = model.predict(image)[0]
    female_prob = preds[1]
    predicted_class = int(female_prob > THRESHOLD)
    label = CLASS_NAMES[predicted_class]
    confidence = female_prob if predicted_class == 1 else 1 - female_prob
    print(f"🧠 Prediction: {label} ({confidence * 100:.2f}%)")

    gradcam_visualize(image_path, model, class_index=predicted_class)

# ========== Run ==========
if __name__ == "__main__":
    predict_gender(IMAGE_PATH)
