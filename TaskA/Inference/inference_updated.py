import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image #type: ignore
import matplotlib.pyplot as plt

# Load the saved model
model_path = "models/keras/mobilenet_gender_final_v2.keras"
model = tf.keras.models.load_model(model_path)

# Class mapping (adjust if yours are different)
class_labels = ['Male', 'Female']  

# Image path to test
img_path = "Data/Task_A/TESTING/women/20.jpg" 

# Preprocess the image
IMG_SIZE = 224
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# Predict
pred = model.predict(img_array)[0][0]
pred_class = 1 if pred >= 0.5 else 0
confidence = pred if pred >= 0.5 else 1 - pred

# Display result
print(f"🧠 Predicted Gender: {class_labels[pred_class]} ({confidence*100:.2f}% confidence)")

# Optional: show the image
plt.imshow(img)
plt.title(f"{class_labels[pred_class]} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()
