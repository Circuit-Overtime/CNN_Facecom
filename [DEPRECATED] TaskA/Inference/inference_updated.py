import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image #type: ignore
import matplotlib.pyplot as plt


model_path = "models/keras/mobilenet_gender_final_v2.keras"
model = tf.keras.models.load_model(model_path)

class_labels = ['Male', 'Female']  


img_path = "Data/Task_A/TESTING/women/20.jpg" 


IMG_SIZE = 224
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)  


pred = model.predict(img_array)[0][0]
pred_class = 1 if pred >= 0.5 else 0
confidence = pred if pred >= 0.5 else 1 - pred


print(f"🧠 Predicted Gender: {class_labels[pred_class]} ({confidence*100:.2f}% confidence)")


plt.imshow(img)
plt.title(f"{class_labels[pred_class]} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()
