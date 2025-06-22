import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore
import matplotlib.pyplot as plt
import time
# Load the trained model
model = load_model("models/keras/mobilenet_gender_final_v2.keras")

IMG_WIDTH, IMG_HEIGHT = 224, 224
labels = ["Male", "Female"]

def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image, verbose=0)[0]
    return np.argmax(prediction)

def evaluate_folder(folder_path, true_label):
    correct = 0
    total = 0
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        pred = predict_image(fpath)
        if pred is not None:
            total += 1
            if pred == true_label:
                correct += 1
    return correct, total

if __name__ == "__main__":
    base_dir = "train"
    male_dir = os.path.join(base_dir, "male")
    female_dir = os.path.join(base_dir, "female")

    male_correct, male_total = evaluate_folder(male_dir, 0)
    time.sleep(3) 
    female_correct, female_total = evaluate_folder(female_dir, 1)

    total_correct = male_correct + female_correct
    total = male_total + female_total
    overall_acc = total_correct / total if total > 0 else 0
    male_acc = male_correct / male_total if male_total > 0 else 0
    female_acc = female_correct / female_total if female_total > 0 else 0

    print(f"Male accuracy: {male_acc:.2%} ({male_correct}/{male_total})")
    print(f"Female accuracy: {female_acc:.2%} ({female_correct}/{female_total})")
    print(f"Overall accuracy: {overall_acc:.2%} ({total_correct}/{total})")

    # Plotting
    plt.bar(['Male', 'Female', 'Overall'], [male_acc, female_acc, overall_acc], color=['blue', 'pink', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Gender Classification Accuracy')
    plt.ylim(0, 1)
    plt.show()