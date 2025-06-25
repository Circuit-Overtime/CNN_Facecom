import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

# ========== Configuration ==========
epochs = 100
lr = 1e-4
batch_size = 64
img_dims = (250, 250, 3)
data_dirs = {
    'male': ['Data/Task_A/train/male', 'Data/Task_A/val/male'],
    'female': ['Data/Task_A/train/female', 'Data/Task_A/val/female']
}

# ========== Load and Preprocess Data ==========
data = []
labels = []

for gender, paths in data_dirs.items():
    label = 0 if gender == 'male' else 1
    for path in paths:
        image_files = glob.glob(os.path.join(path, '*'))
        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (img_dims[0], img_dims[1]))
            image = img_to_array(image)
            data.append(image)
            labels.append(label)

# ========== Convert and Normalize ==========
data = np.array(data, dtype="float32")
data = preprocess_input(data)
labels = np.array(labels)

# ========== Split and Encode ==========
(trainX, testX, trainY_raw, testY_raw) = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels)
trainY = to_categorical(trainY_raw, num_classes=2)
testY = to_categorical(testY_raw, num_classes=2)

# ========== Compute Class Weights ==========
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(trainY_raw), y=trainY_raw)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# ========== Data Augmentation ==========
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# ========== Build Model ==========
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=img_dims))
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# ========== Callbacks ==========
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, verbose=1)
]

# ========== Train Model ==========
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# ========== Evaluate ==========
preds = model.predict(testX, batch_size=64)
preds_class = np.argmax(preds, axis=1)
true_class = np.argmax(testY, axis=1)

print("\nClassification Report:")
print(classification_report(true_class, preds_class, target_names=["Male", "Female"]))
print("Confusion Matrix:")
print(confusion_matrix(true_class, preds_class))

print("\nMetrics:")
print("Accuracy:", accuracy_score(true_class, preds_class))
print("Precision:", precision_score(true_class, preds_class, zero_division=0))
print("Recall:", recall_score(true_class, preds_class, zero_division=0))
print("F1 Score:", f1_score(true_class, preds_class, zero_division=0))

# ========== Plot Training ==========
plt.style.use("ggplot")
plt.figure()
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="Train Acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("training_plot_mobilenetv2_gender.png")
plt.show()

model.save("models/gender_classifier_final.h5")