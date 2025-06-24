# ====================== Imports & Config ======================
import os
import random
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras import backend as K #type: ignore
from imblearn.keras import BalancedBatchGenerator #type: ignore

# ====================== GPU Setup ======================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

random.seed(42)

# ====================== Hyperparameters ======================
img_dims = (224, 224, 3)
batch_size = 64
epochs = 100
learning_rate = 1e-4

# ====================== Focal Loss ======================
def focal_loss(gamma=2., alpha=.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return focal_loss_fixed

# ====================== Data Loading ======================
data_dirs = {
    'male': ['Data/Task_A/train/male', 'Data/Task_A/val/male'],
    'female': ['Data/Task_A/train/female', 'Data/Task_A/val/female']
}

data = []
labels = []

for gender, paths in data_dirs.items():
    label = 0 if gender == 'male' else 1
    for path in paths:
        for img_path in glob.glob(os.path.join(path, '*')):
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (img_dims[0], img_dims[1]))
            image = img_to_array(image)
            data.append(image)
            labels.append(label)

data = np.array(data, dtype="float32")
data = preprocess_input(data)
labels = np.array(labels)

# ====================== Train-Test Split ======================
(trainX, testX, trainY_raw, testY_raw) = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels)

testY = to_categorical(testY_raw, num_classes=2)

# ====================== Balanced Generator ======================
train_gen_raw = BalancedBatchGenerator(
    trainX.reshape((trainX.shape[0], -1)), trainY_raw,
    batch_size=batch_size, random_state=42
)
train_gen = ((x.reshape((-1,) + img_dims), to_categorical(y, 2)) for x, y in train_gen_raw)

# ====================== Model Definition ======================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=img_dims))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ====================== Compile ======================
model.compile(loss=focal_loss(gamma=2., alpha=0.75), optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

# ====================== Callbacks ======================
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, verbose=1),
    ModelCheckpoint("models/best_gender_model.h5", save_best_only=True, monitor="val_accuracy", verbose=1)
]

# ====================== Training ======================
steps_per_epoch = len(trainX) // batch_size

H = model.fit(
    train_gen,
    validation_data=(testX, testY),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# ====================== Fine-tune ======================
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(loss=focal_loss(gamma=2., alpha=0.75), optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])

H2 = model.fit(
    train_gen,
    validation_data=(testX, testY),
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# ====================== Prediction & Threshold Tuning ======================
probs = model.predict(testX, batch_size=64)
true_class = np.argmax(testY, axis=1)
precision, recall, thresholds = precision_recall_curve(true_class, probs[:,1])
f1 = 2 * precision * recall / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1)]
print(f"\n🔍 Best threshold for F1: {best_thresh:.4f}")

preds_class = (probs[:,1] >= best_thresh).astype(int)

# ====================== Evaluation ======================
print("\n📊 Classification Report:")
print(classification_report(true_class, preds_class, target_names=["Male", "Female"]))
print("🧮 Confusion Matrix:")
print(confusion_matrix(true_class, preds_class))
print("✅ Accuracy:", accuracy_score(true_class, preds_class))
print("🎯 Precision:", precision_score(true_class, preds_class, zero_division=0))
print("📈 Recall:", recall_score(true_class, preds_class, zero_division=0))
print("🧠 F1 Score:", f1_score(true_class, preds_class, zero_division=0))

# ====================== Plot Training ======================
plt.style.use("ggplot")
plt.figure()
epochs_range = range(len(H.history["loss"]) + len(H2.history["loss"]))
plt.plot(range(len(H.history["loss"])), H.history["loss"], label="Train Loss (Head)")
plt.plot(range(len(H.history["val_loss"])), H.history["val_loss"], label="Val Loss (Head)")
plt.plot(range(len(H.history["accuracy"])), H.history["accuracy"], label="Train Acc (Head)")
plt.plot(range(len(H.history["val_accuracy"])), H.history["val_accuracy"], label="Val Acc (Head)")
plt.plot(range(len(H.history["loss"]), len(epochs_range)), H2.history["loss"], label="Train Loss (FT)")
plt.plot(range(len(H.history["val_loss"]), len(epochs_range)), H2.history["val_loss"], label="Val Loss (FT)")
plt.plot(range(len(H.history["accuracy"]), len(epochs_range)), H2.history["accuracy"], label="Train Acc (FT)")
plt.plot(range(len(H.history["val_accuracy"]), len(epochs_range)), H2.history["val_accuracy"], label="Val Acc (FT)")
plt.title("Training and Validation")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("training_plot_mobilenetv2_gender.png")
plt.show()

# ====================== Save Final Model ======================
model.save("models/gender_classifier_final.h5")
print("\n📁 Final model saved as models/gender_classifier_final.h5")
