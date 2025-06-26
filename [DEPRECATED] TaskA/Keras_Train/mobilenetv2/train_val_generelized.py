import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore

# ========= Suppress Warnings ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide all but fatal
random.seed(42)

# ========= Enable GPU Memory Growth ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ========= Configuration ==========
epochs = 100
lr = 1e-4
batch_size = 64
img_dims = (224, 224, 3)  # Supported by MobileNetV2
data_dirs = {
    'male': ['Data/Task_A/train/male', 'Data/Task_A/val/male'],
    'female': ['Data/Task_A/train/female', 'Data/Task_A/val/female']
}

# ========= Load and Preprocess Data ==========
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

# ========= Split & Encode ==========
(trainX, testX, trainY_raw, testY_raw) = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels)

trainY = to_categorical(trainY_raw, num_classes=2)
testY = to_categorical(testY_raw, num_classes=2)

# ========= Compute Class Weights ==========
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(trainY_raw), y=trainY_raw)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# ========= Augmentation ==========
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

# ========= Build Model ==========
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=img_dims))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# ========= Callbacks ==========
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, verbose=1),
    ModelCheckpoint("models/best_gender_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
]

# ========= Initial Training ==========
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# ========= Fine-tuning ==========
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

H_finetune = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# ========= Evaluation ==========
preds = model.predict(testX, batch_size=64)
preds_class = np.argmax(preds, axis=1)
true_class = np.argmax(testY, axis=1)

print("\n📊 Classification Report:")
print(classification_report(true_class, preds_class, target_names=["Male", "Female"]))
print("🧮 Confusion Matrix:")
print(confusion_matrix(true_class, preds_class))

print("\n✅ Accuracy:", accuracy_score(true_class, preds_class))
print("🎯 Precision:", precision_score(true_class, preds_class, zero_division=0))
print("📈 Recall:", recall_score(true_class, preds_class, zero_division=0))
print("🧠 F1 Score:", f1_score(true_class, preds_class, zero_division=0))

# ========= Plot Training ==========
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

# ========= Save Final Model ==========
model.save("models/gender_classifier_final.h5")
