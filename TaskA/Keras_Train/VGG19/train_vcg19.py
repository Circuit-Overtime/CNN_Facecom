# ================== Imports & Config ==================
import os, cv2, glob, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.applications import VGG19 #type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input #type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
random.seed(42)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ================== Configuration ==================
IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 100
FINE_TUNE_EPOCHS = 30
LR = 1e-4

train_dirs = {
    "male": "Data/Task_A/train/male",
    "female": "Data/Task_A/train/augmentedFemales"
}
val_dirs = {
    "male": "Data/Task_A/val/male",
    "female": "Data/Task_A/val/augmentedFemales"
}

# ================== Load & Preprocess Images ==================
def load_data(dirs_dict):
    data, labels = [], []
    for label_name, path in dirs_dict.items():
        label = 0 if label_name == "male" else 1
        for img_path in glob.glob(os.path.join(path, "*")):
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
            image = img_to_array(image)
            data.append(image)
            labels.append(label)
    return np.array(data, dtype="float32"), np.array(labels)

print("\U0001F4E5 Loading training and validation data...")
trainX, trainY_raw = load_data(train_dirs)
valX, valY_raw = load_data(val_dirs)

trainX = preprocess_input(trainX)
valX = preprocess_input(valX)
trainY = to_categorical(trainY_raw, num_classes=2)
valY = to_categorical(valY_raw, num_classes=2)

# ================== Data Augmentation ==================
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# ================== Periodic Save Callback ==================
class PeriodicModelSaver(Callback):
    def __init__(self, save_path, every_n_epochs=10):
        super().__init__()
        self.save_path = save_path
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            try:
                save_path = f"{self.save_path}_epoch{epoch+1}.h5"
                self.model.save(save_path)
                print(f"\u2705 Periodic model saved at: {save_path}")
            except Exception as e:
                print(f"\u274C Failed to save at epoch {epoch+1}: {e}")

# ================== Build AgeGender-Inspired Model ==================
print("\u2699\ufe0f Building VGG19-inspired model for gender classification...")
base_model = VGG19(include_top=False, weights="imagenet", input_tensor=Input(shape=IMG_SIZE))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=out)
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=Adam(learning_rate=LR), loss=loss_fn, metrics=["accuracy"])

# ================== Callbacks ==================
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("models/best_vgg19_gender_model.h5", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
    PeriodicModelSaver("models/periodic_vgg19_gender_model", every_n_epochs=10)
]

# ================== Phase 1: Training Head ==================
print("\U0001F680 Training fully connected head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ================== Phase 2: Fine-tuning ==================
print("\U0001F527 Fine-tuning top convolutional blocks...")
for layer in base_model.layers[-8:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])
H_fine = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ================== Evaluation ==================
print("\U0001F4CA Final evaluation on validation data...")
preds = model.predict(valX, batch_size=32)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(valY, axis=1)

print("\U0001F4CB Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=["Male", "Female"]))

cm = confusion_matrix(true_classes, pred_classes)
male_acc = cm[0, 0] / cm[0].sum() if cm.shape[0] > 1 and cm[0].sum() > 0 else 0
female_acc = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 and cm[1].sum() > 0 else 0
print("\U0001F9EE Confusion Matrix:")
print(cm)
print(f"\u2705 Overall Accuracy: {accuracy_score(true_classes, pred_classes):.4f}")
print(f"\U0001F3AF Precision: {precision_score(true_classes, pred_classes):.4f}")
print(f"\U0001F4C8 Recall: {recall_score(true_classes, pred_classes):.4f}")
print(f"\U0001F9E0 F1 Score: {f1_score(true_classes, pred_classes):.4f}")
print(f"\u2642\ufe0f Male Accuracy: {male_acc:.4f}")
print(f"\u2640\ufe0f Female Accuracy: {female_acc:.4f}")

# ================== Plot Training Curve ==================
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
total_epochs_head = len(H.history["loss"])
total_epochs_fine = len(H_fine.history["loss"])
total_epochs = total_epochs_head + total_epochs_fine

plt.plot(range(total_epochs_head), H.history["accuracy"], label="Train Acc (Head)")
plt.plot(range(total_epochs_head), H.history["val_accuracy"], label="Val Acc (Head)")
plt.plot(range(total_epochs_head, total_epochs), H_fine.history["accuracy"], label="Train Acc (Fine)")
plt.plot(range(total_epochs_head, total_epochs), H_fine.history["val_accuracy"], label="Val Acc (Fine)")

plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("training_accuracy_vgg19_gender.png")
plt.show()

# ================== Save Final Model ==================
model.save("models/final_vgg19_gender_model.h5")
print("\u2705 Final model saved as models/final_vgg19_gender_model.h5")
