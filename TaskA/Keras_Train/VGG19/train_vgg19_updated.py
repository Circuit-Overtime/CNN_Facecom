# ================== Imports & Config ==================
import os, cv2, glob, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.applications import VGG19  # type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input  # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

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
    "female": "Data/Task_A/train/female"
}
val_dirs = {
    "male": "Data/Task_A/val/male",
    "female": "Data/Task_A/val/female"
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

# ================== Custom Focal Loss ==================
def focal_loss(gamma=2., alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return alpha * tf.pow(1. - p_t, gamma) * ce
    return loss

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
                print(f"✅ Periodic model saved at: {save_path}")
            except Exception as e:
                print(f"❌ Failed to save at epoch {epoch+1}: {e}")

# ================== Class Weights ==================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(trainY_raw), y=trainY_raw)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("\U0001F522 Class Weights:", class_weight_dict)

# ================== Build Model ==================
print("\u2699\ufe0f Building VGG19 model...")
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
loss_fn = focal_loss(gamma=2.0, alpha=0.5)
model.compile(optimizer=Adam(learning_rate=LR), loss=loss_fn, metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("models/vgg19_train.h5", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
    PeriodicModelSaver("models/periodic_vgg19_gender_model", every_n_epochs=10)
]

# ================== Training ==================
print("\U0001F680 Training fully connected head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ================== Fine-tuning ==================
print("\U0001F527 Fine-tuning top convolutional blocks...")
for layer in base_model.layers[-8:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])
H_fine = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ================== Evaluation with Auto Threshold Tuning ==================
print("\U0001F4CA Final evaluation on validation data...")
preds = model.predict(valX, batch_size=32)
female_probs = preds[:, 1]
true_classes = np.argmax(valY, axis=1)

# Auto-threshold selection based on best F1 score
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.4, 0.61, 0.01):
    pred_classes = (female_probs > thresh).astype(int)
    f1 = f1_score(true_classes, pred_classes)
    if f1 > best_f1:
        best_thresh, best_f1 = thresh, f1

print(f"\U0001F50D Best Threshold: {best_thresh:.2f} | F1 Score: {best_f1:.4f}")
pred_classes = (female_probs > best_thresh).astype(int)

print("\U0001F4CB Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=["Male", "Female"]))

cm = confusion_matrix(true_classes, pred_classes)
male_acc = cm[0, 0] / cm[0].sum() if cm.shape[0] > 1 and cm[0].sum() > 0 else 0
female_acc = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 and cm[1].sum() > 0 else 0
print("\U0001F9EE Confusion Matrix:")
print(cm)
print(f"✅ Overall Accuracy: {accuracy_score(true_classes, pred_classes):.4f}")
print(f"\U0001F3AF Precision: {precision_score(true_classes, pred_classes):.4f}")
print(f"\U0001F4C8 Recall: {recall_score(true_classes, pred_classes):.4f}")
print(f"\U0001F9E0 F1 Score: {f1_score(true_classes, pred_classes):.4f}")
print(f"♂️ Male Accuracy: {male_acc:.4f}")
print(f"♀️ Female Accuracy: {female_acc:.4f}")
# ================== Plot Training History ==================
plt.figure(figsize=(12, 6))
plt.plot(H.history["accuracy"], label="Train Accuracy")
plt.plot(H.history["val_accuracy"], label="Val Accuracy")
plt.plot(H.history["loss"], label="Train Loss")
plt.plot(H.history["val_loss"], label="Val Loss")
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")
plt.legend()
plt.grid()
plt.savefig("plots/vgg19_training_history.png")
plt.show()
# ================== Plot Fine-tuning History ==================
plt.figure(figsize=(12, 6))
plt.plot(H_fine.history["accuracy"], label="Fine-tune Train Accuracy")
plt.plot(H_fine.history["val_accuracy"], label="Fine-tune Val Accuracy")
plt.plot(H_fine.history["loss"], label="Fine-tune Train Loss")
plt.plot(H_fine.history["val_loss"], label="Fine-tune Val Loss")
plt.title("Fine-tuning History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")
plt.legend()
plt.grid()
plt.savefig("plots/vgg19_fine_tuning_history.png")


# ================== Save Final Model ==================
model.save("models/vgg19_final_epoch.h5")
print("✅ Final model saved as models/vgg19_final_epoch.h5")