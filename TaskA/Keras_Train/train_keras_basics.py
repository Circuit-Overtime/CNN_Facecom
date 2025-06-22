import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore

# ✅ Base dataset path
base_path = r'C:\Users\ayush\Desktop\Gender-Detection\gender_dataset_face'

# Params
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Augmentation (with validation split)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  
)

# Data generators
train_gen = datagen.flow_from_directory(
    base_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Model base
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
base_model.trainable = False  # Freeze initially

# Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid', kernel_regularizer=l2(WEIGHT_DECAY))(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile with label smoothing
loss_fn = BinaryCrossentropy(label_smoothing=0.1)
optimizer = Adam(learning_rate=INIT_LR)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint("mobilenet_gender_v2.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# Stage 1: Train top only
print("🔁 Training top layers only...")
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)

# Stage 2: Fine-tune middle + top layers
print("🔧 Unfreezing middle layers for fine-tuning...")
for layer in base_model.layers[-50:]:  # Fine-tune last 50 layers
    layer.trainable = True

# Re-compile with lower LR
model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=['accuracy'])

# Final Training
print("🚀 Fine-tuning full model...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# Save final model
model.save("mobilenet_gender_final_v2.keras")
print("✅ Final model saved as 'mobilenet_gender_final_v2.keras'")
