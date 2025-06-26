# ======= Enhanced Triplet Network for Face Verification =======
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision #type: ignore
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Lambda #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ======= Mixed Precision for GPU Speedup =======
mixed_precision.set_global_policy('mixed_float16')

# ======= Configuration =======
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)
MARGIN = 0.3
BATCH_SIZE = 16
EPOCHS = 25
TRAIN_DIR = "Data/Task_B/train"
VAL_DIR = "Data/Task_B/val"
SAVE_PATH = "models/triplet/resnet50_triplet_model.h5"

# ======= Triplet Loss =======
def triplet_loss(margin=MARGIN):
    def _loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, :256], y_pred[:, 256:512], y_pred[:, 512:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return _loss

# ======= Embedding Model =======
def create_embedding_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256)(x)
    x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    embedding_model = Model(inputs=base.input, outputs=x)
    return embedding_model, base

# ======= Triplet Model =======
def build_triplet_model():
    embedding_model, base = create_embedding_model()
    input_anchor = Input(shape=INPUT_SHAPE, name='anchor')
    input_positive = Input(shape=INPUT_SHAPE, name='positive')
    input_negative = Input(shape=INPUT_SHAPE, name='negative')

    emb_anchor = embedding_model(input_anchor)
    emb_positive = embedding_model(input_positive)
    emb_negative = embedding_model(input_negative)

    merged_output = tf.keras.layers.concatenate([emb_anchor, emb_positive, emb_negative])
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=merged_output)
    model.compile(optimizer='adam', loss=triplet_loss())
    return model, embedding_model, base


# ======= Augmented Image Processing =======
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.keras.applications.resnet50.preprocess_input(img)

# ======= Triplet Generator =======
def generate_triplets(data_dir, batch_size=BATCH_SIZE):
    identities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    def get_images(path):
        return [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    while True:
        anchor_batch, positive_batch, negative_batch = [], [], []

        for _ in range(batch_size):
            pos_id = random.choice(identities)
            neg_id = random.choice([i for i in identities if i != pos_id])

            pos_dir = os.path.join(data_dir, pos_id)
            neg_dir = os.path.join(data_dir, neg_id)

            anchor_candidates = get_images(pos_dir)
            distortion_dir = os.path.join(pos_dir, "distortion")
            positive_candidates = get_images(distortion_dir)
            neg_candidates = get_images(neg_dir)

            if not (anchor_candidates and positive_candidates and neg_candidates):
                continue

            anchor_img = preprocess_image(random.choice(anchor_candidates))
            positive_img = preprocess_image(random.choice(positive_candidates))
            negative_img = preprocess_image(random.choice(neg_candidates))

            anchor_batch.append(anchor_img)
            positive_batch.append(positive_img)
            negative_batch.append(negative_img)

        yield (
            np.array(anchor_batch).astype(np.float32),
            np.array(positive_batch).astype(np.float32),
            np.array(negative_batch).astype(np.float32)
        ), np.zeros((batch_size, 768), dtype=np.float32)

# ======= Dataset Wrapper =======
def wrap_generator(data_dir):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 768), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        lambda: generate_triplets(data_dir, BATCH_SIZE),
        output_signature=output_signature
    )

# ======= Training =======
model, embedding_model, base_cnn = build_triplet_model()
train_dataset = wrap_generator(TRAIN_DIR)
val_dataset = wrap_generator(VAL_DIR)

model.fit(
    train_dataset,
    steps_per_epoch=500,
    epochs=EPOCHS,
    validation_data=val_dataset,
    validation_steps=100,
    verbose=1
)

# Optional: Unfreeze top layers of ResNet50 and fine-tune
base_cnn.trainable = True
for layer in base_cnn.layers[:-10]:
    layer.trainable = False

model.compile(optimizer='adam', loss=triplet_loss())
model.fit(train_dataset, steps_per_epoch=250, epochs=5)

model.save(SAVE_PATH)

# ======= Inference Utility =======
def extract_embedding(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    return embedding_model.predict(image)[0]

# ======= Threshold Tuning =======
def compute_threshold(val_dir):
    distances, labels = [], []
    identities = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    def get_images(path):
        return [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for identity in tqdm(identities):
        id_path = os.path.join(val_dir, identity)
        ref_imgs = get_images(id_path)
        distortion_dir = os.path.join(id_path, "distortion")
        distorted_imgs = get_images(distortion_dir)

        if not (ref_imgs and distorted_imgs):
            continue

        ref_embedding = extract_embedding(random.choice(ref_imgs))
        for dist_img_path in distorted_imgs:
            test_embedding = extract_embedding(dist_img_path)
            distances.append(np.linalg.norm(ref_embedding - test_embedding))
            labels.append(1)

        neg_id = random.choice([nid for nid in identities if nid != identity])
        neg_imgs = get_images(os.path.join(val_dir, neg_id))
        if not neg_imgs:
            continue
        neg_embedding = extract_embedding(random.choice(neg_imgs))
        distances.append(np.linalg.norm(ref_embedding - neg_embedding))
        labels.append(0)

    best_acc, best_thresh = 0, 0
    for thresh in np.linspace(0.2, 1.5, 300):
        preds = [1 if d < thresh else 0 for d in distances]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"\n\u2705 Best Threshold: {best_thresh:.4f} (Accuracy: {best_acc:.4f})")
    print(f"Precision: {precision_score(labels, preds):.4f}, Recall: {recall_score(labels, preds):.4f}, F1: {f1_score(labels, preds):.4f}")
    return best_thresh

compute_threshold(VAL_DIR)
