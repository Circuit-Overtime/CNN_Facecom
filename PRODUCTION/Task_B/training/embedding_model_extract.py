import tensorflow as tf

# Load the full triplet model (do NOT compile because of custom loss)
triplet_model = tf.keras.models.load_model(
    "PRODUCTION/models/tripletNetwork.h5",
    compile=False  # Skip compiling to avoid needing the custom triplet loss
)

# Locate the embedding model (256-d output) among submodels
embedding_model = None
for layer in triplet_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.output_shape[-1] == 256:
        embedding_model = layer
        break

if embedding_model:
    embedding_model.save("PRODUCTION/models/embedding_sequel.h5")
    print("✅ Saved embedding_model.h5 successfully.")
else:
    print("❌ Embedding model not found in triplet_model.")
