import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
