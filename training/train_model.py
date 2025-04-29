import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Generate dummy brightness data
X = np.random.rand(1000, 10)
y = (X.mean(axis=1) > 0.5).astype(int)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=32)

# Save original TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('training/tiny_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 10).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.float16]

quantized_tflite_model = converter.convert()
with open('training/tiny_model_quantized.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

print("✅ Model and Quantized Model Saved")
