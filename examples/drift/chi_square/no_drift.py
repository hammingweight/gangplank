#!/usr/bin/env python
import alibi_detect.cd as cd
import gangplank
import keras
import random
import time

# Load the model
model = keras.models.load_model("../models/mnist_convnet.keras")

# Use the MNIST test data for exercising model inference
(train_images, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

# A proxy that instruments the Keras model. We use the closure to check for
# prediction drift so that we can emit drift metrics.
model = gangplank.PrometheusModel(model, port=8561)

input("Press Enter to continue...")

# Run some predictions without inducing drift
print("Predictions without drift...")
start_time = time.time()
while time.time() - start_time < 900:
    idx = random.randint(0, 9999)
    model.predict(test_images[idx].reshape(1, 28, 28, 1), verbose=0)

# Artificially induce drift by discarding all images of zero.
print("Predictions with drift...")
while True:
    idx = random.randint(0, 9999)
    label = test_labels[idx]
    if label == 0:
        continue
    model.predict(test_images[idx].reshape(1, 28, 28, 1), verbose=0)
