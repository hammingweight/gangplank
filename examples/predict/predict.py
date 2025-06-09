#!/usr/bin/env python
import gangplank
import keras
import random
import time

random.seed(1)

# The Keras model that we'll use
model = keras.models.load_model("../models/mnist_convnet.keras")

# A proxy that instruments the Keras model. The 'port' argument can
# omitted if you're already running a Prometheus server.
model = gangplank.PrometheusModel(model, port=8561)

# Use the MNIST test data for exercising model inference
(_, _), (test_images, _) = keras.datasets.mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

print(
    """Run
    curl -s http://localhost:8561/ | grep ^gangplank | grep -v created
to view metrics
"""
)

# Mindlessly do inference on randomly sized batches with random delays
idx = 0
while idx < 100:
    size = random.randint(1, 10)
    x = test_images[idx : idx + size]
    print(f"Doing inference on images {idx} - {idx + size - 1}")
    # Next line could be replaced with
    #   res = model(x)
    res = model.predict(x, verbose=0)
    print(f"Digits: {res.argmax(axis=1)}")
    time.sleep(random.randint(1, 5))
    idx += size

# Sleep to keep the server up.
print("\nDone. Sleeping; hit Ctrl-C to end...")
time.sleep(3600)
