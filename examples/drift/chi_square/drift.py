#!/usr/bin/env python
import alibi_detect.cd as cd
import gangplank
import keras
import numpy as np
import random
import scipy.stats as stats
import time

# Load the model
model = keras.models.load_model("../../models/mnist_convnet.keras")

# Use the MNIST test data for exercising model inference
(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255


def get_drift_metrics(_X, Y):
    if len(Y) < 50:
        return gangplank.Drift()
    buckets = [0] * 10
    for y in Y:
        buckets[y.argmax()] += 1
    res = stats.chisquare(buckets)
    return gangplank.Drift(p_value=res.pvalue, test_statistic=res.statistic)


# A proxy that instruments the Keras model. We use the closure to check for
# prediction drift so that we can emit drift metrics.
model = gangplank.PrometheusModel(
    model, port=8561, get_drift_metrics_func=get_drift_metrics
)

# Run some predictions without inducing drift
print("Predictions without drift...")
start_time = time.time()
while time.time() - start_time < 900:
    batch = []
    for _ in range(0, random.randint(1, 200)):
        idx = random.randint(0, 9999)
        batch.append(test_images[idx])
    batch = np.array(batch)
    model.predict(batch, verbose=0)

# Artificially induce drift by discarding all images of zero.
print("Predictions with drift...")
while True:
    batch = []
    size = random.randint(1, 200)
    i = 0
    while i < size:
        idx = random.randint(0, 9999)
        if test_labels[idx] == 0:
            continue
        batch.append(test_images[idx])
        i += 1
    batch = np.array(batch)
    model.predict(batch, verbose=0)
