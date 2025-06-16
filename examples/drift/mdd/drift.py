#!/usr/bin/env python
import alibi_detect.cd as cd
import gangplank
import keras
import random
import time

# Load the model
model = keras.models.load_model("../../models/mnist_convnet.keras")

# Use the MNIST test data for exercising model inference
(train_images, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

# Get expected predictions for the first 20000 images in the training model
preds = model.predict(train_images[:20000], verbose=0)

# Create an online drift detector (MMD)
drift_detector = cd.MMDDriftOnline(
    preds, ert=500, window_size=200, backend="pytorch", n_bootstraps=5000
)


# A closure that uses the MMD to check whether predicted values are drifting
# from the training data. The function returns the number of times that
# the MMD reports that drift was detected in the predictions. Note
# that the input values (_X) are discarded since we're interested in
# prediction drift not data drift.
def get_drift_metrics(_X, Y):
    count = 0
    ts = None
    for y in Y:
        res = drift_detector.predict(y, return_test_stat=True)["data"]
        if res["is_drift"] == 1:
            count += 1
        if len(Y) == 1:
            ts = res["test_stat"]
    return gangplank.Drift(drift_detected=count, test_statistic=ts)


# A proxy that instruments the Keras model. We use the closure to check for
# prediction drift so that we can emit drift metrics.
model = gangplank.PrometheusModel(
    model, port=8561, get_drift_metrics_func=get_drift_metrics
)

input("Drift detector created. Press Enter to continue...")

# Run some predictions without inducing drift
print("\nPredictions without drift...")
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

print("Done.")
