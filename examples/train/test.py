#!/usr/bin/env python
import gangplank
import keras

model = keras.models.load_model("mnist_convnet.keras")

# Load the MNIST test data (we're not interested in the training data)
(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

# A callback that will:
#   Create a histogram of model weights in the interval [-0.3, +0.3]
#   Abort the test run if an exception occurs (e.g. the PGW is down)
callback = gangplank.TrainTestExporter(
    "http://localhost:9091",
    "mnist",
    histogram_buckets=gangplank.HISTOGRAM_WEIGHT_BUCKETS_0_3,
    ignore_exceptions=False,
)

# Now, test the model
model.evaluate(test_images, test_labels, callbacks=[callback])
