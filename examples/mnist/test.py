#!/usr/bin/env python
import gangplank
import keras

model = keras.models.load_model("mnist_convnet.keras")

# Load the MNIST test data (we're not interested in the training data)
(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

callback = gangplank.TrainTestExporter(
    "http://localhost:9091",
    "mnist",
    histogram_buckets=gangplank.HISTOGRAM_WEIGHT_BUCKETS_0_3,
    ignore_exceptions=False,
)

model.evaluate(test_images, test_labels, callbacks=[callback])
