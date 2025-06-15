#!/usr/bin/env python
import gangplank
import keras

# The following convolutional neural network for identifying digits in the MNIST
# dataset is copied from François Chollet's github repo
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb
# The same code can be found in Chollet's book published by Manning:
# https://www.manning.com/books/deep-learning-with-python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = keras.layers.MaxPooling2D(pool_size=2)(x)
x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = keras.layers.MaxPooling2D(pool_size=2)(x)
x = keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


# A callback that populates the learning rate in the "logs" dictionary
class LearningRateMetric(keras.callbacks.Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = self.optimizer.learning_rate.numpy().item()
        logs["learning_rate"] = lr


# Load the MNIST training data (we're not interested in the testing data)
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255

# An optimizer with a learning rate of 0.005
optimizer = keras.optimizers.Adagrad(learning_rate=0.005)

# We need the LearningRateMetric callback to be called before
# the TrainTestExporter
callbacks = [
    LearningRateMetric(optimizer),
    gangplank.TrainTestExporter("127.0.0.1:9091", "mnist"),
]

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    train_images,
    train_labels,
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
)
