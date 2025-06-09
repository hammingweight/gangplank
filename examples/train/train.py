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

# Load the MNIST training data (we're not interested in the testing data)
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255

# We use the Adam optimizer for training; Chollet uses RMSprop which is more performant
# but the Adam optimizer demonstrates overfitting more clearly.
optimizer = "adam"

# Create a callback to save the best model (i.e. the model state before overfitting)
# and a callback to push metrics to the Prometheus pushgateway.
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="../models/mnist_convnet.keras",
        save_best_only=True,
        monitor="val_loss",
    ),
    gangplank.TrainTestExporter("127.0.0.1:9091", "mnist"),
]

# Train the model (10% of the training data is reserved for validation.)
validation_split = 0.1
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    train_images,
    train_labels,
    epochs=30,
    validation_split=validation_split,
    batch_size=64,
    callbacks=callbacks,
)
