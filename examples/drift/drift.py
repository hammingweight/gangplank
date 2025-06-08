#!/usr/bin/env python
import alibi_detect.cd as cd
import random
import keras
import numpy as np

#mul = 1.0
def preprocess(x):
    return x # * mul
    #print(x)
    #print(x.shape)
    y=x.argmax(axis=1).astype("float32") * mul
    y=np.expand_dims(y, axis=1)
    #print(y.shape)
    return y
    #return x.argmax(axis=0)
    #:return np.ndarray([x.argmax()]).reshape(1,1)

# The Keras model that we'll use
model = keras.models.load_model("../models/mnist_convnet.keras")

# Use the MNIST test data for exercising model inference
(train_images, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

y_ref = model.predict(train_images[:1000])
dd = cd.MMDDriftOnline(y_ref, ert=100, window_size=200, backend="pytorch", n_bootstraps=2000, preprocess_fn=preprocess)
print(dd)

#mul = 2.0
y_test = model.predict(test_images)
c = 0
t = 0.0
for y in y_test:
    if y.argmax() >= 9:
        if random.random() > 1.9:
            # print("continue")
            continue
    pred = dd.predict(y)
    pred = dd.predict(y)
    pred = dd.predict(y)
    pred = dd.predict(y)
    #pred = dd.predict(y)
    if pred["data"]["is_drift"] == 1:
        c += 1
        if c==1:
            ert = t
        print("X", end="")
    else:
        print(".", end="")
    t += 1.0

print()
print(pred)
print()
print(c)
print(t)
print(c/t)
print(ert)
