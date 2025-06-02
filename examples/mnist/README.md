# MNIST Dataset
This example trains and evaluates a convolutional neural network to classify the handrwritten digits in the MNIST database.
The model used is from one of François Chollet's [Jupyter notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb)
that accompany his book ["Deep Learning with Python"](https://www.manning.com/books/deep-learning-with-python).

The code in this directory instruments the model so that it pushes metrics to a Prometheus Pushgateway during training and testing.

## Training the model

