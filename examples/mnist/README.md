# MNIST Dataset
This example trains and evaluates a convolutional neural network to classify the handrwritten digits in the MNIST database.
The model used is from one of François Chollet's [Jupyter notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb)
that accompany his book ["Deep Learning with Python"]([https://www.manning.com/books/deep-learning-with-python](https://www.manning.com/books/deep-learning-with-python-second-edition)).

The code in this directory instruments the model to push metrics to a Prometheus Pushgateway during training and testing.

## Training the model
The [code](https://github.com/hammingweight/gangplank/blob/main/examples/mnist/train.py) to train the model creates a 
[`gangplank.TrainTestExporter`](https://github.com/hammingweight/gangplank/blob/5bd199e195e89293678fa53fce0592fe1f3a4efd/examples/mnist/train.py#L36C5-L36C32)

```
gangplank.TrainTestExporter("127.0.0.1:9091", "mnist"),
```

that specifies the address of the Prometheus PGW and that the job name is "mnist".

You can run the training script by running `python3 train.py`. Once the first training epoch has finished, you should be able to retrieve some
metrics with the prefix `gangplank_train` from the PGW



