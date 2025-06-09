# Gangplank
## Exposing Keras Metrics to Prometheus
[Keras](https://keras.io/) is a library for creating artificial neural networks. [Prometheus](https://prometheus.io/) is a monitoring system that pulls metrics from applications and infrastructure.
Gangplank is a bridge from Keras to Prometheus that exports Keras training, evaluation and inference metrics to Prometheus.

Keras metrics are exposed in two ways:
 * Training and testing metrics use Keras [callbacks](https://keras.io/api/callbacks/) to push metrics to a Prometheus [Pushgateway](https://prometheus.io/docs/instrumenting/pushing/).
 * Inference metrics are exposed by instrumenting a proxy of a Keras model.

The [examples](https://github.com/hammingweight/gangplank/tree/main/examples) demonstrate both techniques to export metrics to Prometheus.

## What Metrics are exported?
### Training Metrics
During training, the following metrics are exported:
 * The number of completed training epochs
 * The time spent training
 * The number of model weights (both trainable and non-trainable)
 * The model's loss
 * All metrics configured for the model (e.g. accuracy for a classification model or mean absolute error for a regression model)
 * (Optionally) A histogram of the model's trainable weights at the end of the training run

### Testing (Evaluation) Metrics
For testing (i.e. evaluation), the following metrics are exported:
 * The time spent testing
 * The model's loss
 * All metrics configured for the model (accuracy, mean absolute error, etc.)
 * (Optionally) A histogram of the model's trainable weights

### Prediction (Inference) Metrics
A deployed model can expose the following metrics:
 * The total number of model predictions
 * The time spent doing inference
 * (Optionally) Drift metrics; e.g. a p-value

## Installing Gangplank
Gangplank can be installed from PyPI

```
pip install gangplank
```

The installation will also install Keras. Keras needs a tensor arithmetic backend like TensorFlow, JAX or PyTorch. You can install a
backend at the same time as installing Gangplank by running one of the following

```
pip install gangplank[tensorflow]
pip install gangplank[jax]
pip install gangplank[torch]
```

Note: Running, e.g., `pip install gangplank[jax]` will install a CPU-only version of JAX. If you want, say, CUDA support you should install JAX separately

```
pip install gangplank
pip install jax[cuda12]
```

Similar comments apply to TensorFlow and PyTorch.

## Examples
Examples of using Gangplank can be found [here](https://github.com/hammingweight/gangplank/tree/main/examples).

## Acknowledgement
The example code uses a model from ["Deep Learning with Python, Second Edition"](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet.
Gangplank was inspired by the same book's coverage of callbacks and TensorBoard.
