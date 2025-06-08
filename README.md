# Gangplank
## Export Keras Metrics to Prometheus
[Prometheus](https://prometheus.io/) is a monitoring system that pulls metrics from applications and infrastructure.
Gangplank is a Python package for exposing Keras model metrics to Prometheus. Metrics can be exported from training,
evaluation and inference tasks. Training and testing metrics are exported using the Prometheus [Pushgateway](https://prometheus.io/docs/instrumenting/pushing/).
Inference metrics are exposed by instrumenting a proxy of a Keras model.

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
The example code uses a model from ["Deep Learning with Python, Second Edition"](https://www.manning.com/books/deep-learning-with-python-second-edition).
Gangplank was inspired by the same book's coverage of callbacks and TensorBoard.