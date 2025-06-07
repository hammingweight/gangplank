# Prediction/Inference
The example in this directory assumes that you have run [`train.py`](../train/train.py) to create
a Keras model. The model saved (to disk) by the training script is needed for inference by the [`predict.py`](./predict.py)
example in this directory.

## The `PrometheusModel` Class
An instance of the `PrometheusModel` class should be created to instrument a Keras model. 
`PrometheusModel` is a "proxy" class:  methods invoked on the proxy are
delegated to the underlying Keras model. Additionally, the `predict()` and `__call__()` methods of the proxy are instrumented to provide
metrics (e.g. number of invocations and running time).

The constructor of the [`PrometheusModel`](./../../src/gangplank/prometheus_model.py) class takes one mandatory
argument and some optional arguments:
 * `model` is a Keras model that will be proxied and instrumented.
 * `port` is an optional `int` argument; if provided, an HTTP server is started on the port that exposes
    metrics. If the user has already configured a server to publish Prometheus metrics, this argument should
    not be supplied.
 * `registry` is an optional collector registry. If not supplied, the default registry is used which is usually
    the correct choice.

A minimal instantiation of a `PrometheusModel` would be

```
instrumented_model = gangplank.PrometheusModel(keras_model)
```

If you also want to start an HTTP server that publishes metrics on port 9123, you would need code like

```
instrumented_model = gangplank.PrometheusModel(keras_model, port=9123)
```

## Performing Inference
Running `python3 predict.py` will predict the associated digits for a subset of 101 images in the
MNIST dataset.

```
Run
    curl -s http://localhost:8561/ | grep ^gangplank | grep -v created
to view metrics

Doing inference on images 0 - 1
Digits: [7 2]
Doing inference on images 2 - 3
Digits: [1 0]
Doing inference on images 4 - 11
Digits: [4 1 4 9 5 9 0 6]
Doing inference on images 12 - 18
Digits: [9 0 1 5 9 7 5]
Doing inference on images 19 - 20
Digits: [4 9]
Doing inference on images 21 - 21
Digits: [6]
Doing inference on images 22 - 28
Digits: [6 5 4 0 7 4 0]
Doing inference on images 29 - 29
Digits: [1]
Doing inference on images 30 - 34
Digits: [3 1 3 4 7]
Doing inference on images 35 - 44
Digits: [2 7 1 2 1 1 7 4 2 3]
Doing inference on images 45 - 50
Digits: [5 1 2 4 4 6]
Doing inference on images 51 - 51
Digits: [3]
Doing inference on images 52 - 60
Digits: [5 5 6 0 4 1 9 5 7]
Doing inference on images 61 - 67
Digits: [8 5 3 7 4 6 4]
Doing inference on images 68 - 74
Digits: [3 0 7 0 2 8 1]
Doing inference on images 75 - 83
Digits: [7 3 2 9 7 7 6 2 7]
Doing inference on images 84 - 91
Digits: [8 4 7 3 6 1 3 6]
Doing inference on images 92 - 100
Digits: [9 3 1 4 1 7 6 9 6]

Done. Sleeping; hit Ctrl-C to end...
```

The script does not exit on completion so that the `predict.py` process continues to expose metrics. The metrics can be queried by `curl`ing port
8561.

```
$ curl -s http://localhost:8561/ | grep ^gangplank | grep -v created
gangplank_predict_predict_total 101.0
gangplank_predict_predict_time_seconds_total 0.25597500801086426
gangplank_predict_call_total 0.0
gangplank_predict_call_time_seconds_total 0.0
```

The metrics show that 101 predictions were performed and that the total spent on inference was 0.256 seconds. The last two lines are both 0.0; if instead
of invoking `model.predict(x)`, we had invoked `model(x)`, then the metrics `gangplank_predict_call_total` and `gangplank_predict_call_time_seconds_total`
would have been incremented.