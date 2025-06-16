# Training/Testing a Model
## What is a Prometheus Pushgateway?
Prometheus pulls metrics from services or infrastructure at configured intervals. Batch jobs and other ephemeral processes are not well-suited to having
metrics pulled. To allow ephemeral processes to store metrics in Prometheus, the pushgateway was created; processes push metrics to the gateway and Prometheus scrapes
the gateway instead of the process. Machine learning training and testing jobs are ephemeral and, so, the idiomatic way for them to store metrics is for them to push metrics
(like loss, accuracy or mean absolute error) to a gateway.

## The Gangplank `TrainTestExporter` Class
A `TrainTestExporter` object pushes training and testing metrics to a pushgateway. The class's constructor
takes two mandatory arguments and four optional arguments:
 * `pgw_addr` is the address of the pushgateway (e.g. 127.0.0.1:9091).
 * `job` is a name to attach to the metrics.
 * `metrics` is an optional argument to specify which metrics to emit. If omitted, all available metrics are exported.
 * `histogram_buckets` is an optional list of `float`s to specify the buckets that the model's weights will be placed into (e.g. `[-0.3, -0.1, 0.1, 0.3]`). As a convenience, the constants
   `HISTOGRAM_WEIGHT_BUCKETS_1_0` and `HISTOGRAM_WEIGHT_BUCKETS_0_3` provide sensible choices for model weights in the intervals [-1.0, +1.0] and [-0.3, +0.3].
 * `handler` is an optional callback function that must be supplied if the pushgateway requires authentication; see [https://prometheus.github.io/client_python/exporting/pushgateway/](https://prometheus.github.io/client_python/exporting/pushgateway/)
 *  if the optional `ignore_exceptions` argument is `False`, the training or testing run will be aborted if the metrics can't be processed or pushed (e.g. the gateway is down).

An example instantiation of a `TrainTestExporter` would be

```python
callback = gangplank.TrainTestExporter("127.0.0.1:9091", "mnist", histogram_buckets=gangplank.HISTOGRAM_WEIGHT_BUCKETS_0_3)
```

## Training the Model
The [code](https://github.com/hammingweight/gangplank/blob/main/examples/train/train.py) to train the model instantiates a 
`gangplank.TrainTestExporter`

```python
gangplank.TrainTestExporter("127.0.0.1:9091", "mnist"),
```

that specifies the address of the Prometheus PGW and that the job name is "mnist".

You can run the training script by running `python3 train.py`. Once the first training epoch has finished, you should be able to retrieve some
metrics with the prefix `gangplank_train` from the PGW

```
$ curl -s http://localhost:9091/metrics | grep gangplank_train | grep -v '#' 
gangplank_train_accuracy{instance="",job="mnist"} 0.9448703527450562
gangplank_train_elapsed_time_seconds{instance="",job="mnist"} 53.397014141082764
gangplank_train_epochs_count{instance="",job="mnist"} 1
gangplank_train_loss{instance="",job="mnist"} 0.18397289514541626
gangplank_train_model_parameters_count{instance="",job="mnist"} 104202
gangplank_train_val_accuracy{instance="",job="mnist"} 0.9835000038146973
gangplank_train_val_loss{instance="",job="mnist"} 0.05415859818458557
```
The metrics include the training and validation loss and accuracy, the number of completed epochs, the running time and the number of weights (parameters) in the model.

The Prometheus server dashboard can be used to query or view the metrics. For example, the image shows that validation loss (`gangplank_train_val_loss`) reached a minimum at 09:38 (epoch 13) and the training
started to overfit the data after that.

![Training validation loss](./train_val_loss.png)


## Testing (Evaluating) the Model
The training code saves the best model to a file, "mnist_convnet.keras". The testing [code](https://github.com/hammingweight/gangplank/blob/main/examples/mnist/test.py)
loads the model and evaluates the model using the MNIST test data (10,000 samples). The code instantiates a `TrainTestExporter`

```python
callback = gangplank.TrainTestExporter(
    "http://localhost:9091",
    "mnist",
    histogram_buckets=gangplank.HISTOGRAM_WEIGHT_BUCKETS_0_3,
    ignore_exceptions=False,
)
```
to
 * Emit a histogram of model weights in buckets in the interval [-0.3, 0.3]
 * Abort the test run if an exception occurs (e.g. if the PGW is down)

To test the model, run `python test.py`.

The test/evaluation metrics are emitted with a `gangplank_test` prefix

```
$ curl -s http://localhost:9091/metrics | grep -v '#' | grep gangplank_test
gangplank_test_accuracy{instance="",job="mnist"} 0.9896000027656555
gangplank_test_elapsed_time_seconds{instance="",job="mnist"} 2.2292304039001465
gangplank_test_loss{instance="",job="mnist"} 0.041046421974897385
gangplank_test_model_parameters_count{instance="",job="mnist"} 104202
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.3"} 1122
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.25"} 2456
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.2"} 5280
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.15"} 10759
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.1"} 20734
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.05"} 36836
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0"} 58190
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.05"} 77984
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.1"} 91480
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.15"} 99129
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.2"} 102498
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.25"} 103643
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.3"} 104013
gangplank_test_model_weights_bucket{instance="",job="mnist",le="+Inf"} 104202
gangplank_test_model_weights_sum{instance="",job="mnist"} 0
gangplank_test_model_weights_count{instance="",job="mnist"} 104202
gangplank_test_model_weights_created{instance="",job="mnist"} 1.7488593562987614e+09
```

Some information that can be gleaned from the metrics is that:
 * The model accuracy is 98.96%
 * It took 2.23 seconds to evaluate the 10,000 test samples
 * There are 104,202 model weights
 * 1122 model weights have a value less than -0.3
