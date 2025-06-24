# Testing a Model
The example in this directory assumes that you have run [`train.py`](../train/train.py) to create
a Keras model.

## The `test.py` Script
The `TrainTestExporter` class can be used for exporting both training and testing metrics. In the [test.py](./test.py) script,
a `TrainTestExporter` object is instantiated as

```python
callback = gangplank.TrainTestExporter(
    "http://localhost:9091",
    "mnist",
    histogram_buckets=gangplank.HISTOGRAM_WEIGHT_BUCKETS_0_3,
    ignore_exceptions=False,
)
```

This instantiation has two more arguments than were passed to the constructor of the [training script](../train/train.py).
The `histogram_buckets` argument specifies the buckets that the model's weights will be placed into. The buckets are in the interval [-0.3, +0.3].
The `ignore_exceptions` = False argument will cause model testing to fail if the Gangplank callback throws an exception. For example, if the pushgateway is inaccessible, the `TrainTestExporter` will throw an exception rather than just printing the problem to `stderr`.

The training script saved a model to a file, "mnist_convnet.keras"; the training script loads the model from disk and then
evaluates the model using the MNIST test images

```python
model.evaluate(test_images, test_labels, callbacks=[callback])
```

## Running the `test.py` Script
Running `python test.py` will evaluate the model and push test metrics to the pushgateway.
The test/evaluation metrics are emitted with a `gangplank_test` prefix

```
$ curl -s http://localhost:9091/metrics | grep -v '#' | grep gangplank_test
gangplank_test_accuracy{instance="",job="mnist"} 0.9901000261306763
gangplank_test_elapsed_time_seconds{instance="",job="mnist"} 1.7885098457336426
gangplank_test_loss{instance="",job="mnist"} 0.03201884403824806
gangplank_test_model_parameters_count{instance="",job="mnist"} 104202
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.3"} 279
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.25"} 728
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.2"} 1969
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.15"} 5192
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.1"} 12984
gangplank_test_model_weights_bucket{instance="",job="mnist",le="-0.05"} 30828
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0"} 57258
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.05"} 82053
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.1"} 96056
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.15"} 101798
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.2"} 103518
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.25"} 104010
gangplank_test_model_weights_bucket{instance="",job="mnist",le="0.3"} 104147
gangplank_test_model_weights_bucket{instance="",job="mnist",le="+Inf"} 104202
gangplank_test_model_weights_sum{instance="",job="mnist"} 0
gangplank_test_model_weights_count{instance="",job="mnist"} 104202
gangplank_test_model_weights_created{instance="",job="mnist"} 1.7507642169909902e+09
```

Some information that can be gleaned from the metrics is that:
 * The model accuracy is 99.01%
 * It took 1.79 seconds to evaluate the 10,000 test samples
 * There are 104,202 model weights
 * 279 model weights have a value less than -0.3

The same metrics can be queried via the [Prometheus dashboard](http://localhost:9090) 