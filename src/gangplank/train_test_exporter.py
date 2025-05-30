try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

import numbers
import time
from prometheus_client import CollectorRegistry, Gauge, Histogram, push_to_gateway

GANGPLANK_HISTOGRAM_BUCKETS = [
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


class TrainTestExporter(keras.callbacks.Callback):
    def __init__(
        self, pgw_addr, job, metrics=None, histogram_buckets=None, handler=None
    ):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.metrics = metrics
        self.histogram_buckets = histogram_buckets
        self.handler = handler
        self.registry = CollectorRegistry()
        self.gauges = {}
        # We need to distinguish between training and testing.
        # We'll set this to True if on_training_start is called.
        self.is_training = False

    def _get_metrics(self, logs):
        if self.metrics is not None:
            return self.metrics
        metrics = []
        for k, v in logs.items():
            if isinstance(v, numbers.Number):
                metrics.append(k)

        return metrics

    def _get_gauge(self, name, desc):
        if not self.gauges.get(name):
            self.gauges[name] = Gauge(name, desc, registry=self.registry)
        return self.gauges[name]

    def _push_weight_histogram(self, name):
        histogram = Histogram(
            name,
            "model weights/parameters",
            buckets=self.histogram_buckets,
            registry=self.registry,
        )
        histogram.observe(0)

    def on_test_end(self, logs):
        if self.is_training:
            return

        metrics = self._get_metrics(logs)
        for k in metrics:
            v = logs.get(k)
            if v is not None:
                gauge = self._get_gauge("gangplank_test_" + k, k)
                gauge.set(v)

        gauge = self._get_gauge(
            "gangplank_test_model_parameter_count",
            "the number of model parameters/weights",
        )
        gauge.set(self.model.count_params())

        if self.histogram_buckets:
            self._push_weight_histogram("gangplank_test_model_weights")

        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)

    def on_train_begin(self, logs):
        self.is_training = True
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs):
        metrics = self._get_metrics(logs)
        for k in metrics:
            v = logs.get(k)
            if v is not None:
                gauge = self._get_gauge("gangplank_train_" + k, k)
                gauge.set(v)

        # "total" is a suffix should be used with Counters not Gauges.
        # We need a Counter but we want to set its value rather than increment
        # it, so we're using a Gauge.
        gauge = self._get_gauge(
            "gangplank_train_epoch_total", "the number of completed training epochs"
        )
        gauge.set(epoch + 1)

        gauge = self._get_gauge(
            "gangplank_train_model_parameter_count",
            "the number of model parameters/weights",
        )
        gauge.set(self.model.count_params())

        gauge = self._get_gauge(
            "gangplank_train_elapsed_time_seconds",
            "the amount of time spent training the model",
        )
        gauge.set(time.time() - self.start_time)

        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)

    def on_train_end(self, logs):
        if not self.histogram_buckets:
            return

        self._push_weight_histogram("gangplank_train_model_weights")

        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)
