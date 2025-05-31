try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

import numbers
import sys
import time
import traceback
from prometheus_client import CollectorRegistry, Gauge, Histogram, push_to_gateway

HISTOGRAM_WEIGHT_BUCKETS_1_0 = [
    -1.0,
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
    1.0,
]

HISTOGRAM_WEIGHT_BUCKETS_0_3 = [
    -0.30,
    -0.25,
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


class TrainTestExporter(keras.callbacks.Callback):
    """
    A Keras Callback for exporting training and testing metrics to a Prometheus
    Pushgateway.

    This callback collects metrics during model training and evaluation, and pushes
    them to a Prometheus Pushgateway for monitoring and analysis. It supports both
    gauge and histogram metrics, and can be configured to handle exceptions gracefully.

    Args:
        pgw_addr (str): The address of the Prometheus Pushgateway.
        job (str): The job name under which metrics will be pushed.
        metrics (list, optional): List of metric names to export. If None, all numeric
                                    metrics from logs are exported.
        histogram_buckets (list, optional): Buckets for histogram metrics. If provided,
                                            model weights are exported as histograms.
        handler (callable, optional): Optional handler for push_to_gateway.
        ignore_exceptions (bool, optional): If True, exceptions during metric export
                                                are caught and printed; otherwise, they
                                                are raised.

    Attributes:
        pgw_addr (str): The Pushgateway address.
        job (str): The job name for metrics.
        metrics (list): Metrics to export.
        histogram_buckets (list): Histogram bucket boundaries.
        handler (callable): Optional handler for push_to_gateway.
        ignore_exceptions (bool): Whether to ignore exceptions during export.
        registry (CollectorRegistry): Prometheus registry for metrics.
        gauges (dict): Dictionary of Prometheus Gauge objects.
        is_done (bool): Indicates if the callback has completed a run.
        is_training (bool): True if training has started; is False for a test run.

    Methods:
        exception_handler(func): Decorator to handle exceptions in callback methods.
        _get_metrics(logs): Returns the list of metrics to export based on logs and
                            configuration.
        _get_gauge(name, desc): Retrieves or creates a Prometheus Gauge metric.
        _push_to_gateway(): Pushes collected metrics to the Pushgateway.
        _construct_histogram(name): Constructs and populates a Prometheus Histogram
                                    for model weights.
        on_test_begin(logs): Called at the start of model evaluation.
        on_test_end(logs): Called at the end of model evaluation; exports test metrics.
        on_train_begin(logs): Called at the start of model training.
        on_epoch_end(epoch, logs): Called at the end of each training epoch; exports
                                    training metrics.
        on_train_end(logs): Called at the end of model training; exports final metrics
                            and histograms.

    Raises:
        RuntimeError: If the callback is reused for a new run after completion.
    """

    def __init__(
        self,
        pgw_addr,
        job,
        metrics=None,
        histogram_buckets=None,
        handler=None,
        ignore_exceptions=True,
    ):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.metrics = metrics
        self.histogram_buckets = histogram_buckets
        self.handler = handler
        self.ignore_exceptions = ignore_exceptions
        self.registry = CollectorRegistry()
        self.gauges = {}
        self.is_done = False
        # We need to distinguish between training and testing.
        # We'll set this to True if on_training_start is called.
        self.is_training = False

    @staticmethod
    def exception_handler(func):
        def wrapper_func(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if self.ignore_exceptions:
                    traceback.print_exc(file=sys.stderr)
                else:
                    raise e

        return wrapper_func

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

    def _push_to_gateway(self):
        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)

    def _construct_histogram(self, name):
        histogram = Histogram(
            name,
            "model trainable weights",
            buckets=self.histogram_buckets,
            registry=self.registry,
        )
        for layer in self.model.layers:
            if not layer.trainable:
                continue
            weights = layer.get_weights()
            for weight in weights:
                weight = weight.flatten()
                for w in weight:
                    histogram.observe(w)

    @exception_handler
    def on_test_begin(self, logs):
        if self.is_done:
            raise RuntimeError("cannot reuse this callback for a new run.")

    @exception_handler
    def on_test_end(self, logs):
        if self.is_training:
            return

        self.is_done = True

        metrics = self._get_metrics(logs)
        for k in metrics:
            v = logs.get(k)
            if v is not None:
                gauge = self._get_gauge("gangplank_test" + k, k)
                gauge.set(v)

        gauge = self._get_gauge(
            "gangplank_test_model_parameters_count",
            "the number of trainable and non-trainable model weights",
        )
        gauge.set(self.model.count_params())

        if self.histogram_buckets:
            self._construct_histogram("gangplank_test_model_weights")

        self._push_to_gateway()

    @exception_handler
    def on_train_begin(self, logs):
        if self.is_done:
            raise RuntimeError("cannot reuse this callback for a new run.")

        self.is_training = True
        self.start_time = time.time()

    @exception_handler
    def on_epoch_end(self, epoch, logs):
        metrics = self._get_metrics(logs)
        for k in metrics:
            v = logs.get(k)
            if v is not None:
                gauge = self._get_gauge("gangplank_train_" + k, k)
                gauge.set(v)

        gauge = self._get_gauge(
            "gangplank_train_test_model_parameters_count",
            "the number of trainable and not trainable model weights",
        )
        gauge.set(self.model.count_params())

        gauge = self._get_gauge(
            "gangplank_train_epochs_count", "the number of completed training epochs"
        )
        gauge.set(epoch + 1)

        gauge = self._get_gauge(
            "gangplank_train_elapsed_time_seconds",
            "the amount of time spent training the model",
        )
        gauge.set(time.time() - self.start_time)

        self._push_to_gateway()

    @exception_handler
    def on_train_end(self, logs):
        self.is_done = True

        if not self.histogram_buckets:
            return

        self._construct_histogram("gangplank_train_model_weights")

        self._push_to_gateway()
