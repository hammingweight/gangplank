try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

import numbers
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class TrainTestExporter(keras.callbacks.Callback):
    def __init__(self, pgw_addr, job, metrics=None, handler=None):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.metrics = metrics
        self.handler = handler
        self.registry = CollectorRegistry()
        self.gauges = {}

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

    def on_test_end(self, logs):
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

        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)

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

        if self.handler:
            push_to_gateway(
                self.pgw_addr, self.job, self.registry, handler=self.handler
            )
        else:
            push_to_gateway(self.pgw_addr, self.job, self.registry)
