try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

import numbers
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class TrainTestExporter(keras.callbacks.Callback):
    def __init__(self, pgw_addr, job, metrics=None):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.metrics = metrics
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
            gauge = self._get_gauge("gangplank_test_" + k, k)
            gauge.set(logs[k])

        push_to_gateway(self.pgw_addr, self.job, self.registry)

    def on_epoch_end(self, epoch, logs):
        metrics = self._get_metrics(logs)
        for k in metrics:
            gauge = self._get_gauge("gangplank_train_" + k, k)
            gauge.set(logs[k])

        push_to_gateway(self.pgw_addr, self.job, self.registry)
