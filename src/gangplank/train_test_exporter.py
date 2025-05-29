try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

import numbers
from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway


class TrainTestExporter(keras.callbacks.Callback):
    def __init__(self, pgw_addr, job, metrics=None):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.metrics = metrics
        self.registry = CollectorRegistry()

    def _get_metrics(self, logs):
        if self.metrics is not None:
            return self.metrics
        metrics = []
        for k, v in logs.items():
            if isinstance(v, numbers.Number):
                metrics.append(k)

        return metrics

    def on_test_end(self, logs):
        metrics = self._get_metrics(logs)
        for k in metrics:
            gauge = Gauge("gangplank_" + k, k, registry=self.registry)
            gauge.set(logs[k])

        pushadd_to_gateway(self.pgw_addr, self.job, self.registry)
