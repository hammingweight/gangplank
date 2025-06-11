"""
This module provides the PrometheusModel class, a proxy for a Keras machine learning
model that enables automatic collection and exposition of Prometheus metrics of
inference operations. It tracks the number of predictions and model calls, as well as
the time spent performing these operations. Optionally, it can start a Prometheus HTTP
server for metrics exposition.

Classes:
    - PrometheusModel: Proxies a Keras model to monitor prediction and call metrics
        for storing in Prometheus.

Dependencies:
    - prometheus_client
    - time
"""

import prometheus_client
import time


class PrometheusModel:
    def __init__(
        self,
        model,
        registry=prometheus_client.REGISTRY,
        port=None,
        get_drift_metrics_func=None,
    ):
        """
        Initializes the PrometheusModel with a Keras model to proxy.

        Args:
            model: The machine learning model to be proxied and monitored.
            registry (prometheus_client.CollectorRegistry, optional): The Prometheus
                registry to use for metrics. Defaults to prometheus_client.REGISTRY.
            port (int, optional): If provided, starts a Prometheus HTTP server on the
                specified port for metrics exposition.
            get_drift_metrics_func(Callable[[X: ndarray, Y: ndarray], (float, float)],
                    optional):
                A function that implements a drift detection algorithm. The function
                takes two arguments: Iterables of input data and the model's
                predictions. The function returns two values: (1) A count of the number
                of drift incidents found in the input/output data and (2) a statistical
                value for the drift (e.g. a p-value or mean maximum distance). Either
                or both values can be None, in which case the associated Prometheus
                metric is not updated.
        """
        self.model = model
        self.registry = registry
        self.predict_counter = prometheus_client.Counter(
            "gangplank_predict_total",
            "The number of model predictions",
            registry=registry,
        )
        self.predict_time = prometheus_client.Counter(
            "gangplank_predict_time_seconds",
            "The amount of time spent in the predict method",
            registry=registry,
        )
        self.call_counter = prometheus_client.Counter(
            "gangplank_predict_call_total",
            "The number of __call__ invocations",
            registry=registry,
        )
        self.call_time = prometheus_client.Counter(
            "gangplank_predict_call_time_seconds",
            "The amount of time spent in the __call__ method",
            registry=registry,
        )
        if port is not None:
            prometheus_client.start_http_server(port, registry=registry)
        self.get_drift_metrics_func = get_drift_metrics_func
        self.drift_counter = None
        self.drift_gauge = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _update_drift_counter(self, count):
        if count is None:
            return
        if self.drift_counter is None:
            self.drift_counter = prometheus_client.Counter(
                "gangplank_predict_drift_incidents_total",
                "A count of drift detection incidents",
                registry=self.registry,
            )
        self.drift_counter.inc(count)

    def _update_drift_value(self, value):
        if value is None:
            return
        if self.drift_gauge is None:
            self.drift_gauge = prometheus_client.Gauge(
                "gangplank_predict_drift_statistical_value",
                "A statistical measure of drift",
                registry=self.registry,
            )
        self.drift_gauge.set(value)

    def predict(self, x, batch_size=32, verbose="auto", steps=None, callbacks=[]):
        start_time = time.time()

        try:
            y = self.model.predict(x, batch_size, verbose, steps, callbacks)
            self.predict_counter.inc(len(y))

            if self.get_drift_metrics_func is not None:
                (count, value) = self.get_drift_metrics_func(x, y)
                self._update_drift_counter(count)
                self._update_drift_value(value)
            return y
        finally:
            self.predict_time.inc(time.time() - start_time)

    def __call__(self, *args, **kwds):
        start_time = time.time()
        try:
            res = self.model.__call__(*args, **kwds)
            self.call_counter.inc(len(res))
            return res
        finally:
            self.call_time.inc(time.time() - start_time)
