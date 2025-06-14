"""
This module provides the PrometheusModel class, a proxy for a Keras machine learning
model that enables automatic collection and exposition of Prometheus metrics of
inference operations. It tracks the number of predictions and model calls, as well as
the time spent performing these operations. Optionally, it can start a Prometheus HTTP
server for metrics exposition.

Classes:
    - Drift: A class that measures drift between observed data and the expected data
        distribution.
    - PrometheusModel: Proxies a Keras model to monitor prediction and call metrics
        for storing in Prometheus.

Dependencies:
    - prometheus_client
    - time
"""

import typing
import prometheus_client
import time


class Drift(typing.NamedTuple):
    """
    Represents the result of a drift detection test.

    Attributes:
        drift_detected (int, optional): A count of the number of drift incidents
            detected.
        p_value (float, optional): The p-value resulting from the test.
        test_statistic (float, optional): The test statistic value from the test.
    """

    drift_detected: int = None
    p_value: float = None
    test_statistic: float = None


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
            get_drift_metrics_func(Callable[[X: ndarray, Y: ndarray], Drift],
                    optional):
                A function that implements a drift detection algorithm. The function
                takes two arguments: Iterables of input data and the model's
                predictions. The function must returns a Drift object but all Drift
                attributes are optional.
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
        self.drift_p_gauge = None
        self.drift_ts_gauge = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _update_drift_counter(self, count):
        if count is None:
            return
        if self.drift_counter is None:
            self.drift_counter = prometheus_client.Counter(
                "gangplank_predict_drift_detected_total",
                "A count of drift detection incidents",
                registry=self.registry,
            )
        self.drift_counter.inc(count)

    def _update_drift_p_value(self, value):
        if value is None:
            return
        if self.drift_p_gauge is None:
            self.drift_p_gauge = prometheus_client.Gauge(
                "gangplank_predict_drift_p_value",
                "A p-value that quantifies the likelihood that drift has not occurred",
                registry=self.registry,
            )
        self.drift_p_gauge.set(value)

    def _update_drift_test_statistic(self, value):
        if value is None:
            return
        if self.drift_ts_gauge is None:
            self.drift_ts_gauge = prometheus_client.Gauge(
                "gangplank_predict_drift_test_statistic",
                "A measure of the distance between observed and expected data",
                registry=self.registry,
            )
        self.drift_ts_gauge.set(value)

    def predict(self, x, batch_size=32, verbose="auto", steps=None, callbacks=[]):
        start_time = time.time()

        try:
            y = self.model.predict(x, batch_size, verbose, steps, callbacks)
            self.predict_counter.inc(len(y))

            if self.get_drift_metrics_func is not None:
                drift = self.get_drift_metrics_func(x, y)
                self._update_drift_counter(drift.drift_detected)
                self._update_drift_p_value(drift.p_value)
                self._update_drift_test_statistic(drift.test_statistic)
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
