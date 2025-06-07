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
    def __init__(self, model, registry=prometheus_client.REGISTRY, port=None):
        """
        Initializes the PrometheusModel with the given model and Prometheus metrics.

        Args:
            model: The machine learning model to be proxied and monitored.
            registry (prometheus_client.CollectorRegistry, optional): The Prometheus
                registry to use for metrics. Defaults to prometheus_client.REGISTRY.
            port (int, optional): If provided, starts a Prometheus HTTP server on the
                specified port for metrics exposition.
        """
        self.model = model
        self.registry = registry
        self.predict_counter = prometheus_client.Counter(
            "gangplank_predict_predict_total",
            "The number of model predictions",
            registry=registry,
        )
        self.predict_time = prometheus_client.Counter(
            "gangplank_predict_predict_time_seconds",
            "The amount of time spent in the prediction method",
            registry=registry,
        )
        self.call_counter = prometheus_client.Counter(
            "gangplank_predict_call_total",
            "The number of model calls",
            registry=registry,
        )
        self.call_time = prometheus_client.Counter(
            "gangplank_predict_call_time_seconds",
            "The amount of time spent in the __call__ emthod",
            registry=registry,
        )
        if port is not None:
            prometheus_client.start_http_server(port, registry=registry)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, x, batch_size=32, verbose="auto", steps=None, callbacks=[]):
        start_time = time.time()
        try:
            res = self.model.predict(x, batch_size, verbose, steps, callbacks)
            self.predict_counter.inc(len(res))
            return res
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
