import random
try:
    import keras
except ModuleNotFoundError:
    import tensorflow.keras as keras

from prometheus_client import CollectorRegistry, Gauge, Histogram, pushadd_to_gateway

class EvaluationExporter(keras.callbacks.Callback):
    def __init__(self, pgw_addr="localhost:9091", job="gangplank"):
        super().__init__()
        self.pgw_addr = pgw_addr
        self.job = job
        self.registry = CollectorRegistry()
        self.gauge_loss = Gauge("gangplank_loss", "Evaluation loss", registry=self.registry)
        self.gauge_accuracy = Gauge("gangplank_accuracy", "Evaluation accuracy", registry=self.registry)
        
    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        for (k, v) in logs.items():
            print(f"{k}:{v} ({type(v)})")
        self.gauge_accuracy.set(logs["accuracy"])
        self.gauge_loss.set(logs["loss"])

        h = Histogram("gangplank_foo", "The foo histogram", buckets=[0.1, 0.2, 0.5], registry=self.registry)
        for _ in range(10000):
            h.observe(random.random()*2)
        pushadd_to_gateway(self.pgw_addr, self.job, self.registry)

        

