# gangplank
## Export Keras training metrics to Prometheus
[Prometheus](https://prometheus.io/) is a monitoring system that pulls metrics from applications and infrastructure.
While polling works for applications that are continuously running, scraping metrics does not work well with batch jobs such as
machine learning training or evaluation jobs. The Prometheus [Pushgateway](https://prometheus.io/docs/instrumenting/pushing/)
is middleware that connects batch jobs to Prometheus.

Gangplank is a Keras [callback](https://keras.io/api/callbacks/) for pushing Keras training and testing metrics to Prometheus via a
pushgateway.

### What metrics are exported?




