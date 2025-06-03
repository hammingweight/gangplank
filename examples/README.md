# Examples
## Running Prometheus and the Pushgateway (PGW)
The Gangplank examples require that you run a Prometheus server bound to `127.0.0.1:9090` that can scrape a gateway that is
bound to `127.0.0.1:9091`. If you've installed Docker, you can run `./start_prometheus.sh` to start both the server and the gateway

```
$ ./start_prometheus.sh 
c3b0e9147f1506a27021502be0d3d1cb2cdf730c59396bbc33f675221b7f4f6a
210a2869ed768497a02729428cf0de3347a86b25a9b3dd3d52b01013d7a74c4f
CONTAINER ID   IMAGE              COMMAND                  CREATED          STATUS          PORTS     NAMES
210a2869ed76   prom/pushgateway   "/bin/pushgateway"       11 seconds ago   Up 10 seconds             pgw
c3b0e9147f15   prom/prometheus    "/bin/prometheus --c…"   11 seconds ago   Up 10 seconds             prometheus
```

Running `curl http://127.0.0.1:9090/metrics` and `curl http://127.0.0.1:9091/metrics` should return the metrics exposed by the
server and the gateway.

## The `TrainTestExporter` Class
The `TrainTestExporter` class extends `keras.callbacks.Callback` to push metrics to a pushgateway. The class's constructor
takes two mandatory arguments and four optional arguments:
 * `pgw_addr` is the address of the pushgateway (e.g. 127.0.0.1:9091)
 * `job` is a name to attach to the metrics
 * `metrics` is an optional argument to specify which metrics to emit. If omitted, all available metrics are exported
 * `histogram_buckets` is an optional list of `float`s to specify the buckets that the model's weights will be placed into (e.g. `[-0.3, -0.1, 0.1, 0.3]`). As a convenience, the constants
   `HISTOGRAM_WEIGHT_BUCKETS_1_0` and `HISTOGRAM_WEIGHT_BUCKETS_0_3` provide sensible choices for model weights in the intervals [-1.0, +1.0] and [-0.3, +0.3].
 * `handler` is an optional callback function that must be supplied if the pushgateway requires authentication; see [https://prometheus.github.io/client_python/exporting/pushgateway/](https://prometheus.github.io/client_python/exporting/pushgateway/)
 *  if the optional `ignore_exceptions` argument is `True`, the training or testing run will be aborted if the metrics can't be processed or pushed (e.g. the gateway is down.)

## Usage examples
[MNIST handwritten digit recognition](https://github.com/hammingweight/gangplank/tree/main/examples/mnist)

