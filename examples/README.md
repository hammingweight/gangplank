# Examples
## Running Prometheus and a Pushgateway (PGW) Locally
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

The training/testing examples push metrics to the PGW and the Prometheus server scrapes the metrics from the gateway.

The Prometheus server will also try to collect metrics from `127.0.0.1:8561`. The "predict" examples run a model for inference that starts an HTTP server that exposes
the inference metrics (on port 8561). 

## MNIST Dataset
The examples use a convolutional neural network (CNN) to classify the handwritten digits in the MNIST database.
The CNN model is from one of François Chollet's [Jupyter notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb)
that accompany his book ["Deep Learning with Python"]([https://www.manning.com/books/deep-learning-with-python](https://www.manning.com/books/deep-learning-with-python-second-edition)).

The examples instrument the model to push metrics to the Prometheus Pushgateway during training/testing and to publish metrics on port 8561 during inference.


## Usage examples
[Training and Testing](https://github.com/hammingweight/gangplank/tree/main/examples/train)

[Prediction/Inference](https://github.com/hammingweight/gangplank/tree/main/examples/predict)

