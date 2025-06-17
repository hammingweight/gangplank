# Examples
## Running Prometheus and a Pushgateway (PGW) Locally
The Gangplank examples require that you run a Prometheus server bound to `127.0.0.1:9090` that can scrape a gateway that is
bound to `127.0.0.1:9091`. If you've installed [Docker](https://www.docker.com/), you can run `./start_prometheus.sh` to start a server, a gateway and
a Prometheus alertmanager

```
$ ./start_prometheus.sh 
f8cccd7201dab455523450546629064df0a9ac590da997718de86fb3c8059cdd
1677ef1c098562999cc5160a52aab5f568256f7df6b05bb208129e7c7e25329f
977eee5dc5149d9f2cc5a757df3ded2cce393282062c4597e2b8eb8bfdd25ce3
CONTAINER ID   IMAGE               COMMAND                  CREATED          STATUS          PORTS     NAMES
977eee5dc514   prom/alertmanager   "/bin/alertmanager -…"   10 seconds ago   Up 10 seconds             alertmanager
1677ef1c0985   prom/pushgateway    "/bin/pushgateway"       11 seconds ago   Up 10 seconds             pgw
f8cccd7201da   prom/prometheus     "/bin/prometheus --c…"   11 seconds ago   Up 11 seconds             prometheus
```

Running `curl http://127.0.0.1:9090/metrics`, `curl http://127.0.0.1:9091/metrics` and `curl http://127.0.0.1:9093/metrics` should return the metrics
exposed by the server, the gateway and the alertmanager.

Looking at the [prometheus.yml](./prometheus/prometheus.yml) configuration file shows that Prometheus is configured to collect metrics from itself, the pushgateway and
the alertmanager. Opening [`http://localhost:9090/targets`](http://localhost:9090/targets) will show that those targets are up. You'll also see that Prometheus is
trying to scrape metrics from a service running on port 8561 but that the service ("mnist") is down. The "mnist" service can be started by running the
[prediction/inference](https://github.com/hammingweight/gangplank/tree/main/examples/predict) example code.

![Target health](./targets.png)


## Gangplank Metrics
Metrics stored in Prometheus have names that are (usually) in snake case and prefixed with a library name. Gangplank metrics are exposed with a `gangplank_` prefix:
 * `gangplank_train_` for training metrics
 * `gangplank_test_` for testing/evaluation metrics
 * `gangplank_predict_` for inference/prediction metrics.


## MNIST Dataset and Keras Model
The examples use a convolutional neural network (CNN) to classify handwritten digits from an MNIST dataset. The dataset consists of 70,000 black and white images, of 28 x 28 pixel resolution, where each image is of a single digit from 0 to 9.

The CNN model is from one of François Chollet's [Jupyter notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter08_intro-to-dl-for-computer-vision.ipynb)
that accompany his book ["Deep Learning with Python, second edition"]([https://www.manning.com/books/deep-learning-with-python](https://www.manning.com/books/deep-learning-with-python-second-edition)).

The inputs to the CNN are batches of tensors of shape (28, 28, 1) and the outputs from the CNN are one-hot encodings of the values 0 to 9; i.e. if an image can be categorized confidently as one of the decimal digits, then one of the CNN's ten outputs will have a value close to one while the other nine outputs will have a value close to zero.


## Usage Examples
[Training and Testing](https://github.com/hammingweight/gangplank/tree/main/examples/train)

[Prediction/Inference](https://github.com/hammingweight/gangplank/tree/main/examples/predict)
