# Examples
## Running Prometheus and the Pushgateway (PGW)
The Gangplank examples require that you run a Prometheus server bound to `127.0.0.1:9090` that can scrape a gateway that is
bound to `127.0.0.1:9091`. If you've installed Docker, you can run

```
./start_prometheus.sh
```

to start both the server and the gateway

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

