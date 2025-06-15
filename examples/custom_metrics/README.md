# Custom Training Metrics
Keras callbacks have access to a dictionary whose entries can be populated or
retrieved by all the callbacks. The [example](./train.py) script in this directory
adds a callback that stores the optimizer's learning rate under the key "learning_rate".
The Gangplank callback reads the metric and exposes it to Prometheus (the Gangplank
exporter assumes that any numeric values in the dictionary are metrics that should
be pushed to Prometheus.)

If you run `python train.py` and wait for the first training epoch to complete,
you can then get the learning rate metric from the pushgateway

```
$ curl -s http://localhost:9091/metrics | grep gangplank_train_learning_rate
# HELP gangplank_train_learning_rate learning_rate
# TYPE gangplank_train_learning_rate gauge
gangplank_train_learning_rate{instance="",job="mnist"} 0.004999999888241291
```

The metric can also be seen in the [Prometheus QueryExplorer](http://localhost:9090/query?g0.expr=gangplank_train_learning_rate&g0.show_tree=0&g0.tab=table&g0.range_input=1h&g0.res_type=auto&g0.res_density=medium&g0.display_mode=lines&g0.show_exemplars=0)
