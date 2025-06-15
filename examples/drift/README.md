# Drift Metrics
## Drift
Over time, the statistical properties of the data in production may drift from the data that was used in training. Drift may take several forms:
 * Data drift
 * Prediction drift (or label drift when the ML system produces categorical data)
 * Concept drift

There are also several ways to expose drift as a metric:
 * A *p*-value (i.e. a probability that quantifies the probability of observing data)
 * A test statistic (a measure of the "distance" between an observation and the expected distribution)
 * A count of the number of times that input or output data drifted from the expected distribution

Gangplank is agnostic about the form of drift and the metric and the metric used to quantify the drift; it simply provides a means of exposing the
metric to Prometheus.

## The `Drift` Class
A `Drift` object must be created to report drift

```
class Drift(typing.NamedTuple):
    drift_detected: int = None
    p_value: float = None
    test_statistic: float = None
```

The `drift_detected` value is a count of the number of occurrences of drift seen in a batch of predictions. All three values are optional. For example, setting `drift_detected` to `1` and `p_value` to 0.0001 while leaving the `test_statistic` value unset is legitimate. You can also set only one value or even none of them.

Gangplank uses the following names to expose the metrics:
 * `gangplank_predict_drift_detected_total`
 * `gangplank_predict_drift_p_value`
 * `gangplank_predict_drift_test_statistic`

Note that `gangplank_predict_drift_detected_total` is a Prometheus `Counter`; if the counter has a value of 7 and a `Drift` object returns the value 2, the counter will be increased to 9.
