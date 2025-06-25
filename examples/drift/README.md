# Drift Metrics
## Drift
Over time, the statistical properties of the data in production may drift from the data that was used in training. Drift may take several forms:
 * Data drift; the distribution of input data differs from the input data used for training.
 * Prediction drift; the statistics of output data has drifted from the output seen in training. Prediction drift is also known as label drift when the ML system produces categorical data.
 * Concept drift; the relationship beteen input and output data has changed since the model has trained.

There are also several ways to expose drift as a metric:
 * A *p*-value (i.e. a probability that quantifies the probability of observing data under an assumption that the data conforms to some statistical distribution)
 * A test statistic (a measure of the "distance" between an observation and the expected distribution)
 * A count of the number of times that input or output data drifted from the expected distribution

Gangplank is agnostic about the form of drift and the metric used to quantify the drift; it simply provides a means of exposing the
metric to Prometheus.

## The `Drift` Class
A `Drift` object must be created to report drift

```python
class Drift(typing.NamedTuple):
    drift_detected: int = None
    p_value: float = None
    test_statistic: float = None
```
The first attribute (`drift_detected`) is a count of the number of observed drift occurrences while the meaning of the other two attributes should be obvious.

All three class attributes are optional. For example, setting `drift_detected` to `1` and `p_value` to 0.0001 while leaving the `test_statistic` value unset is legitimate. You can also set only one value or even none of them.

Gangplank uses the following names to expose the metrics:
 * `gangplank_predict_drift_detected_total`
 * `gangplank_predict_drift_p_value`
 * `gangplank_predict_drift_test_statistic`

Note that `gangplank_predict_drift_detected_total` is a Prometheus `Counter`; if the counter has a value of 7 and a `Drift` object returns that `drift_detected` has the value 2, the counter will be increased to 9. Both `gangplank_predict_drift_p_value` and `gangplank_predict_drift_test_statistic` are Prometheus `Gauge`s.

## The `get_drift_metrics_func` Callback
Gangplank has no way to measure drift for an arbitrary model so, if you want to emit drift metrics, you need to provide a callback function that returns a
`Drift` object. The function must expect two arguments:
 * `X`: a batch of inputs to the model
 * `Y`: a NumPy array of the associated predictions for the input data

A minimal implementation of a callback function would be

```python
def null_drift_detector(X, Y):
    return Drift()
```

The example code is more instructive.

## Examples
### Measuring Drift with Alibi Detect
The [MDD example](./mdd/) uses the MMD Online drift detector from the `alibi detect` package to measure drift.

### Measuring Drift With SciPy Stats
The [chi_square example](./chi_square/) uses the `scipy.stats` module's `chisquare` function to measure drift.
This example also illustrates how to create alerts in Prometheus.
