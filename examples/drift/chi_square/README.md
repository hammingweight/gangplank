# Drift Detection using SciPy Stats
## The Chi-square Test
The MNIST dataset has 70,000 images of the digits 0 to 9 with 7,000 images of each digit. We trained a model with a
uniform distribution of the digits under the assumption that all digits are equally likely. If, in production, we
see our model predicting a non-uniform distribution, we should investigate.

The chi-square test is a popular choice for testing whether observations differ significantly from an expected
distribution. The test returns a *p*-value in the range [0, 1] where a small value (e.g. <0.05) would be suggestive
that the observed data is different from the assumed distribution. A small *p*-value would suggest that data
in production has drifted from the data that was used in training.

## A `get_drift_metrics_func` for the chi-square test
The following function is used by [drift.py](./drift.py) to return a drift metric
```
def get_drift_metrics(_X, Y):
    if len(Y) < 50:
        return gangplank.Drift()
    buckets = [0] * 10
    for y in Y:
        buckets[y.argmax()] += 1
    res = stats.chisquare(buckets)
    return gangplank.Drift(p_value=res.pvalue, test_statistic=res.statistic)
```

Note that no metrics are returned if the number of predictions is less than 50 (since the chi-square test is of little value for small samples). If there sufficient samples, both the *p*-value
and the test statistic are returned.
