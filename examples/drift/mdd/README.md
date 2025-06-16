# Drift Detection using Alibi Detect
## Alibi Detect
Alibi detect is a Python library that provides drift detection tests. Alibi's tests can be classified as online (realtime)
or batch (offline). In this example, we'll use the online MDD (maximum mean discrepancy) test, `MMDDriftOnline`. 
The `alibi-detect` and `pytorch` libraries must be installed to run this drift detection code

```
$ pip install alibi-detect[torch]
```

## Configuring an `MMDDriftOnline` Drift Detector
To compare observed and expected distributions, it is necessary to determine the expected distribution. In this example, we want
to check for *prediction* drift, so we need to determine the distribution of the predictions for the training data. That's achieved
by the following lines in the [drift.py](./drift.py) script.

```
preds = model.predict(train_images[:20000], verbose=0)
drift_detector = cd.MMDDriftOnline(
    preds, ert=500, window_size=200, backend="pytorch", n_bootstraps=5000
)
```

where `preds` are the predictions for 20,000 training images from the MNIST dataset. The `window_size=200` argument is used to
specify that a rolling window of 200 predicted values must be used to detect whether drift has occurred.

**Note:** This example is artificial since we know that the expected distribution should be a uniform distribution of the values 0 to 9. `alibi-detect` provides
heavier machinery than is actually needed. The [chi-square](../chi_square) example uses more elementary statistics than is used here.


## A `get_drift_metrics_func` for the MMD Test

## Running the `drift.py` Script

```
$ KERAS_BACKEND=torch python drift.py 
No GPU detected, fall back on CPU.
Generating permutations of kernel matrix..
100%|███████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [53:04<00:00,  1.57it/s]
Computing thresholds: 100%|███████████████████████████████████████████████████████████████████████| 200/200 [40:13<00:00, 12.07s/it]
Drift detector created. Press Enter to continue...

Predictions without drift...
```

For 15 minutes the script runs without inducing any drift, however if we plot the `gangplank_predict_drift_detected_total` we see that drift incidents are
being reported

![drift_detected counts](./gp_drift_total.png)

That's not surprising since a statistical test can report false positives. Better than simply reporting the total number of drift incidents, is to look at the ratio of drift incidents to the number of predictions

![drift detected ratio](gp_drift_rate.png)

We can see that about 7% of predictions are reported as "drift detected" incidents.

After 15 minutes, the script performs inference on images but discards any images of the digit 0. Unsurprisingly, the ratio of "drift detected" incidents climbs from 7% to 100%

![drift for real](./gp_drift_detected.png)

The `get_drift_metrics` function returns not only a count of "drift detected" incidents but also a test statistic showing the distance between samples and the
expected distribution.

![test statistic](./gp_drift_test_statistic.png)

The plot of the test statistic shows that its value increases from close to 0 to about 0.007 when drift is induced.
