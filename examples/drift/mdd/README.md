# Drift Detection using Alibi Detect
## Alibi Detect
Alibi detect is a Python library that provides drift detection tests. The tests can be classified as online (realtime)
or batch (offline). In this example, we'll use the online MDD (maximum mean discrepancy) test, `MMDDriftOnline`. 
The example needs both the alibi_detect and pytorch libraries.

```
$ pip install alibi-detect[torch]
```

## Configuring `MMDDriftOnline`

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