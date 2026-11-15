# NO₂ level prediction: how the evaluation protocol changes the conclusions

A small study on the UCI Air Quality dataset. A model that looks good on randomly
held-out hours performs much worse on later months, and its errors change character.
The repository contains the code behind that comparison.

## Task

Predict the **NO₂ concentration level** (`NO2(GT)`), discretised into five
equal-frequency classes, from the other sensors and weather variables. This is
not the Air Quality Index. Class edges are computed on the training part only.

Two kinds of inputs exist in the data:

- **Reference analyzers** (`CO(GT)`, `C6H6(GT)`, `NOx(GT)`): expensive, accurate measurements.
- **Low-cost sensors and weather** (`PT08.S1…S5`, `T`, `RH`, `AH`): the realistic input
  for a cheap monitoring device.

## What is compared

| Dimension | Options |
|---|---|
| Split | `random` hours · `block` (random whole weeks) · `chrono` (last 20% of the timeline is the test set) |
| Features | `all` · `all_rawtime` (+ raw hour/day) · `all_time_lags` (+ cyclical time + lags) · `sensors` (no reference analyzers) · `sensors_time_lags` |
| Models | feedforward network (128-64-32), with (`mlp_reg`) and without (`mlp_plain`) BatchNorm + Dropout · gradient boosting classifier (`hgb_clf`) · gradient boosting regressor with predictions binned into classes (`hgb_reg`) |

Preprocessing (median imputation, standard scaling) is fitted on the training part
only. `-200` marks missing readings. Lag features (previous hour, mean of the previous
three hours) use past values only.

Metrics: accuracy, macro-F1, share of predictions within one class, mean class
distance, share of errors falling between neighbouring classes, and share of errors
that under-predict.

## Findings (from my runs)

Macro-F1 of the gradient-boosting classifier, 3 seeds for `random` and `block`
(the `chrono` split is a single fixed split):

| Features | random | block | chrono |
|---|---|---|---|
| all | 0.735 | 0.678 | 0.544 |
| sensors | 0.663 | 0.604 | 0.490 |
| sensors_time_lags | 0.688 | 0.627 | 0.506 |

1. **A random split over-estimates performance.** Moving from random hours to random
   weeks costs about 0.06 macro-F1, which I attribute to dependence between
   neighbouring hours. Moving from weeks to the chronological split costs another
   0.12–0.13, which I attribute to a shift in the data over time. The cause of that
   shift (season or sensor drift) was **not** tested.
2. **Errors change character.** On random and block splits, errors are roughly
   symmetric (about half under-predict). On the chronological split 84–92% of the
   errors under-predict. The test period has about 53% of its samples in the top
   class (against 20% in training), so NO₂ levels were higher there.
3. **Most errors are between neighbouring classes.** With all features, about 95% of
   errors fall between adjacent classes and 98–99% of predictions are within one class
   (87–92% adjacent errors with sensors only). Extreme classes are classified more
   accurately than the middle ones.
4. **One reference measurement dominates.** With the reference analyzers available,
   shuffling `NOx(GT)` drops macro-F1 by about 0.29 in an earlier run, while every other
   feature drops it by less than 0.02. NOx includes NO₂, so the `sensors` setting is the
   more meaningful one: it is about 0.08 lower under a random split. Permutation
   importance with correlated features should not be over-interpreted.
5. **Gradient boosting matched or beat the neural network** in every setting I ran,
   by more under the chronological split (2 seeds; macro-F1, `all` features):

   | | hgb_clf | hgb_reg | mlp_plain | mlp_reg |
   |---|---|---|---|---|
   | random | 0.742 | 0.731 | 0.725 | 0.718 |
   | chrono | 0.544 | 0.580 | 0.478 | 0.449 |

   BatchNorm and Dropout had no consistent effect. Raw hour/day features did not help;
   cyclical time and lag features gave small gains (+0.02 to +0.03) in the
   `sensors` setting. Regression-then-binning equalled classification on random splits
   and was 0.03–0.04 better on the chronological split.

## Limitations

- Two or three seeds; differences of about 0.01–0.02 are within noise.
- The chronological split has a single test period, so its numbers have no standard
  deviation for the deterministic gradient-boosting models.
- The cause of the shift over time is a hypothesis.
- One dataset from one site, about one year of hourly data.
- Neural-network results vary slightly between machines and TensorFlow versions.

## Run it

```bash
pip install -r requirements.txt
# download AirQualityUCI.csv into data/ (see data/README.md)

python run_experiments.py --quick                    # lighter run
python run_experiments.py --models hgb_clf hgb_reg   # gradient boosting only, no TensorFlow
python run_experiments.py --seeds 5                  # fuller run
```

The neural-network runs are the slow part. To keep CPU use down on a laptop, limit
threads first, e.g. `OMP_NUM_THREADS=2 python run_experiments.py --quick`
(on Windows: `set OMP_NUM_THREADS=2`).

Outputs are written to `results/`: `runs.csv` (every run), `summary.csv` (mean and
standard deviation over seeds), confusion matrices and per-class reports,
permutation importance (CSV and figures), and `macro_f1_by_split.png`.

## Layout

```
no2_pipeline.py     data loading, features, splits, models, metrics
run_experiments.py  experiment grid, diagnostics, figures
data/               put AirQualityUCI.csv here (not committed)
results/            generated outputs
```
