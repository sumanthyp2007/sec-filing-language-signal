# Results

> **extractor=lexicon, offset=1**

Out-of-sample split: 2022-01-01 | cost assumption: 10.0 bps per leg round trip

| Horizon | Events | IC (mean) | IC t | LS gross | LS t | LS net | OOS prior IC | OOS fitted IC |
|---|---|---|---|---|---|---|---|---|
| 1d | 1036 | -0.02 | -0.65 | -0.23% | -1.35 | -0.43% | -0.02 | -0.02 |
| 5d | 1036 | 0.03 | 0.86 | 0.59% | 1.68 | 0.39% | 0.03 | 0.04 |
| 20d | 1031 | 0.03 | 0.74 | 0.57% | 0.85 | 0.37% | -0.01 | -0.03 |

LS = Q5 minus Q1 mean cumulative abnormal return per formation quarter. t-stats are across quarters.
IC = Spearman rank correlation between signal and abnormal return, per quarter.
