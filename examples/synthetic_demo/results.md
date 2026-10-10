# Results

> **SYNTHETIC DEMO DATA (planted effect=0.02). Validates the pipeline only; says nothing about real markets.**

Out-of-sample split: 2019-01-01 | cost assumption: 10 bps per leg round trip

| Horizon | Events | IC (mean) | IC t | LS gross | LS t | LS net | OOS prior IC | OOS fitted IC |
|---|---|---|---|---|---|---|---|---|
| 1d | 3360 | 0.02 | 1.33 | 0.07% | 0.94 | -0.13% | -0.00 | -0.05 |
| 5d | 3360 | 0.07 | 4.16 | 0.55% | 3.54 | 0.35% | 0.07 | 0.05 |
| 20d | 3360 | 0.12 | 6.95 | 2.29% | 6.62 | 2.09% | 0.12 | 0.11 |

LS = Q5 minus Q1 mean cumulative abnormal return per formation quarter. t-stats are across quarters.
IC = Spearman rank correlation between signal and abnormal return, per quarter.
