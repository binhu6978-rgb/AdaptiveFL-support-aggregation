# Cross-Scale Update Consistency Diagnostic (50 rounds)

## Experiment identity and integrity

- Purpose: read-only measurement of shared-core client update consistency; no aggregation method was added or changed.
- Command: `D:\software\minicoonda\envs\fd\python.exe main_fed_ori.py`
- Algorithm and aggregation: heterogeneous `AdaptiveFL`; original `Aggregation_AdaptiveFL` for rounds 0–49.
- Diagnostic rounds: 1–49. Round 0 was trained and aggregated normally but explicitly skipped by the consistency diagnostic.
- Full `(1.0,2)` was tested once per round as an integrity signal only.
- Diagnostic records: 49 rounds, 2,205 pairwise observations (45/round), and 490 LOO observations (10/round).
- All 49 diagnostic input checks had identical before/after checksums and `max_abs_diff=0`.
- RNG-state assertion passed; all cosine and norm values were finite.
- NaN / Inf / exception: none detected.
- Forbidden aggregation/gating mechanism mentions in the run log: 0.

## Frozen configuration

- dataset/model: CIFAR-100 / ResNet
- existing split: `data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json`
- `iid=0`, `data_beta=0.3`, `generate_data=0`, `seed=1`
- `num_users=100`, `frac=0.1`
- `local_ep=5`, `local_bs=50`
- optimizer/lr/decay: SGD / 0.01 / 0.998
- `client_chosen_mode=available`; duplicate clients retained
- `depth_saved=[2,3,4]`, `width_ration=[0.4,0.66,1.0]`
- epochs: 50 (rounds 0–49)
- client sample sizes (min / median / max): 262 / 491.5 / 757
- BatchNorm: 147 modules, all `track_running_stats=False`
- shared core: shapes from profile `(0.4,2)`, 2,120,985 floating-point coordinates

## Model profiles

| Index | Profile | Parameters (M) |
|---:|---:|---:|
| 0 | (0.4,2) | 2.120985 |
| 1 | (0.4,3) | 2.646356 |
| 2 | (0.4,4) | 4.647740 |
| 3 | (0.66,2) | 5.159621 |
| 4 | (0.66,3) | 5.528077 |
| 5 | (0.66,4) | 6.945581 |
| 6 | (1.0,2) | 11.424356 |

## Pairwise cosine: all observations

| Count | Mean | Std | Median | P10 | P25 | P75 | P90 | Min | Max | Negative | `<0.3` | `<0.5` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2205 | 0.028064 | 0.050767 | 0.016874 | -0.003086 | 0.005149 | 0.034673 | 0.065072 | -0.050714 | 0.671639 | 14.7846% | 99.2290% | 99.7732% |

## Pairwise groups

| Group | Count | Mean | Median | Negative ratio |
|---|---:|---:|---:|---:|
| Same exact profile | 290 | 0.042135 | 0.024393 | 7.9310% |
| Same width | 843 | 0.035189 | 0.022527 | 10.4389% |
| Cross width | 1362 | 0.023654 | 0.014243 | 17.4743% |

## LOO client-consensus cosine: all observations

| Count | Mean | Std | Median | P10 | P25 | P75 | Min | Negative | `<0.3` | `<0.5` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 490 | 0.070352 | 0.059403 | 0.047212 | 0.014122 | 0.025995 | 0.102858 | -0.040679 | 2.4490% | 100.0000% | 100.0000% |

## LOO cosine by profile

| Profile | Observations | Mean | Median | P10 | Negative ratio | Mean update norm |
|---:|---:|---:|---:|---:|---:|---:|
| (0.4,2) | 74 | 0.065651 | 0.042709 | 0.017550 | 1.3514% | 1.554605 |
| (0.4,3) | 68 | 0.088670 | 0.072694 | 0.014587 | 1.4706% | 1.283267 |
| (0.4,4) | 68 | 0.093166 | 0.071900 | 0.019397 | 1.4706% | 0.980455 |
| (0.66,2) | 63 | 0.052851 | 0.034240 | 0.005778 | 6.3492% | 1.125800 |
| (0.66,3) | 77 | 0.064447 | 0.054119 | 0.013839 | 2.5974% | 1.042414 |
| (0.66,4) | 67 | 0.067728 | 0.045017 | 0.012622 | 1.4925% | 0.939361 |
| (1.0,2) | 73 | 0.060544 | 0.035805 | 0.017009 | 2.7397% | 0.871671 |

## Full cross-width pair comparisons

| Pair group | Count | Mean | Median | Negative ratio |
|---|---:|---:|---:|---:|
| Full `(1.0,2)` vs width 0.4 | 273 | 0.019947 | 0.012927 | 17.9487% |
| Full `(1.0,2)` vs width 0.66 | 294 | 0.026258 | 0.013764 | 13.2653% |

## Full accuracy for rounds 0–49

Integrity-only sequence (%):

`[1.24, 2.99, 4.74, 6.89, 7.29, 7.92, 10.04, 11.87, 11.73, 12.94, 15.18, 15.62, 14.48, 16.57, 16.30, 16.63, 19.09, 17.87, 17.68, 19.25, 20.14, 19.16, 21.50, 21.66, 23.35, 24.12, 23.87, 24.06, 25.02, 22.92, 24.34, 25.44, 25.02, 26.08, 24.68, 27.41, 26.62, 26.36, 27.00, 26.58, 28.25, 27.30, 27.12, 27.48, 28.27, 28.13, 29.10, 28.50, 29.97, 28.77]`

## Runtime

- total wall time: 467.984628 seconds
- average wall time per round: 9.359693 seconds

## Objective judgment

**mixed / inconclusive**

Shared-core pairwise consistency is generally very low, and cross-width pairs have a lower mean and a substantially higher negative ratio than same-width pairs, so there is measurable cross-scale structure. However, LOO cosines are concentrated at low positive values, negative LOO observations are only 2.45%, and Full-profile LOO statistics do not yet show a clearly separated high/low population. The 50-round diagnostic therefore shows a candidate consistency signal but does not by itself establish a strong, discriminative client-gating signal.

No consensus gate, V2 method, 300-round run, or follow-up experiment was started.
