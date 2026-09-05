# Fixed 4:3:3 Profile-Quota Oracle — 500-round experiment

## Run integrity

- Branch: `fixed-profile-quota-oracle-exp500`
- Method: clean `Aggregation_AdaptiveFL` and unchanged `LocalUpdate_AdaptiveFL`, with only `fixed_profile_quota_oracle=True` changing `ration_users`.
- Configuration: CIFAR-100 / ResNet; cached Dirichlet split (`iid=0`, `beta=0.3`, seed 1); 100 clients; 10 slots/round; 5 local epochs; batch size 50; SGD (`lr=0.01`, `lr_decay=0.998`); available client selection; seven original profiles; BatchNorm `track_running_stats=False`.
- Completed 500/500 rounds with no exception. Runtime: 9,428.475 seconds (18.857 seconds/round).
- The cached split remained intact: min / median / max client samples = `262 / 491.5 / 757`, total 50,000.

## Dispatch integrity and actual exposure

Every round has exactly 4 Small, 3 Medium, and 3 Full slots. The final profile counts are `[631, 664, 705, 509, 505, 486, 1500]`, for Small / Medium / Full totals of `2000 / 1500 / 1500`.

- Total slots: 5,000
- Full mean occurrences/round: 3.0; Full zero-update-round ratio: 0%
- Full selected-client range: 70–99; unique Full clients: 30
- Full participation counts for clients 70–99: `70:49, 71:59, 72:51, 73:45, 74:63, 75:54, 76:46, 77:52, 78:50, 79:48, 80:50, 81:43, 82:58, 83:47, 84:47, 85:45, 86:68, 87:60, 88:45, 89:55, 90:42, 91:55, 92:46, 93:48, 94:41, 95:54, 96:40, 97:53, 98:34, 99:52` (min / median / max = `34 / 49.5 / 68`). These are dispatch integrity counts, not fairness metrics.

| Region | Slot exposure | Sample-weighted exposure | LR-weighted sample proxy | Occurrence ratio vs baseline | Sample ratio vs baseline | LR proxy ratio vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared | 5,000 | 2,480,032 | 15,686.58 | 1.0000 | 0.9992 | 0.9989 |
| Medium-capable | 3,000 | 1,480,815 | 9,359.93 | 1.0493 | 1.0440 | 1.0426 |
| Full-exclusive | 1,500 | 733,308 | 4,628.68 | 2.0776 | 2.0734 | 2.0476 |

The LR-weighted value is only an exposure proxy, not a gradient-contribution estimate.

## Accuracy by profile

Accuracy is in percent. Peak differences are Fixed Quota minus the first 500 rounds of the local AdaptiveFL baseline.

| Profile | Peak 0–299 (round) | Peak 0–499 (round) | Acc@299 | Acc@499 | Baseline peak | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.4, 2) | 39.17 (299) | 39.96 (412) | 39.17 | 38.76 | 40.09 | -0.13 |
| (0.4, 3) | 39.72 (283) | 39.84 (366) | 38.98 | 38.73 | 39.75 | +0.09 |
| (0.4, 4) | 38.49 (265) | 38.49 (265) | 37.73 | 37.39 | 38.72 | -0.23 |
| (0.66, 2) | 38.08 (282) | 38.08 (282) | 37.45 | 36.82 | 38.03 | +0.05 |
| (0.66, 3) | 38.31 (265) | 38.31 (265) | 37.59 | 36.97 | 37.94 | +0.37 |
| (0.66, 4) | 38.17 (265) | 38.17 (265) | 37.43 | 36.92 | 37.94 | +0.23 |
| (1.0, 2) Full | 35.93 (282) | 35.93 (282) | 34.98 | 35.14 | 37.13 | -1.20 |

The six non-Full profiles do not show a broad collapse, but the experiment does not improve its primary Full objective.

## Full trajectory and comparison

Full late-window statistics:

- mean R240–269 = 34.26; mean R270–299 = 34.69; Trend_300 = +0.43
- mean R440–469 = 35.12; mean R470–499 = 34.92; Trend_500 = -0.20
- Peak-to-last-30-mean drop = 1.01
- Post-peak OLS slope (R282–499) = +0.00238 percentage points/round

Compared with AdaptiveFL Full, Fixed Quota minus baseline is:

| Metric | Difference (pp) |
| --- | ---: |
| Peak 0–499 | -1.20 |
| Trajectory mean R0–499 | -1.35 |
| Trajectory median R0–499 | -1.50 |
| Strict win-round ratio | 6.0% |
| Early R0–149 mean | -1.08 |
| Middle R150–299 mean | -1.33 |
| Late R300–499 mean | -1.57 |
| R400–499 mean | -1.53 |

This is a descriptive single-seed trajectory comparison, not a statistical-significance claim.

## Three-way Full comparison

| Run | Full peak | Full late mean R470–499 | Full trajectory mean R0–499 |
| --- | ---: | ---: | ---: |
| AdaptiveFL | 37.13 | 36.47 | 33.80 |
| Fixed 4:3:3 Quota | 35.93 | 34.92 | 32.45 |
| Full-Access Oracle | 40.14 | 39.53 | 36.64 |

The Full-Access values are read from the previously committed Full-Access Oracle result (`a6d3058`) and are not a new run.

## Outcome

**NEGATIVE** for the fixed 4:3:3 quota hypothesis: despite the expected ~2.08× Full-exclusive occurrence exposure, Full peak and trajectory are materially below baseline, including the late phase. This does not support a quota sweep or further fixed-ratio tuning. It also does not negate the separate Full-Access Oracle result, because the two diagnostics change different mechanisms: exposure frequency versus eligible population.
