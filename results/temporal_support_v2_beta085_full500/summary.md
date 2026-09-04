# TemporalSupport V2 beta=0.85 Full-only 500-round result

## Fixed experiment

- Dataset/model: CIFAR-100 / ResNet; existing `data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json` split loaded with `generate_data=0`.
- Training: 100 clients, fraction 0.1, local epoch 5, local batch size 50, SGD, learning rate 0.01, decay 0.998, seed 1, `available` client selection.
- Profiles trained and synchronized every round: `(0.4,2)`, `(0.4,3)`, `(0.4,4)`, `(0.66,2)`, `(0.66,3)`, `(0.66,4)`, `(1.0,2)`. BatchNorm has `track_running_stats=False`.
- Evaluation: only Full `(1.0,2)`, exactly once after each aggregation round.
- Runtime: 4,442.175 seconds total; 8.884 seconds per round. Completed 500/500 rounds without an exception.

## Aggregation definition

Round 0 uses `Aggregation_AdaptiveFL` only.  From Round 1, with sample-weighted support `s`, AdaptiveFL update `u`, and beta `0.85`, the floating-coordinate evidence state is

`z_new = beta * z_prev + s * u`, `q_new = beta * q_prev + s` for `s > 0`, and both states only decay for `s == 0`.

For `0 < s < 1`, `Delta = s * u + (1 - s) * (z_new / q_new)`.  For `s == 0`, `Delta = 0`; for `s == 1`, the output is explicitly copied from AdaptiveFL.  Beta 0 reduces to AdaptiveFL on every covered coordinate.

## Full accuracy (%)

| Metric | Result |
| --- | ---: |
| Peak@0-299 | 36.18 @ R265 |
| Peak@0-499 | 36.42 @ R422 |
| Accuracy@299 | 36.15 |
| Accuracy@499 | 35.72 |
| mean(acc[240:270]) | 35.39 |
| mean(acc[270:300]) | 35.20 |
| Trend_300 | -0.19 |
| mean(acc[440:470]) | 36.05 |
| mean(acc[470:500]) | 35.81 |
| Trend_500 | -0.24 |
| PeakToLast30Drop | 0.61 |
| Post-peak OLS slope (R422-499) | -0.004061 percentage points/round |

## Reference comparison

| Method | Peak@0-299 | Peak@0-499 |
| --- | ---: | ---: |
| AdaptiveFL Full (known reference) | 36.45 @ R230 | 37.13 @ R433 |
| Support epsilon=0.2 Full (known reference) | 36.66 | 37.00 |
| TemporalSupport V2 beta=0.85 | 36.18 @ R265 | 36.42 @ R422 |

## Exploratory paired trajectory robustness check

`resut_ori/AdaptiveFL.log` contains 1,000 rounds with seven profile evaluations per round.  Taking its seventh value per round gives the Full profile and exactly reproduces the known reference peaks (36.45 @ R230 and 37.13 @ R433), so its first 500 rounds were used as the paired baseline.

- Mean new-minus-AdaptiveFL difference: -0.260580 percentage points; median: -0.350000.
- Win-round ratio: 0.244 (122/500).
- Early (R0-149), middle (R150-299), and late (R300-499) mean differences: +0.096333, -0.184400, and -0.585400 percentage points.

This is an **exploratory paired trajectory robustness check**, not a statistical significance test: it describes one matched seed trajectory and does not substitute for multi-seed inference.

## Temporal diagnostics (mean over R1-R499)

- support mean: 0.482821; support nonzero ratio: 0.926584.
- current update L2: 0.356951; temporal mean L2: 0.110152; memory component L2: 0.041188; final update L2: 0.291670.
- Mean evidence mass by support band: low 1.148369; medium 3.601708; high 3.670250; full 5.771215; zero 0.147213.
- Full-support temporal component L2 is exactly 0 by construction; zero-support final-update L2 is exactly 0 by the safety rule.

## Outcome

**negative**.  This single fixed beta=0.85 run is below both supplied AdaptiveFL peak references, and its final 30-round mean is below its own peak.  No beta tuning or follow-on mechanism was run.
