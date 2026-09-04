# All-Large AdaptiveFL-path Oracle (300 rounds)

## Experiment identity

- Purpose: controlled All-Large capacity/oracle diagnostic; this is not a new aggregation method and not a strict aggregation-only ablation.
- Algorithm path: `AdaptiveFL`
- Aggregation for rounds 0–299: original `Aggregation_AdaptiveFL`
- Support aggregation: not called
- Command: `D:\software\minicoonda\envs\fd\python.exe main_fed_ori.py`

## Configuration

- dataset/model: CIFAR-100 / ResNet
- split: existing non-IID Dirichlet split, alpha=0.3, seed=1 (`generate_data=0`)
- users / participation: 100 / 0.1
- local epochs / batch size: 5 / 50
- optimizer: SGD
- learning rate / decay: 0.01 / 0.998
- client selection: AdaptiveFL `available` path
- communication rounds: 300 (rounds 0–299)
- model pool: `depth_saved=[2]`, `width_ration=[1.0]`

## Pre-run confirmations

- client sample sizes (min / median / max): 262 / 491.5 / 757
- `len(net_glob_list)`: 1
- Full profile: `(width=1.0, depth=2, parameters=11.424356M)`
- BatchNorm modules: 21; all use `track_running_stats=False`
- all 300 dispatch vectors contained only profile index 0
- the existing AdaptiveFL selection path was retained; duplicate selected clients were not removed

## Results

- completed rounds: 300 / 300
- Peak@0–299: 33.3300% at round 127
- Accuracy@299: 31.2900%
- mean(acc[240:270]): 31.7020%
- mean(acc[270:300]): 31.5513%
- Trend_300: -0.1507 percentage points
- DeltaOracle_300 versus heterogeneous AdaptiveFL peak 36.45%: -3.1200 percentage points
- total wall time: 2891.098439 seconds
- average wall time per round: 9.636995 seconds
- NaN / Inf / exception: none detected

## Comparison and interpretation

| Experiment | Peak accuracy in rounds 0–299 |
|---|---:|
| AdaptiveFL heterogeneous | 36.45% |
| Support epsilon=0.2 | 36.66% |
| All-Large AdaptiveFL-path oracle | 33.33% |

This result falls into classification C: the All-Large oracle is below the heterogeneous AdaptiveFL result. Heterogeneous parameter sharing may therefore provide a net positive effect to the Full model in this regime. This does not establish that aggregation conflict is absent, and the difference must be described only as a capacity/oracle gap, not as aggregation loss. Trend_300 is not greater than +0.10 percentage points, so this diagnostic does not trigger the stated recommendation to extend automatically; no further run was started.

## Artifacts

- `results/all_large_oracle_300/full_accuracy.json`
- `results/all_large_oracle_300/all_large_oracle_300round.log`
