# All-Small AdaptiveFL-path Oracle (300 rounds)

## Experiment identity

- Purpose: controlled All-Small capacity/oracle diagnostic; this is not a new aggregation method and not a strict aggregation-only ablation.
- Algorithm path: `AdaptiveFL`
- Aggregation for rounds 0–299: original `Aggregation_AdaptiveFL`
- Support aggregation: not called
- Command: `D:\software\minicoonda\envs\fd\python.exe main_fed_ori.py`

## Configuration

- dataset/model: CIFAR-100 / ResNet
- split: existing non-IID Dirichlet split, alpha=0.3, seed=1 (`generate_data=0`)
- split file: `data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json`
- users / participation: 100 / 0.1
- local epochs / batch size: 5 / 50
- optimizer: SGD
- learning rate / decay: 0.01 / 0.998
- client selection: AdaptiveFL `available` path
- communication rounds: 300 (rounds 0–299)
- model pool: `depth_saved=[2]`, `width_ration=[0.4]`

## Pre-run confirmations

- client sample sizes (min / median / max): 262 / 491.5 / 757
- `len(net_glob_list)`: 1
- Small profile: `(width=0.4, depth=2, parameters=2.120985M)`
- BatchNorm modules: 21; all use `track_running_stats=False`
- all 300 dispatch vectors contained only profile index 0
- the existing AdaptiveFL selection path was retained; duplicate selected clients were not removed

## Results

- completed rounds: 300 / 300
- Peak@0–299: 31.6100% at round 143
- Accuracy@299: 30.0700%
- mean(acc[240:270]): 30.5080%
- mean(acc[270:300]): 30.2180%
- Trend_300: -0.2900 percentage points
- total wall time: 2080.973312 seconds
- average wall time per round: 6.936578 seconds
- NaN / Inf / exception: none detected

## Joint comparison

| Setting | Profile | Peak@0–299 |
|---|---:|---:|
| Heterogeneous AdaptiveFL | (0.4,2) | 40.06% |
| Heterogeneous AdaptiveFL | (1.0,2) | 36.45% |
| Support epsilon=0.2 | (1.0,2) | 36.66% |
| All-Large oracle | (1.0,2) | 33.33% |
| All-Small oracle | (0.4,2) | 31.61% |

## Interpretation

This result falls into classification B: All-Small reaches about 31–33%, while the heterogeneous Small profile reaches 40.06%. Under the specified diagnostic interpretation, heterogeneous cross-scale training provides a strong additional benefit; the result does not support the simpler explanation that the Small architecture alone is inherently better in this regime. No difference here should be described as aggregation loss.

Trend_300 is not greater than +0.10 percentage points, so the stated continuation condition is not met. No 500-round continuation or other experiment was started.

## Artifacts

- `results/all_small_oracle_300/small_accuracy.json`
- `results/all_small_oracle_300/all_small_oracle_300round.log`
