# Support Aggregation V1 — Full-only 500-round experiment

## Run identity

- Repository: https://github.com/binhu6978-rgb/AdaptiveFL-support-aggregation
- Branch: `support-v1-exp500-full-only`
- Preparation commit: `446d467`
- Dataset/model: CIFAR-100 / ResNet
- Full evaluation profile: `(width=1.0, depth=2, parameters=11.424356M)`
- Completed rounds: 500 (`Round 0` through `Round 499`)
- Total wall time: 6177.167133 seconds (1:42:57.167)
- Average wall time per round: 12.354334 seconds
- NaN / Inf / traceback / exception: none detected

## Frozen training configuration

- `iid=0`
- Dirichlet `data_beta=0.3`
- `num_users=100`
- `frac=0.1`
- `local_ep=5`
- optimizer: SGD
- `lr=0.01`
- `lr_decay=0.998`
- `seed=1`
- `client_chosen_mode=available`
- `depth_saved=[2, 3, 4]`
- `width_ration=[0.4, 0.66, 1.0]`
- round 0 aggregation: original `Aggregation_AdaptiveFL`
- rounds 1–499 aggregation: `Aggregation_Support(epsilon=0.2)`
- all seven heterogeneous profiles trained and synchronized every round
- evaluation: Full profile only, exactly one `test()` per round
- split record: existing `data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json`; it was loaded with `generate_data=0` and was not regenerated

Client sample sizes: min 262, median 491.5, max 757.

## Command

```powershell
D:\software\minicoonda\envs\fd\python.exe main_fed_ori.py --gpu 0 --epochs 500 --num_users 100 --frac 0.1 --local_ep 5 --optimizer sgd --lr 0.01 --lr_decay 0.998 --iid 0 --data_beta 0.3 --generate_data 0 --seed 1 --client_chosen_mode available --depth_saved 2 3 4 --width_ration 0.4 0.66 1.0
```

## Full-model results

| Metric | Result | AdaptiveFL baseline | Delta |
|---|---:|---:|---:|
| Peak@0–299 | 36.66% at Round 231 | 36.45% | +0.21 pp |
| Accuracy@299 | 36.20% | — | — |
| Peak@0–499 | 37.00% at Round 353 | 37.13% | -0.13 pp |
| Accuracy@499 | 36.00% | — | — |

Trend near Round 300:

- `mean(acc[240:270]) = 35.657667`
- `mean(acc[270:300]) = 35.495000`
- difference = `-0.162666` percentage points

Trend near Round 500:

- `mean(acc[440:470]) = 36.478333`
- `mean(acc[470:500]) = 36.190667`
- difference = `-0.287667` percentage points

The first-stage 300-round threshold was exceeded by 0.21 percentage points. The 500-round peak finished 0.13 percentage points below the AdaptiveFL reference, and both requested endpoint trend indicators were negative.

## Support diagnostics summary

Statistics below aggregate the 499 Support-aggregation rounds (rounds 1–499). Each row reports the across-round mean, minimum, and maximum of the logged scalar.

| Diagnostic | Mean | Min | Max |
|---|---:|---:|---:|
| support min_nonzero | 0.180034 | 0.052706 | 0.504035 |
| support mean | 0.482821 | 0.290420 | 0.704042 |
| support P25 | 0.142695 | 0.000000 | 0.429399 |
| support P50 | 0.406206 | 0.000000 | 0.804719 |
| support P75 | 0.720204 | 0.296373 | 1.000000 |
| coordinate ratio `s<0.3` | 0.398657 | 0.000000 | 0.768358 |
| coordinate ratio `0.3<=s<0.8` | 0.359047 | 0.000000 | 0.814345 |
| coordinate ratio `s>=0.8` | 0.242295 | 0.185655 | 0.597718 |
| update L2 `s<0.3` | 0.051343 | 0.000000 | 0.458903 |
| update L2 `0.3<=s<0.8` | 0.191588 | 0.000000 | 0.644966 |
| update L2 `s>=0.8` | 0.252691 | 0.043606 | 0.619401 |
| global update L2 | 0.329278 | 0.046435 | 0.856550 |

## Integrity checks

- Full profile startup assertion: passed; final profile was `(1.0, 2, 11.424356)`.
- Round 1 prefix consistency assertion: passed.
- Round log sequence: exactly 500 entries, contiguous from 0 to 499.
- Full accuracy entries: exactly 500.
- Support diagnostic entries: exactly 499, as expected because round 0 used the original aggregation.
- Non-finite accuracy values: zero.
- Formal log and process stderr contained no traceback, runtime error, exception, NaN, or Inf.

## Files

Experiment-preparation source modifications:

- `Algorithm/Training_AdaptiveFL.py`: kept all-profile synchronization, changed evaluation to Full-only, added profile/prefix assertions, and saved one Full accuracy curve.
- `main_fed_ori.py`: froze the 500-round configuration, printed split statistics, used an isolated result directory, replaced silent exception handling with traceback plus raise, recorded wall time, and delayed the unused Transformer import.
- `utils/options.py`: set the non-IID result case default to 4.

Result artifacts:

- `results/support_v1_eps02_full/full_accuracy.json`
- `results/support_v1_eps02_full/support_v1_eps02_full_500round.log`
- `results/support_v1_eps02_full/summary.md`
