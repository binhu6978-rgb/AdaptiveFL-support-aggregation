# AdaptiveFL baseline profile-exposure audit

Source: real `ration_users` and `idx_users` records parsed from the first 500 rounds of `resut_ori/AdaptiveFL.log`. Sample-weighted exposure sums the selected client dataset sizes; LR-weighted exposure is only the proxy `sum(lr_t * samples)` with `lr_t = 0.01 * 0.998^t`, not a gradient-contribution estimate.

## Profile occurrence

- Total slots: 5000 (500 rounds × 10)
- Profile 0–6 counts: [710, 688, 743, 708, 690, 739, 722]
- Small / Medium / Full counts: 2141 / 2137 / 722
- Full mean occurrences per round: 1.4440
- Full zero-update-round ratio: 18.80%

## Parameter-region exposure

| Region | Slot total | Sample-weighted total | LR-weighted sample proxy | Mean slots/round | Median | P10 | P90 | Zero-round ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_small | 5000 | 2481937 | 15703.51 | 10.000 | 10.0 | 10.0 | 10.0 | 0.00% |\n| medium_capable | 2859 | 1418377 | 8977.56 | 5.718 | 6.0 | 4.0 | 8.0 | 0.00% |\n| full_exclusive | 722 | 353671 | 2260.54 | 1.444 | 1.0 | 0.0 | 3.0 | 18.80% |\n
## Exposure ratios to shared

| Region / shared | Occurrence | Sample-weighted | LR-weighted sample proxy |
| --- | ---: | ---: | ---: |
| medium_capable_over_shared | 0.571800 | 0.571480 | 0.571691 |\n| full_exclusive_over_shared | 0.144400 | 0.142498 | 0.143951 |\n