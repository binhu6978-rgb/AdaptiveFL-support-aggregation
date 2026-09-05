# Full-Access Oracle — 500-round experiment

## Run integrity

- Branch: `full-access-oracle-exp500`
- Method: the unmodified `Aggregation_AdaptiveFL` code path with
  `--full_access_oracle` enabled.
- Configuration: CIFAR-100, ResNet, `iid=0`, `beta=0.3`, `seed=1`, 100
  clients, 10 selected clients/round, 5 local epochs, SGD (`lr=0.01`,
  `lr_decay=0.998`), and the original seven available-model profiles.
- Completed 500/500 rounds without an exception. Elapsed time: 9,243.818 s
  (18.488 s/round).
- Evaluation was performed for all seven profiles at every round.
- The causal-isolation dispatch implementation was used: the global RNG is
  advanced by the original Full-client draw while the oracle client is drawn
  from a cloned pre-draw RNG state. The corresponding mixed-sequence test
  passed for seeds `0, 1, 2, 6, 9, 17, 12345, 2026`.

## Dispatch evidence

Over 500 rounds there were 10,589 profile assignments. The seven profile
assignment counts, in evaluation order, were `1492, 1600, 1469, 1442, 1573,
1475, 1538`. Full profile assignments numbered 1,538, of which 722 selected a
Full client. Those 722 oracle Full selections covered all 100 client IDs
(`min=0`, `max=99`), including 306 / 227 / 189 selections in the original
weak / medium / strong client-ID blocks `[0,42)`, `[42,73)`, `[73,100)`.

`dispatch_log.json` records the profile assignment and selected client IDs for
each round. It is therefore possible to verify both the per-round Full-access
draws and the unchanged non-Full assignment trajectory.

## Per-profile accuracy

Accuracy is reported in percent. Peak differences compare this Oracle run with
the first 500 rounds of the local AdaptiveFL baseline trajectory.

| Profile | Oracle peak 0–299 (round) | Oracle peak 0–499 (round) | R299 | R499 | Baseline peak 0–499 | Peak difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.4, 2) | 40.01 (299) | 40.64 (404) | 40.01 | 39.54 | 40.09 | +0.55 |
| (0.4, 3) | 40.02 (298) | 40.64 (306) | 39.97 | 38.97 | 39.75 | +0.89 |
| (0.4, 4) | 38.76 (299) | 38.78 (318) | 38.76 | 37.72 | 38.72 | +0.06 |
| (0.66, 2) | 38.47 (230) | 38.47 (230) | 38.10 | 37.32 | 38.03 | +0.44 |
| (0.66, 3) | 38.58 (230) | 38.85 (325) | 38.52 | 37.13 | 37.94 | +0.91 |
| (0.66, 4) | 39.25 (258) | 39.26 (306) | 38.59 | 38.38 | 37.94 | +1.32 |
| (1.0, 2) Full | 39.86 (230) | 40.14 (336) | 39.49 | 39.40 | 37.13 | +3.01 |

## Full-profile comparison with AdaptiveFL

The known baseline Full peaks are 36.45 at round 230 in rounds 0–299, and
37.13 at round 433 in rounds 0–499. The Oracle Full profile reaches 39.86 at
round 230 and 40.14 at round 336, respectively. The 0–499 peak improvement is
**+3.01 percentage points**.

| Trajectory comparison (Oracle − baseline) | Value (percentage points) |
| --- | ---: |
| Paired mean difference, R0–499 | +2.84 |
| Paired median difference, R0–499 | +3.02 |
| Oracle wins (strictly higher), R0–499 | 97.6% |
| Mean difference, R0–149 | +2.05 |
| Mean difference, R150–299 | +3.45 |
| Mean difference, R300–499 | +2.97 |
| Mean difference, R400–499 | +2.97 |

Full late-training values remain high: the R240–269 and R270–299 means are
39.07 and 38.92 (change −0.15); the R440–469 and R470–499 means are 39.63 and
39.53 (change −0.09). The final 30-round mean is 0.61 points below the run
peak. The least-squares slope from the Full peak (R336) through R499 is
+0.00265 percentage points/round.

## Outcome

**Strong positive effect for the Full profile.** The effect is not confined to
a single peak: it persists in the early, middle, late, and R400–499 trajectory
averages, with a 97.6% per-round win ratio. This is a controlled single-seed
oracle comparison, so the result is reported descriptively rather than as a
multi-seed statistical claim.
