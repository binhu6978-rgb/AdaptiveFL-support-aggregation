# Width-Group Consensus Geometry Analysis

## Scope and exact reconstruction

This is a zero-training-cost, offline analysis of `pairwise_consistency.json` and `client_loo_consistency.json` for diagnostic R1-R49.  It reconstructs each round's exact client-update Gram matrix using `G_ij = ||Delta_i|| * ||Delta_j|| * cosine_ij` and `G_ii = ||Delta_i||^2`; it does not reconstruct, embed, or approximate update vectors.

Occurrence-order alignment passed for all 49 rounds: each round contained exactly 10 client records and 45 triangular pair records, and every pair's client id, model index, and width matched the corresponding canonical client occurrences.  Duplicate client ids occurred in 25 rounds (28 duplicate occurrences in total); they were retained as distinct occurrences rather than matched by id.

## Natural width-group participation

Values are sample-mass shares `alpha_g`; client counts are per round.

| Width | Client count mean / median | Sample share mean | median | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 4.29 / 4 | 0.4317 | 0.3985 | 0.2266 | 0.6914 |
| 0.66 | 4.22 / 4 | 0.4240 | 0.4200 | 0.2233 | 0.6051 |
| 1.0 | 1.49 / 1 | 0.1443 | 0.1201 | 0.0000 | 0.2919 |

Phase mean sample shares (0.4 / 0.66 / 1.0): Early `0.4724 / 0.4104 / 0.1173`; Middle `0.4074 / 0.4376 / 0.1550`; Late `0.4200 / 0.4226 / 0.1574`.  Full was absent in 9/49 rounds and was a single-client group in 18/49 rounds, so its concentration must not be interpreted as within-group consensus on those single-client rounds.

## Group-direction cosine

| Pair | Valid | Mean ± std | Median | P10 / P25 / P75 / P90 | Min / max | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 vs 0.66 | 49 | 0.0745 ± 0.0630 | 0.0566 | 0.0109 / 0.0300 / 0.1143 / 0.1577 | -0.0304 / 0.2561 | 2.0% |
| 0.4 vs 1.0 | 40 | 0.0405 ± 0.0393 | 0.0279 | 0.0081 / 0.0172 / 0.0463 / 0.0947 | -0.0044 / 0.1750 | 2.5% |
| 0.66 vs 1.0 | 40 | 0.0614 ± 0.0540 | 0.0354 | 0.0161 / 0.0210 / 0.0850 / 0.1583 | 0.0061 / 0.1990 | 0.0% |

| Pair | Early mean / median (valid) | Middle mean / median (valid) | Late mean / median (valid) |
| --- | ---: | ---: | ---: |
| 0.4 vs 0.66 | 0.1475 / 0.1373 (15) | 0.0464 / 0.0463 (17) | 0.0382 / 0.0333 (17) |
| 0.4 vs 1.0 | 0.0721 / 0.0674 (12) | 0.0227 / 0.0223 (14) | 0.0312 / 0.0193 (14) |
| 0.66 vs 1.0 | 0.1038 / 0.0974 (12) | 0.0368 / 0.0270 (14) | 0.0498 / 0.0352 (14) |

The OLS cosine slopes per round are `-0.003232`, `-0.001323`, and `-0.001928` respectively; Spearman correlations with round are `-0.7184`, `-0.5634`, and `-0.4298`.  These are descriptive trends, not causal estimates.

## Internal concentration

`C_g = ||S_g|| / sum_i n_i ||Delta_i||`.  Overall median concentrations are 0.5263 (0.4), 0.5203 (0.66), and 0.7512 (1.0, 40 observed rounds).  In Late, medians are 0.5263, 0.5266, and 0.7247.  Thus the small/medium group means have moderate internal cancellation, but not near-total cancellation; Full is more concentrated when present, with the single-client caveat above.

## Counterfactual geometry

Both counterfactuals are exact 3x3 group-Gram calculations and are geometry diagnostics only, not proposed aggregation methods.

| Counterfactual | Overall valid | Median cosine to natural | Median angle | Angle P25 / P75 | Median norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal width (1/3 each) | 40 | 0.9025 | 25.48° | 17.18° / 33.61° | 1.0134 |
| Full=20%, remaining 80% naturally split | 40 | 0.9849 | 9.97° | 2.67° / 13.05° | 0.9972 |

| Counterfactual | Early median / mean angle; median norm | Middle median / mean angle; median norm | Late median / mean angle; median norm |
| --- | ---: | ---: | ---: |
| Equal width | 27.63° / 26.95°; 1.0287 | 25.48° / 26.44°; 1.0096 | 20.08° / 21.18°; 1.0118 |
| Full=20% | 10.77° / 8.49°; 0.9975 | 6.00° / 7.31°; 0.9964 | 11.32° / 10.24°; 1.0013 |

## Late-stage focus

| Late R33-R49 metric | Result |
| --- | ---: |
| Full vs 0.4 group cosine, median (14 valid) | 0.0193 |
| Full vs 0.66 group cosine, median (14 valid) | 0.0352 |
| Full natural sample share, mean / median | 0.1574 / 0.1126 |
| Full=20% median angle / norm ratio | 11.32° / 1.0013 |
| Equal-width median angle / norm ratio | 20.08° / 1.0118 |

## Direction judgment

**shared-scale reweighting has meaningful geometric freedom.**  Width-group directions remain distinct rather than converging in Late, and the conservative Full=20% counterfactual has a non-negligible median directional rotation (11.32°) while leaving the norm essentially unchanged.  The group means show moderate, not near-total, internal cancellation.  This establishes only geometric freedom: it does not say that the Full direction is better, that any reweighting improves accuracy, or that a new aggregation should be implemented without a separate review.
