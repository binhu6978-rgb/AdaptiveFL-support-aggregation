# Multi-client Full Geometry Robustness Analysis

## Exact offline filter

This analysis reuses the same two saved diagnostic JSON files and reconstructs the exact per-round Gram matrix from saved update norms and pairwise cosines.  It retains a round only when widths 0.4, 0.66, and 1.0 are all present and the Full (width 1.0) **occurrence** count is at least two.  Duplicate client ids are deliberately retained as separate occurrences.

- Original diagnostic rounds: 49.
- Full-present rounds: 40.
- Full-count-equals-one rounds: 18.
- Final valid rounds: 22.
- Valid IDs: `1, 3, 5, 8, 9, 16, 19, 22, 26, 27, 28, 30, 31, 32, 33, 34, 35, 39, 40, 46, 48, 49`.
- Occurrence-order alignment passed again in all 49 input rounds: 10 client records and 45 triangular pair records each.  The input has 28 duplicate occurrences across 25 rounds; no occurrence was deduplicated.

## Group cosine after filtering

| Pair | Valid | Mean ± std | Median | P25 / P75 | Min / max | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 vs 0.66 | 22 | 0.0554 ± 0.0577 | 0.0375 | 0.0205 / 0.0671 | -0.0304 / 0.2276 | 4.5% |
| 0.4 vs 1.0 | 22 | 0.0473 ± 0.0467 | 0.0308 | 0.0125 / 0.0676 | 0.0028 / 0.1750 | 0.0% |
| 0.66 vs 1.0 | 22 | 0.0718 ± 0.0547 | 0.0400 | 0.0314 / 0.1050 | 0.0128 / 0.1795 | 0.0% |

## Internal concentration and natural sample shares

| Width | C mean | C median | C P25 / P75 | Share mean | Share median | Share P10 / P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.6179 | 0.6068 | 0.5223 / 0.6653 | 0.3481 | 0.3184 | 0.2044 / 0.5109 |
| 0.66 | 0.5502 | 0.5262 | 0.4813 / 0.5937 | 0.4143 | 0.4283 | 0.2289 / 0.5793 |
| 1.0 | 0.6785 | 0.7174 | 0.5997 / 0.7304 | 0.2376 | 0.2105 | 0.1716 / 0.3005 |

Because every retained Full group has at least two occurrences, the Full concentration values above are no longer mechanically equal to one from a single occurrence.  They remain reasonably structured rather than indicating total within-group cancellation.

## Counterfactual geometry

Equal-width is retained only as a reference, not a recommendation.

| Counterfactual | Median cosine | Mean cosine | Median angle | Mean angle | Angle P25 / P75 | Median / mean norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal-width | 0.9535 | 0.9472 | 17.54° | 17.14° | 11.56° / 20.12° | 1.0012 / 1.0055 |
| Full=20% | 0.9981 | 0.9924 | 3.44° | 5.00° | 1.28° / 7.70° | 1.0033 / 1.0202 |

## Phase results

All phases have at least three retained rounds, so no phase is marked insufficient descriptive coverage.

| Phase | Valid | Full vs 0.4 median cosine | Full vs 0.66 median cosine | Full-share median | Full=20% median / mean angle | Full=20% median norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early R1-15 | 5 | 0.0944 | 0.1326 | 0.1926 | 2.33° / 3.11° | 0.9991 |
| Middle R16-32 | 9 | 0.0257 | 0.0330 | 0.1984 | 1.64° / 4.33° | 0.9997 |
| Late R33-49 | 8 | 0.0165 | 0.0609 | 0.2847 | 7.05° / 6.92° | 1.0481 |

## Before/after comparison

The “all Full-present” column contains the 40 rounds with all three groups and any positive Full occurrence count.  Delta is `Full>=2 - all Full-present`.

| Metric | All Full-present | Full>=2 | Delta |
| --- | ---: | ---: | ---: |
| Full vs 0.4 median cosine | 0.0279 | 0.0308 | +0.0029 |
| Full vs 0.66 median cosine | 0.0354 | 0.0400 | +0.0046 |
| Full concentration median | 0.7512 | 0.7174 | -0.0338 |
| Full natural-share median | 0.1668 | 0.2105 | +0.0437 |
| Full=20% median angle | 9.97° | 3.44° | -6.53° |
| Full=20% median norm ratio | 0.9972 | 1.0033 | +0.0061 |
| Late Full=20% median angle | 11.32° | 7.05° | -4.27° |

## Direction judgment

**geometric freedom largely explained by single-client rounds**.  After requiring at least two Full occurrences, group directions remain distinct and Full concentration remains non-degenerate, but moving Full to the conservative 20% share has a much smaller overall median rotation (3.44° versus 9.97°).  The Late rotation remains 7.05° but is based on eight rounds and is reduced from 11.32°.  This is a geometry-only conclusion; it does not establish any scale direction as superior and does not authorize an aggregation implementation or training run.
