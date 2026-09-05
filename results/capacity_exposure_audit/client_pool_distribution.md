# Client-pool label-distribution audit

This is an offline pooled-label analysis of the existing cached split; it does not assess feature difficulty, optimization, model-conditioned utility, or finite-round client-selection variance.

## Strong pool (clients 70–99)

- Clients / samples: 30 / 14696
- TVD to actual all-client distribution: 0.122221
- JSD to actual all-client distribution: 0.018742 bits
- L1 / cosine / entropy difference: 0.244442 / 0.956031 / -0.071943 bits
- Coverage: 100/100 classes

## Random-subset null tests (10,000 simulations)

| Null | Metric | Mean | Median | P5 | P25 | P75 | P95 | Actual percentile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random 30 | TVD | 0.109016 | 0.108848 | 0.095952 | 0.103493 | 0.114415 | 0.122568 | 94.53% |
| Random 30 | JSD | 0.013597 | 0.013497 | 0.010660 | 0.012287 | 0.014817 | 0.016855 | 99.42% |
| Random 60 | TVD | 0.058367 | 0.058260 | 0.051337 | 0.055389 | 0.061285 | 0.065575 | 48.13% |
| Random 60 | JSD | 0.003912 | 0.003887 | 0.003035 | 0.003527 | 0.004272 | 0.004873 | 35.85% |

## Classification

**STRONGLY_BIASED** for pooled label-distribution evidence only.
