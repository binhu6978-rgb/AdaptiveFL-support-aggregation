# Strong-pool repairability

Classification: **POORLY_REPAIRABLE**

Uniform pooled baseline: TVD 0.122221; JSD 0.018742 bits.

| lambda | TVD | JSD | ESS alpha | ESS pi | max pi | top-3 | top-5 | TVD reduction | JSD reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.112346 | 0.015190 | 23.29 | 22.14 | 0.0984 | 0.2197 | 0.3272 | 8.08% | 18.95% |
| 0.0001 | 0.112335 | 0.015192 | 23.36 | 22.21 | 0.0981 | 0.2189 | 0.3261 | 8.09% | 18.94% |
| 0.001 | 0.112305 | 0.015214 | 23.91 | 22.80 | 0.0952 | 0.2124 | 0.3176 | 8.11% | 18.83% |
| 0.01 | 0.113432 | 0.015604 | 26.50 | 25.96 | 0.0778 | 0.1789 | 0.2734 | 7.19% | 16.75% |
| 0.1 | 0.117688 | 0.017317 | 28.59 | 29.52 | 0.0470 | 0.1265 | 0.2012 | 3.71% | 7.60% |
| 1 | 0.121496 | 0.018518 | 28.65 | 29.99 | 0.0352 | 0.1037 | 0.1714 | 0.59% | 1.20% |
| 10 | 0.122146 | 0.018718 | 28.62 | 30.00 | 0.0335 | 0.1004 | 0.1672 | 0.06% | 0.13% |

## Selected diversity-constrained candidate

Constraint tier: strict; lambda=0; TVD/JSD=0.112346/0.015190; ESS alpha/pi=23.29/22.14; max pi=0.0984.

Finite 722-occurrence TVD mean: uniform 0.124861, candidate 0.114293; JSD mean: uniform 0.019440, candidate 0.015860.

This is a label-oracle upper-bound analysis; labels are not used to define any training mechanism.
