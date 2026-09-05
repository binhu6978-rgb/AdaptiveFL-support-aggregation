# Fixed 4:3:3 quota offline dispatch simulation

This is a deterministic 500-round dispatch simulation (`seed=1`), not FL training. Each round has exactly four Small slots (profiles 0–2 chosen uniformly), three Medium slots (profiles 3–5 chosen uniformly), and three Full slots (profile 6); the 10 slots are shuffled. Client draws use the original feasible pools and continue to allow duplicate clients.

## Simulated profile exposure

- Profile 0–6 counts: [648, 715, 637, 480, 507, 513, 1500]
- Small / Medium / Full counts: 2000 / 1500 / 1500
- Full occurrences per round: 3.00
- Full zero-occurrence-round ratio: 0.00%

| Region | Slot total | Sample-weighted total | LR-weighted sample proxy | Ratio to baseline occurrence | Ratio to baseline samples | Ratio to baseline LR proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_small | 5000 | 2478980 | 15701.67 | 1.000000 | 0.998809 | 0.999883 |\n| medium_capable | 3000 | 1479852 | 9373.85 | 1.049318 | 1.043342 | 1.044141 |\n| full_exclusive | 1500 | 730535 | 4625.73 | 2.077562 | 2.065578 | 2.046295 |\n