# Future direction: capacity-aware dual memory

This document is a design record only. No memory, routing, aggregation, loss,
or client-ranking mechanism is implemented in this branch.

If the fixed profile-quota oracle later demonstrates an accuracy benefit, the
next direction should use a comparable signal to maintain two small memories:

```text
Comparable Signal
        |
Semantic Memory + Reliability Memory
        |
Adaptive Profile Allocation
        |
Random client selection within each feasible pool
        |
Aggregation
```

The first use of the memories should be deciding how many training slots each
profile or capacity group receives. It should not concentrate aggregation
weight on a small set of apparently best clients. Client-level semantic routing
and memory-guided aggregation remain later hypotheses and require separate
controlled validation.
