# Comparable-signal feasibility check

The seven ResNet profiles all instantiate `fc` as
`nn.Linear(int(512 * tmp_scale), num_classes)`. Consequently, `fc.weight` has
profile-dependent input width and is not directly comparable. In contrast,
`fc.bias` has the identical shape `(100,)` in every profile and directly
receives cross-entropy gradient through the 100-class logits.

Therefore a future short probe can use the local `fc.bias` delta or gradient as
a natural, fixed-dimensional, zero-architecture-change comparable signal.
It is a limited classifier-head signal rather than a replacement for a complete
representation comparison, so it should be validated before any memory policy
depends on it.

No Comparable Module is currently needed. If future evidence rejects the bias
signal, the least invasive module would be a fixed-dimensional projection of
the pooled final representation immediately before `fc`; that alternative is
not implemented here.
