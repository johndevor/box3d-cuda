# Narrow MuJoCo 3.3.7 inertia admission r2

This numerical-only revision does not alter XML, runtime, physics, r1 checker
or r1 evidence. It targets the pinned CPU compiler, not arbitrary simulators.
The original head failure stays rejected; r2 must be a separate attempt.

## Reproduction

`scripts/mujoco337_inertia_numeric.py` adapts the Apache-2.0 source
`src/user/user_util.cc` at MuJoCo 3.3.7, SHA
`f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522`.
It preserves scalar multiply/add order, quaternion normalization skipped within
1e-14, internal plus caller normalization, quaternion matrix formula, maximum
off-diagonal tie rules, the two stopping conditions and three sorting passes.
Nonfinite intermediate values and the 500-iteration cap reject. It imports no
simulator. The captured head stops at iteration 4 on the cosine condition and
matches the captured compiled reconstruction within 1.31e-18 kg m².

## Error bound, not a fitted tolerance

Let A be a symmetric positive-definite source tensor, a=||A||F, R the
PRE-sort frame, and D the computed RᵀAR. Every iteration recomputes D from A,
so there is no 500-step matrix-drift assumption. For Schur tangent t and
diagonal gap g, |Dij|=g|t|/(1−t²). The cosine stop c>1−e bounds
|t| by sqrt(e(2−e))/(1−e). Use e=1e-12+32u, u=2^-53, with outward
rounding. The guard covers scalar stop arithmetic; the reproduced trace must
also independently satisfy the resulting off-diagonal bound.

For dot3, gamma5=5u/(1−5u). Set
eta >= gamma5(2+gamma5)||R||F² a to cover the two matrix multiplications,
and delta >= ||RRᵀ−I||F, including its dot3 evaluation error. SPD gives
G=(1+delta)a+2eta as a conservative computed diagonal-gap bound. Define
M=G*tmax/(1−tmax²). Reject the absolute-stop-dominated domain M<1e-12;
the compiler can materially erase off-diagonals in such tiny tensors, even if
the reproduction matches perfectly. This explicitly limits r2's numerical
domain; it is not a universal tiny-inertia adapter.

The pre-sort tensor error is bounded by
delta(2+delta)a + (1+delta)(sqrt(6)M+6eta), plus the analogous
two-multiply error for R diag(D) Rᵀ. Six eta conservatively covers upper/lower
off-diagonal and diagonal rounding. Norms use hypot, bounds round outward,
nonfinite/unsupported scales reject. Actual source→pre-sort residual must fit
this bound; the code does not assume a successful source trace suffices.

## Two mandatory comparison layers

1. Compiled tensor versus PRE-sort compiler reconstruction must satisfy BOTH
   original entrywise `1e-10 + 1e-8*abs(reference)` and relative Frobenius
   `1e-8*||A||F`. The latter reuses the original relative tolerance and prevents
   a fixed absolute tolerance from hiding small-scale unit/frame errors.
2. Source versus pre-sort reconstruction must satisfy the derived bound.
   The triangle inequality bounds source→actual by that bound plus the limited
   comparison error. Post-sort reproduction must also satisfy the tight relative
   consistency check; no unbounded measured sorting error is added to the budget.

Sorting changes principal-axis representation, not the tensor. Comparing to
PRE-sort reconstruction avoids having to infer an extra sorting-error allowance.
Positivity and the physical inertia triangle are checked on source and actual
spectra (roundoff-only triangle allowance). All original mass/COM/body orientation,
joint/actuator/site/sensor/collision/reset/physics-health checks remain unchanged.
Near-degenerate eigenvectors are not individually compared: the full tensor is.

## Bounded execution condition

Only after independent numeric tests and exact r1→r2 source-delta review may
one fresh local reset→1→8-frame CPU attempt run in the existing isolated runtime,
under a 60-second parent cap, with the identical pinned model and fixture inputs.
Stop on its first failure; do not alter a second failed check or repeat the run.
No policy, weights, pickle, new dependencies, GPU, provider, native ABI or World
changes. A passing bounded fixture would not establish walking or CUDA parity.
