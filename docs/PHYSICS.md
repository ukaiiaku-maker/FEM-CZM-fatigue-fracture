# Physics contract

## One constitutive model for fracture and fatigue

Monotonic and cyclic loading call the same `CrackFront.step` method. Fatigue is
not a Paris-law overlay and does not have a separate damage barrier. It changes
only the applied history `K(t)` and the numerical integration strategy.

## Reusable sources

The emission hazard is evaluated every step from the local source stress. There
is no available-site counter, source capacity, one-emission rule, or source
refresh length. The former class-specific source counts and refresh lengths are
archived only for traceability and are never read by the active equations.

## Emergent emission suppression

Retained lines produce two distinct effects:

1. a slip-system source backstress that reduces the stress entering the emission
   barrier;
2. a direct unresolved `K_shield` contribution that reduces both emission and
   cleavage tip stresses.

Bulk plastic strain and its stress redistribution belong to the 2-D FEM field
and are already included in `K_J`. The front-local shielding term must therefore
represent only unresolved retained lines and must be subtracted once.

## Peierls--Taylor kinetics

Peierls and Taylor use independent `H0`, activation entropy, `alpha`, and `n`.
Both inherit the emission stress scale. Net rates are forward minus zero-stress
reverse rates. Taylor completion has an uncapped density-dependent hit order.
There is no Taylor stress-amplification cap, hit-order cap, mobile-density
saturation, jump-length floor, or constitutive plastic-rate cap.

## Signed backstress limitation

The reduced process-zone kernel is a front-local signed-line approximation. A
full 2-D implementation should evolve slip-system-resolved signed GND fields or
a Nye tensor. Scalar total density may be used for Taylor forest resistance but
must not be converted into a signed backstress.
