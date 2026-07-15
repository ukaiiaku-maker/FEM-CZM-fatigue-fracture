# Physics contract

## One constitutive model for fracture and fatigue

Monotonic and cyclic loading call the same `CrackFront.step` method. Fatigue is
not a Paris-law overlay and does not have a separate damage barrier. It changes
only the applied history `K(t)` and the numerical integration strategy.

## Reusable, geometry-dependent sources

There is no available-site counter, one-emission rule, fixed source capacity, or
crack-advance source refresh. The former class-specific source counts and refresh
lengths are archived only for traceability and are never read by the active
equations.

For the reduced 2-D crack-tip model, the expected number of geometrically
accessible sources on each active slip system is

```text
N_source = theta_active r_eff / l_source,
```

where `theta_active` is the active crack-tip arc angle and `l_source` is a source
spacing. A sharper tip samples a shorter arc and therefore fewer sources; a
blunter tip samples more. The expected population may be fractional. This is a
geometric source density, not a consumable inventory.

A reusable source has a nucleation waiting time and a reload/reconstruction time:

```text
rate_per_source = lambda_nuc / (1 + lambda_nuc tau_reload).
```

The law approaches the Arrhenius nucleation rate at low hazard and approaches
`1/tau_reload` at high hazard. Source spacing, active angle, and reload time are
shared physical parameters exposed on the command line and must be subjected to
sensitivity analysis before calibration.

## Emergent emission suppression

Near-tip mobile and retained lines create a same-system pile-up resistance that
reduces the stress entering the emission barrier. Retained lines additionally
produce a direct unresolved `K_shield` contribution that reduces both emission
and cleavage tip stresses. Mobile lines do not directly contribute to
`K_shield` in the reduced baseline, but they do oppose immediate repeat emission
while they remain near the source.

Bulk plastic strain and its stress redistribution belong to the 2-D FEM field
and are already included in `K_J`. The front-local shielding term must therefore
represent only unresolved retained lines and must be subtracted once.

## Blunting

Stationary mobile population does not blunt the crack. The reduced model records
line-bin crossings as an irreversible glide/slip ledger and uses only this local
slip to increase `r_eff`. When the crack advances, old slip translates into the
wake with the other moving-process-zone fields. In the full 2-D model, the
preferred replacement is a tip radius or crack-tip opening measured directly
from the deformed cohesive/FEM geometry.

## Event-limited crack advance

A constitutive call may commit at most one crack increment. The solver stops at
the first cleavage-clock crossing, advances the crack, and returns unused time
to the caller. The caller then recomputes the process zone and tip mechanics at
the same external load. Hazard accumulated beyond a crossing in the obsolete tip
geometry is discarded rather than converted into thousands of crack increments.

## Peierls--Taylor kinetics

Peierls and Taylor use independent `H0`, activation entropy, `alpha`, and `n`.
Both inherit the emission stress scale. Net rates are forward minus zero-stress
reverse rates. Taylor completion has an uncapped density-dependent hit order.
There is no Taylor stress-amplification cap, hit-order cap, mobile-density
saturation, jump-length floor, or constitutive plastic-rate cap.

## Backstress limitation

The reduced process-zone kernel is a same-sign pile-up approximation. It is not
a signed 2-D GND or Nye-tensor formulation. The full 2-D implementation should
evolve slip-system-resolved signed GND fields or a Nye tensor and resolve their
stress tensor onto each source. Scalar total density may be used for Taylor
forest resistance but must not be converted into a signed backstress.
