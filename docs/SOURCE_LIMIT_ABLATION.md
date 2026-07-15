# Finite-source diagnostic

This branch adds an optional reproduction of the original v9 finite-site source
law. It is an ablation, not the proposed final constitutive model.

The diagnostic holds the cleavage/emission barriers, independent Peierls--Taylor
kinetics, transport, density normalization, blunting, crack-event integration and
loading protocol fixed. Only the source production rule is changed.

## Source modes

- `reusable_emergent`: geometry-dependent reusable sources with detailed-balance
  net directed emission and source backstress.
- `legacy_finite_site`: the original one-shot finite-site rule. Each system starts
  with the retained candidate's legacy site count. During an accepted interval,
  each available site emits with probability `1-exp(-H)`, where `H` is the
  integrated raw per-site Arrhenius hazard. Emitted sites are removed. Crack
  advance refreshes a fraction `min(da/L_refresh,1)` of the depleted inventory.
  Time recovery defaults to zero.

For the DBTT candidate the retained provenance values are 14.0087 sites per
system and a 54.7736 micrometre refresh length.

The finite-site mode deliberately does not use the radius-dependent geometric
source count, reload time, detailed-balance subtraction or source backstress in
the emission law. This makes the ablation as close as practical to the original
calibrated source closure while preserving the updated transport and crack-front
solver.
