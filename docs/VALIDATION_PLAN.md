# Validation plan

## Gate 1: constitutive regression

- reproduce all four independent EXP-floor surfaces from the selected tables;
- verify zero net Peierls/Taylor flow at zero stress;
- verify no finite-source state exists;
- verify shielding and source backstress have the expected sign and scaling;
- verify monotonic and fatigue drivers call the identical front state update.

## Gate 2: reduced spatial response

Run ceramic, weakT and DBTT at 300, 700, 900 and 1200 K. Because the source
inventory and source-refresh parameters have been removed, the old calibrated
curves are reference targets rather than expected exact reproductions. No
refitting is allowed until the emergent-shielding response is audited.

## Gate 3: 2-D FEM/CZM coupling

Port the adaptive mesh, interaction/domain integral, crack insertion,
unilateral contact, anisotropic elasticity and branching infrastructure. Verify:

- `K_J` contains resolved bulk-plastic shielding;
- front-local retained-line shielding is subtracted once;
- scalar forest density affects Taylor kinetics only;
- signed GND/backstress uses slip-system-resolved state;
- process-zone length and J contours are mesh independent.

## Gate 4: fatigue

For all three classes, calculate S-N initiation, threshold and `da/dN-DeltaK`
using the same state evolution as monotonic fracture. Audit cycle-block
convergence against explicit cycles and aggregate every adaptive sub-block in
event-averaged `da/dN`.
