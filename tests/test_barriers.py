import numpy as np
from fem_czm.barriers import exp_floor_barrier_eV, transport_exp_floor_barrier_eV
from fem_czm.parameters import get_material


def test_independent_barriers_decrease_with_stress():
    m = get_material("DBTT")
    stress = np.array([0.0, 1e9, 5e9])
    for b in (m.cleavage, m.emission):
        G = exp_floor_barrier_eV(stress, 700.0, b)
        assert np.all(np.diff(G) <= 0.0)
    for mech in (m.peierls, m.taylor):
        G = transport_exp_floor_barrier_eV(stress, 700.0, mech, m.emission)
        assert np.all(np.diff(G) <= 0.0)


def test_three_parameterizations_have_no_active_source_inventory_fields():
    m = get_material("weakT")
    assert not hasattr(m.state, "source_sites_per_system")
    assert m.state.legacy_source_sites_per_system > 0.0
