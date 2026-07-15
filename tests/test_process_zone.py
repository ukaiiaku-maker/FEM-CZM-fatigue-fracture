import numpy as np
from fem_czm.parameters import get_material
from fem_czm.process_zone import ProcessZoneConfig, ReusableSourceProcessZone


def test_no_source_counter_or_refresh_state():
    pz = ReusableSourceProcessZone(get_material("DBTT"), ProcessZoneConfig(n_bins=20))
    assert not hasattr(pz, "available_sites")
    out = pz.evolve(1e-6, 700.0, 30e6)
    assert out["source_inventory_active"] == 0.0
    assert out["source_refresh_active"] == 0.0


def test_retained_lines_create_shielding_and_backstress():
    pz = ReusableSourceProcessZone(get_material("DBTT"), ProcessZoneConfig(n_bins=20))
    pz.retained[:, 0] = 5.0
    assert pz.shielding_K_Pa_sqrt_m() > 0.0
    assert np.all(pz.source_backstress_Pa() > 0.0)
    assert pz.effective_tip_stress_Pa(30e6) < 30e6 / np.sqrt(2*np.pi*pz.cfg.r0_m)


def test_crack_advance_moves_old_state_to_wake_without_source_refresh():
    pz = ReusableSourceProcessZone(get_material("weakT"), ProcessZoneConfig(n_bins=20))
    pz.retained[:, :3] = 1.0
    before = pz.retained_count
    wake = pz.advance(2.0 * pz.dx)
    assert pz.retained_count < before
    assert wake["wake_retained"] > 0.0
