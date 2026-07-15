import numpy as np

from fem_czm.finite_site_process_zone import (
    LegacyFiniteSiteConfig,
    LegacyFiniteSiteProcessZone,
)
from fem_czm.parameters import get_material


def test_finite_site_inventory_cannot_emit_more_than_capacity():
    cfg = LegacyFiniteSiteConfig(
        n_bins=20,
        source_sites_per_system=3.0,
        source_recovery_rate_s=0.0,
        source_refresh_length_m=10.0e-6,
    )
    pz = LegacyFiniteSiteProcessZone(get_material("DBTT"), cfg)
    total_capacity = float(np.sum(pz.site_capacity))
    first = pz.evolve(1.0e6, 700.0, 80.0e6)
    second = pz.evolve(1.0e6, 700.0, 80.0e6)
    assert pz.emitted_total <= total_capacity + 1.0e-12
    assert np.all(pz.available_sites >= 0.0)
    assert first["source_inventory_active"] == 1.0
    assert second["legacy_available_site_fraction"] <= first[
        "legacy_available_site_fraction"
    ]


def test_zero_time_recovery_never_replenishes_depleted_inventory():
    cfg = LegacyFiniteSiteConfig(
        n_bins=20,
        source_sites_per_system=5.0,
        source_recovery_rate_s=0.0,
    )
    pz = LegacyFiniteSiteProcessZone(get_material("DBTT"), cfg)
    pz.available_sites[:] = 2.0
    available = pz.available_sites.copy()
    pz.evolve(1000.0, 700.0, 0.0)
    assert np.all(pz.available_sites <= available)


def test_crack_advance_refreshes_original_linear_fraction():
    cfg = LegacyFiniteSiteConfig(
        n_bins=20,
        source_sites_per_system=10.0,
        source_recovery_rate_s=0.0,
        source_refresh_length_m=20.0e-6,
    )
    pz = LegacyFiniteSiteProcessZone(get_material("DBTT"), cfg)
    pz.available_sites[:] = 0.0
    out = pz.advance(5.0e-6)
    assert np.allclose(pz.available_sites, 2.5)
    assert np.isclose(out["source_sites_refreshed"], 5.0)


def test_dbtt_legacy_provenance_values_are_available():
    state = get_material("DBTT").state
    assert np.isclose(state.legacy_source_sites_per_system, 14.0087)
    assert np.isclose(state.legacy_source_refresh_length_um, 54.7736)
