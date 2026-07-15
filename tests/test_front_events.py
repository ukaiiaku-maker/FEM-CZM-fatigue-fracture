from fem_czm.front import CrackFront, FrontConfig
from fem_czm.parameters import get_material
from fem_czm.process_zone import ProcessZoneConfig


def test_precompleted_clock_commits_only_one_geometry_event():
    da = 5.0e-6
    front = CrackFront(
        get_material("DBTT"),
        FrontConfig(advance_increment_m=da),
        ProcessZoneConfig(n_bins=20),
    )
    front.cleavage_clock = 3.25
    out = front.step(20.0e6, 700.0, 1.0)
    assert out["n_fire"] == 1
    assert front.crack_extension_m == da
    assert front.n_advances == 1
    assert front.cleavage_clock == 0.0
    assert out["unused_dt_s"] == 1.0
    assert out["cleavage_clock_overflow_discarded"] == 2.25


def test_stationary_mobile_population_does_not_create_tip_blunting():
    front = CrackFront(
        get_material("DBTT"),
        process_zone_config=ProcessZoneConfig(n_bins=20),
    )
    r0 = front.process_zone.blunted_radius_m()
    front.process_zone.mobile[:, 0] = 1.0e6
    assert front.process_zone.blunted_radius_m() == r0
    assert front.process_zone.source_backstress_Pa().max() > 0.0
