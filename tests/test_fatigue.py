from fem_czm.fatigue import FatigueConfig, FatigueIntegrator
from fem_czm.front import CrackFront
from fem_czm.parameters import get_material
from fem_czm.process_zone import ProcessZoneConfig


def test_fatigue_uses_same_front_state_and_accumulates_time():
    front = CrackFront(get_material("DBTT"), process_zone_config=ProcessZoneConfig(n_bins=20))
    fatigue = FatigueIntegrator(front, FatigueConfig(frequency_Hz=10.0, phase_points=8, max_cycles_per_chunk=2.0))
    out = fatigue.advance_cycles(20e6, 700.0, 4.0)
    assert out["cycles_total"] == 4.0
    assert front.time_s > 0.0
    assert out["source_inventory_active"] == 0.0


def test_higher_fatigue_drive_does_not_reduce_emission_for_fresh_state():
    def run(K):
        front = CrackFront(get_material("weakT"), process_zone_config=ProcessZoneConfig(n_bins=20))
        fatigue = FatigueIntegrator(front, FatigueConfig(frequency_Hz=10.0, phase_points=8, max_cycles_per_chunk=1.0))
        return fatigue.advance_cycles(K, 700.0, 1.0)["dN_emit_block"]
    assert run(25e6) >= run(15e6)
