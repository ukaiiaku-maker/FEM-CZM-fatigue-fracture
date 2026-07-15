import pytest

from fem_czm.v9102_parity import V9102RunConfig, run_v9102_parity


def test_dbtt_700K_reproduces_archived_v9103_spatial_response():
    result = run_v9102_parity(
        "DBTT",
        V9102RunConfig(
            temperature_K=700.0,
            target_extension_um=1000.0,
            dK_MPa_sqrt_m=0.25,
            Kdot_MPa_sqrt_m_per_s=0.005,
            Kmax_MPa_sqrt_m=80.0,
            advance_increment_um=5.0,
        ),
    )
    metrics = result.metrics

    assert metrics["completed"] is True
    assert metrics["right_censored_at_Kmax"] is False
    assert metrics["n_events"] == 201
    assert metrics["K_init_MPa_sqrt_m"] == pytest.approx(
        28.123898, abs=2.0e-3
    )
    assert metrics["K_plateau_MPa_sqrt_m"] == pytest.approx(
        30.963772, abs=2.0e-3
    )
    assert metrics["max_tip_radius_ratio"] == pytest.approx(
        1.009401, abs=2.0e-5
    )
    assert metrics["final_emitted_total"] == pytest.approx(
        539.531349, abs=2.0e-2
    )


def test_parity_blunting_ledger_is_source_slip_not_line_crossings():
    result = run_v9102_parity(
        "DBTT",
        V9102RunConfig(
            temperature_K=700.0,
            target_extension_um=25.0,
            dK_MPa_sqrt_m=0.25,
            Kdot_MPa_sqrt_m_per_s=0.005,
            Kmax_MPa_sqrt_m=80.0,
            advance_increment_um=5.0,
        ),
    )
    assert result.events
    first = result.events[0]
    # The source ledger is committed once at emission. It is not multiplied by
    # the number of subsequent spatial-bin crossings.
    assert first["local_slip_count"] <= first["emitted_total"] + 1.0e-12
