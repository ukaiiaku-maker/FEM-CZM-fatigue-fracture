from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fem_czm_analysis.k_vs_t import analyze_campaign


def _write_case(root: Path, option: str, temperature: int, values: list[float]) -> None:
    case = root / option / f"T{temperature:04d}"
    case.mkdir(parents=True)
    lines = []
    for index, value in enumerate(values, start=1):
        a_mm = 0.500 + 0.005 * index
        lines.append(
            f"  [T={temperature}K] step {100 * index:4d}  KJ={value:7.3f}  "
            f"sig_tip=  5.00GPa  B=0.000  N_em=0.00  a={a_mm:.3f}mm  "
            "nfr=1  << ADVANCE"
        )
    (case / "run.log").write_text("\n".join(lines) + "\n")
    (case / "barrier_only_case_status_v10_0_5_13.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "target_completed": True,
                "target_extension_um": 20.0,
            }
        )
    )


def test_event_weighted_summary_from_case_logs(tmp_path: Path):
    root = tmp_path / "campaign"
    root.mkdir()
    plan = {
        "options": ["ceramic_primary", "dbtt_primary"],
        "temperatures_K": [300, 700],
    }
    (root / "barrier_only_campaign_plan_v10_0_5_13.json").write_text(json.dumps(plan))

    _write_case(root, "ceramic_primary", 300, [10.0, 12.0, 14.0, 16.0])
    _write_case(root, "ceramic_primary", 700, [8.0, 9.0, 10.0, 11.0])
    _write_case(root, "dbtt_primary", 300, [20.0, 22.0, 24.0, 26.0])
    _write_case(root, "dbtt_primary", 700, [30.0, 34.0, 38.0, 42.0])

    output = tmp_path / "plots"
    summary = analyze_campaign(
        root,
        output_dir=output,
        formats=("png",),
        late_fraction=0.25,
        min_late_events=2,
        strict=True,
    )

    row = summary[
        (summary["option_key"] == "ceramic_primary") & (summary["T_K"] == 300)
    ].iloc[0]
    assert row["K_initial_MPa_sqrt_m"] == 10.0
    assert row["K_mean_MPa_sqrt_m"] == 13.0
    assert row["K_late_MPa_sqrt_m"] == 15.0
    assert row["n_growth_events"] == 4
    assert row["n_late_events"] == 2
    assert row["source_kind"] == "log"

    assert (output / "K_vs_T_event_summary.csv").is_file()
    assert (output / "K_vs_T_event_summary.json").is_file()
    assert (output / "K_initial_vs_T.png").is_file()
    assert (output / "K_mean_vs_T.png").is_file()
    assert (output / "K_late_vs_T.png").is_file()


def test_r_curve_csv_is_reduced_to_one_value_per_advance(tmp_path: Path):
    root = tmp_path / "campaign"
    case = root / "peak_primary" / "T0900"
    case.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_index": [1, 2, 3, 4],
            "crack_extension_um": [5.0, 10.0, 15.0, 20.0],
            "KJ_reference_MPa_sqrt_m": [15.0, 17.0, 19.0, 21.0],
        }
    ).to_csv(case / "r_curve.csv", index=False)
    (case / "case_status.json").write_text(
        json.dumps({"status": "complete", "target_completed": True})
    )
    (root / "campaign_plan.json").write_text(
        json.dumps({"options": ["peak_primary"], "temperatures_K": [900]})
    )

    summary = analyze_campaign(
        root,
        output_dir=tmp_path / "output",
        formats=("png",),
        late_fraction=0.5,
        min_late_events=1,
        strict=True,
    )
    row = summary.iloc[0]
    assert row["source_kind"] == "csv"
    assert row["K_initial_MPa_sqrt_m"] == 15.0
    assert row["K_mean_MPa_sqrt_m"] == 18.0
    assert row["K_late_MPa_sqrt_m"] == 20.0
    assert np.isclose(row["K_late_std_MPa_sqrt_m"], 1.0)
