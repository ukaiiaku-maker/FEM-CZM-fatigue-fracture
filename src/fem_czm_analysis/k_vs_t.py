"""Event-based K-versus-temperature analysis for FEM/CZM campaigns.

The analysis deliberately uses crack-advance events rather than arbitrary solver
print steps.  This prevents long waiting intervals from biasing the mean K value.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KNOWN_OPTION_ORDER = (
    "ceramic_primary",
    "weakT_primary",
    "dbtt_primary",
    "peak_primary",
)

OPTION_LABELS = {
    "ceramic_primary": "ceramic-like",
    "weakT_primary": "weak T",
    "dbtt_primary": "DBTT",
    "peak_primary": "toughness peak",
}

OPTION_MARKERS = {
    "ceramic_primary": "o",
    "weakT_primary": "s",
    "dbtt_primary": "^",
    "peak_primary": "D",
}

CASE_RE = re.compile(r"^T(?P<T>\d{4})$")
ADVANCE_RE = re.compile(
    r"\[T=(?P<T>\d+)K\]\s+step\s+(?P<step>\d+)"
    r"\s+KJ=\s*(?P<K>[+\-0-9.eE]+).*?"
    r"a=(?P<a_mm>[+\-0-9.eE]+)mm.*?<<\s*ADVANCE",
    re.IGNORECASE,
)
PREFIXED_ADVANCE_RE = re.compile(
    r"\[(?P<option>[^/\]]+)/T(?P<Tprefix>\d+)K\].*?"
    r"\[T=(?P<T>\d+)K\]\s+step\s+(?P<step>\d+)"
    r"\s+KJ=\s*(?P<K>[+\-0-9.eE]+).*?"
    r"a=(?P<a_mm>[+\-0-9.eE]+)mm.*?<<\s*ADVANCE",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EventHistory:
    events: pd.DataFrame
    source_kind: str
    source_path: Path


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _dict_rows(value: object) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _campaign_plan(outroot: Path) -> dict:
    candidates = sorted(outroot.glob("*campaign_plan*.json"))
    for path in candidates:
        data = _read_json(path)
        if isinstance(data, dict) and data.get("options") and data.get("temperatures_K"):
            return data
    return {}


def _discover_case_dirs(outroot: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for option_dir in outroot.iterdir() if outroot.is_dir() else []:
        if not option_dir.is_dir():
            continue
        for case_dir in option_dir.iterdir():
            if not case_dir.is_dir():
                continue
            match = CASE_RE.match(case_dir.name)
            if match:
                found[(option_dir.name, int(match.group("T")))] = case_dir
    return found


def _case_status(case_dir: Path) -> tuple[str, bool | None]:
    paths = sorted(case_dir.glob("*case_status*.json"))
    for path in paths:
        rows = _dict_rows(_read_json(path))
        if not rows:
            continue
        row = rows[0]
        status = str(row.get("status", "unknown"))
        target_completed = row.get("target_completed")
        if isinstance(target_completed, bool):
            return status, target_completed
        return status, None
    return "unknown", None


def _filename_priority(path: Path) -> int:
    name = _norm(path.stem)
    score = 0
    if "rcurve" in name:
        score += 100
    if "advance" in name:
        score += 80
    if "event" in name:
        score += 60
    if "growth" in name:
        score += 30
    if "history" in name:
        score += 10
    if "summary" in name:
        score -= 25
    return score


def _choose_k_column(columns: Iterable[object]) -> object | None:
    scored: list[tuple[int, object]] = []
    for raw in columns:
        name = _norm(raw)
        score = -1
        if name in {
            "kjmpasqrtm",
            "kjreferencempasqrtm",
            "kjreferencefirstmpasqrtm",
            "kcmpasqrtm",
            "kmpasqrtm",
        }:
            score = 100
        elif name.startswith("kj") and ("sqrt" in name or "mpa" in name):
            score = 90
        elif name.startswith("kc") and ("sqrt" in name or "mpa" in name):
            score = 75
        elif name == "kj":
            score = 70
        elif name == "k":
            score = 20
        if "first" in name:
            score -= 25
        if score >= 0:
            scored.append((score, raw))
    return max(scored, default=(-1, None), key=lambda item: item[0])[1]


def _choose_event_column(columns: Iterable[object]) -> object | None:
    preferred = (
        "eventindex",
        "growthevent",
        "advanceindex",
        "advanceevent",
        "event",
    )
    normalized = {_norm(raw): raw for raw in columns}
    for name in preferred:
        if name in normalized:
            return normalized[name]
    return None


def _choose_position_column(columns: Iterable[object]) -> tuple[object | None, str]:
    normalized = {_norm(raw): raw for raw in columns}
    candidates = (
        ("crackextensionum", "extension_um"),
        ("extensionum", "extension_um"),
        ("deltaaum", "extension_um"),
        ("aum", "absolute_um"),
        ("amm", "absolute_mm"),
        ("cracklengthum", "absolute_um"),
        ("cracklengthmm", "absolute_mm"),
        ("cracklengthm", "absolute_m"),
    )
    for name, kind in candidates:
        if name in normalized:
            return normalized[name], kind
    for name, raw in normalized.items():
        if "extension" in name and name.endswith("um"):
            return raw, "extension_um"
    return None, ""


def _position_um(values: pd.Series, kind: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if kind == "absolute_mm":
        numeric = numeric * 1.0e3
    elif kind == "absolute_m":
        numeric = numeric * 1.0e6
    if kind.startswith("absolute") and numeric.notna().any():
        numeric = numeric - float(numeric.min())
    return numeric


def _csv_event_history(path: Path) -> EventHistory | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None

    k_column = _choose_k_column(frame.columns)
    if k_column is None:
        return None

    event_column = _choose_event_column(frame.columns)
    position_column, position_kind = _choose_position_column(frame.columns)
    priority = _filename_priority(path)
    if event_column is None and position_column is None and priority <= 0:
        return None

    work = pd.DataFrame(
        {
            "row_order": np.arange(len(frame), dtype=int),
            "K_MPa_sqrt_m": pd.to_numeric(frame[k_column], errors="coerce"),
        }
    )
    if event_column is not None:
        work["event_index"] = pd.to_numeric(frame[event_column], errors="coerce")
    else:
        work["event_index"] = np.nan
    if position_column is not None:
        work["position_um"] = _position_um(frame[position_column], position_kind)
    else:
        work["position_um"] = np.nan

    work = work[np.isfinite(work["K_MPa_sqrt_m"]) & (work["K_MPa_sqrt_m"] > 0.0)]
    if work.empty:
        return None

    if work["event_index"].notna().any():
        work = (
            work.sort_values("row_order")
            .groupby("event_index", sort=True, as_index=False)
            .first()
        )
    elif work["position_um"].notna().any():
        counts = work.groupby("position_um")["row_order"].count()
        work = (
            work.sort_values("row_order")
            .groupby("position_um", sort=True, as_index=False)
            .first()
        )
        # Generic step-history files usually contain many pre-growth rows at the
        # minimum crack length.  Event/R-curve files normally do not.
        if priority <= 0 and len(work) > 1:
            first_position = float(work["position_um"].min())
            if int(counts.get(first_position, 0)) > 1:
                work = work[work["position_um"] > first_position]
    else:
        work = work.sort_values("row_order")

    work = work.reset_index(drop=True)
    if work.empty:
        return None
    work["event_order"] = np.arange(1, len(work) + 1, dtype=int)
    return EventHistory(work, "csv", path)


def _best_csv_history(case_dir: Path) -> EventHistory | None:
    candidates: list[tuple[tuple[int, int], EventHistory]] = []
    for path in case_dir.rglob("*.csv"):
        history = _csv_event_history(path)
        if history is None:
            continue
        candidates.append(((_filename_priority(path), len(history.events)), history))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _log_event_history(path: Path) -> EventHistory | None:
    rows = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return None
    for line in lines:
        match = ADVANCE_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "event_index": len(rows) + 1,
                "event_order": len(rows) + 1,
                "step": int(match.group("step")),
                "K_MPa_sqrt_m": float(match.group("K")),
                "a_mm": float(match.group("a_mm")),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows).drop_duplicates(subset=["step", "a_mm"], keep="first")
    frame = frame.sort_values(["step", "a_mm"]).reset_index(drop=True)
    frame["event_order"] = np.arange(1, len(frame) + 1, dtype=int)
    frame["position_um"] = (frame["a_mm"] - float(frame["a_mm"].iloc[0])) * 1.0e3
    return EventHistory(frame, "log", path)


def _case_log_history(case_dir: Path) -> EventHistory | None:
    preferred = [case_dir / "run.log", case_dir / "console.log"]
    paths = [path for path in preferred if path.is_file()]
    paths.extend(path for path in sorted(case_dir.glob("*.log")) if path not in paths)
    candidates = [history for path in paths if (history := _log_event_history(path)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: len(item.events))


def _global_console_histories(outroot: Path) -> dict[tuple[str, int], EventHistory]:
    rows: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(outroot.glob("*.log")):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            match = PREFIXED_ADVANCE_RE.search(line)
            if not match:
                continue
            key = (match.group("option"), int(match.group("Tprefix")))
            rows.setdefault(key, []).append(
                {
                    "step": int(match.group("step")),
                    "K_MPa_sqrt_m": float(match.group("K")),
                    "a_mm": float(match.group("a_mm")),
                    "source_path": path,
                }
            )
    histories: dict[tuple[str, int], EventHistory] = {}
    for key, values in rows.items():
        frame = pd.DataFrame(values).drop_duplicates(subset=["step", "a_mm"], keep="first")
        frame = frame.sort_values(["step", "a_mm"]).reset_index(drop=True)
        frame["event_order"] = np.arange(1, len(frame) + 1, dtype=int)
        frame["position_um"] = (frame["a_mm"] - float(frame["a_mm"].iloc[0])) * 1.0e3
        histories[key] = EventHistory(frame, "campaign_log", Path(frame["source_path"].iloc[0]))
    return histories


def _first_passage_fallback(case_dir: Path) -> EventHistory | None:
    candidates = list(case_dir.glob("*first_passage*summary*.json"))
    candidates.extend(case_dir.glob("*case_status*.json"))
    keys = (
        "KJ_reference_first_MPa_sqrt_m",
        "Kc_first_existing_MPa_sqrt_m",
        "K_FP_MPa_sqrt_m",
    )
    for path in candidates:
        for row in _dict_rows(_read_json(path)):
            for key in keys:
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    frame = pd.DataFrame(
                        [{"event_order": 1, "event_index": 1, "K_MPa_sqrt_m": value}]
                    )
                    return EventHistory(frame, "first_passage_json", path)
    return None


def _select_history(case_dir: Path, global_history: EventHistory | None) -> EventHistory | None:
    csv_history = _best_csv_history(case_dir)
    log_history = _case_log_history(case_dir)
    candidates = [item for item in (csv_history, log_history, global_history) if item is not None]
    if candidates:
        # Prefer the most complete event history.  For ties, use CSV over logs.
        rank = {"csv": 3, "log": 2, "campaign_log": 1}
        return max(candidates, key=lambda item: (len(item.events), rank.get(item.source_kind, 0)))
    return _first_passage_fallback(case_dir)


def _summarize_history(
    history: EventHistory,
    late_fraction: float,
    min_late_events: int,
) -> dict[str, float | int | str]:
    values = pd.to_numeric(history.events["K_MPa_sqrt_m"], errors="coerce")
    values = values[np.isfinite(values) & (values > 0.0)].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("event history contains no finite positive K values")
    late_count = max(int(min_late_events), int(math.ceil(float(late_fraction) * len(values))))
    late_count = min(late_count, len(values))
    return {
        "K_initial_MPa_sqrt_m": float(values[0]),
        "K_mean_MPa_sqrt_m": float(np.mean(values)),
        "K_late_MPa_sqrt_m": float(np.mean(values[-late_count:])),
        "K_late_std_MPa_sqrt_m": float(np.std(values[-late_count:], ddof=0)),
        "n_growth_events": int(len(values)),
        "n_late_events": int(late_count),
        "source_kind": history.source_kind,
        "source_path": str(history.source_path),
    }


def _option_sort_key(option: str) -> tuple[int, str]:
    try:
        return KNOWN_OPTION_ORDER.index(option), option
    except ValueError:
        return len(KNOWN_OPTION_ORDER), option


def _plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    available = summary[np.isfinite(pd.to_numeric(summary[metric], errors="coerce"))]
    options = sorted(available["option_key"].unique(), key=_option_sort_key)
    fallback_markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, option in enumerate(options):
        subset = available[available["option_key"] == option].sort_values("T_K")
        axis.plot(
            subset["T_K"],
            subset[metric],
            marker=OPTION_MARKERS.get(option, fallback_markers[index % len(fallback_markers)]),
            linewidth=1.5,
            markersize=6.0,
            label=OPTION_LABELS.get(option, option),
        )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=True)
    axis.tick_params(direction="in", top=True, right=True)
    figure.tight_layout()
    for suffix in formats:
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def analyze_campaign(
    outroot: str | Path,
    *,
    output_dir: str | Path | None = None,
    late_fraction: float = 0.25,
    min_late_events: int = 3,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
    include_incomplete: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Analyze one campaign and write summary tables and K-versus-T plots."""
    root = Path(outroot).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"campaign OUTROOT does not exist: {root}")
    if not 0.0 < late_fraction <= 1.0:
        raise ValueError("late_fraction must be in (0, 1]")
    if min_late_events < 1:
        raise ValueError("min_late_events must be at least 1")

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "analysis_K_vs_T"
    )
    destination.mkdir(parents=True, exist_ok=True)

    plan = _campaign_plan(root)
    discovered = _discover_case_dirs(root)
    global_histories = _global_console_histories(root)

    if plan:
        expected = [
            (str(option), int(temperature))
            for option in plan.get("options", [])
            for temperature in plan.get("temperatures_K", [])
        ]
    else:
        expected = sorted(discovered, key=lambda key: (_option_sort_key(key[0]), key[1]))

    rows: list[dict] = []
    for option, temperature in expected:
        case_dir = discovered.get((option, temperature))
        base = {
            "option_key": option,
            "option_label": OPTION_LABELS.get(option, option),
            "T_K": int(temperature),
            "case_dir": "" if case_dir is None else str(case_dir),
        }
        if case_dir is None:
            rows.append({**base, "analysis_status": "missing_case_directory"})
            continue

        status, target_completed = _case_status(case_dir)
        base.update({"run_status": status, "target_completed": target_completed})
        if not include_incomplete and (
            status not in {"complete", "unknown"} or target_completed is False
        ):
            rows.append({**base, "analysis_status": "skipped_incomplete"})
            continue

        history = _select_history(case_dir, global_histories.get((option, temperature)))
        if history is None:
            rows.append({**base, "analysis_status": "missing_event_history"})
            continue
        try:
            metrics = _summarize_history(history, late_fraction, min_late_events)
        except ValueError as exc:
            rows.append({**base, "analysis_status": "invalid_event_history", "error": str(exc)})
            continue
        rows.append({**base, "analysis_status": "ok", **metrics})

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError(f"no campaign cases were discovered under {root}")

    summary["_option_order"] = summary["option_key"].map(
        {key: index for index, key in enumerate(KNOWN_OPTION_ORDER)}
    ).fillna(len(KNOWN_OPTION_ORDER))
    summary = summary.sort_values(["_option_order", "option_key", "T_K"]).drop(
        columns="_option_order"
    )

    summary_path = destination / "K_vs_T_event_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "schema": "fem_czm_event_K_vs_T_v1",
        "campaign_outroot": str(root),
        "late_definition": "mean of final fraction of crack-advance events",
        "late_fraction": float(late_fraction),
        "min_late_events": int(min_late_events),
        "event_weighting": "one K value per accepted crack advance",
        "summary_csv": str(summary_path),
        "n_expected_cases": int(len(expected)),
        "n_analyzed_cases": int((summary["analysis_status"] == "ok").sum()),
        "status_counts": summary["analysis_status"].value_counts(dropna=False).to_dict(),
    }
    (destination / "K_vs_T_event_summary.json").write_text(json.dumps(metadata, indent=2))

    _plot_metric(
        summary,
        "K_initial_MPa_sqrt_m",
        r"Initial crack-advance $K_J$ (MPa$\sqrt{m}$)",
        destination / "K_initial_vs_T",
        formats,
        dpi,
    )
    _plot_metric(
        summary,
        "K_mean_MPa_sqrt_m",
        r"Mean crack-advance $K_J$ (MPa$\sqrt{m}$)",
        destination / "K_mean_vs_T",
        formats,
        dpi,
    )
    _plot_metric(
        summary,
        "K_late_MPa_sqrt_m",
        rf"Late-growth mean $K_J$ (final {100.0 * late_fraction:g}% of events; MPa$\sqrt{{m}}$)",
        destination / "K_late_vs_T",
        formats,
        dpi,
    )

    failures = summary[summary["analysis_status"] != "ok"]
    print(f"Campaign: {root}")
    print(f"Analyzed cases: {(summary['analysis_status'] == 'ok').sum()}/{len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Plots: {destination}")
    if not failures.empty:
        print("Cases not analyzed:")
        print(failures[["option_key", "T_K", "analysis_status"]].to_string(index=False))
    if strict and not failures.empty:
        raise RuntimeError(f"strict analysis failed for {len(failures)} case(s)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate event-weighted initial, mean, and late-growth K-versus-T "
            "plots from an FEM/CZM campaign directory."
        )
    )
    parser.add_argument("outroot", type=Path, help="campaign output root")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=0.25,
        help="fraction of final crack-advance events used for late K (default: 0.25)",
    )
    parser.add_argument(
        "--min-late-events",
        type=int,
        default=3,
        help="minimum number of events included in late K (default: 3)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze_campaign(
        args.outroot,
        output_dir=args.output_dir,
        late_fraction=args.late_fraction,
        min_late_events=args.min_late_events,
        formats=args.formats,
        dpi=args.dpi,
        include_incomplete=args.include_incomplete,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
