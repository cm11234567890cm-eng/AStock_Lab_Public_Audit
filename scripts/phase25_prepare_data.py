"""Build the frozen 250-session Phase25 research table.

This module is research-only.  It reads the Phase21 universe and Phase23C
event-presence artifacts, keeps a 60-session warm-up, and writes only the
Phase25 artifacts.  It never changes the universe, production gates, or any
Phase23C/Phase24A artifact.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from src.universe_policy import predictive_universe, report_audit, guard_cached_artifact

try:
    from scripts.run_phase21_research import ROOT, read_prices
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_phase21_research import ROOT, read_prices

from src.research.coverage_recovery import (
    FEATURE_WINDOWS,
    RULE_DEPENDENCIES,
    CORE_SHORT_FEATURE_SET,
    eligibility,
    interval_flags,
)
from src.research.label_censoring import (
    feature_continuity,
    fixed_signals,
    intrinsic_validity,
    price_volume_values,
)


OUT = ROOT / "outputs" / "phase25"
LOCAL = ROOT / "data" / "research" / "phase25"
AUDIT = ROOT / "outputs" / "phase21" / "universe_audit.csv"
CALENDAR = ROOT / "outputs" / "phase21" / "observed_market_calendar.csv"
OLD_STOCK = ROOT / "data" / "research" / "phase23c" / "stock_days.parquet"
OLD_OUTCOMES = ROOT / "data" / "research" / "phase24a" / "outcomes.parquet"
EVENT_TRUTH = ROOT / "outputs" / "phase23c" / "CA_EVENT_TRUTH.csv"
BOUNDARIES = ROOT / "outputs" / "phase23c" / "historical_price_continuity_boundaries.csv"
CERTIFICATES = ROOT / "outputs" / "phase23c" / "presence_source_certificates.csv"
KEYS = ["exchange", "symbol", "date"]
WARMUP_SESSIONS = 60
RESEARCH_SESSIONS = 250
CALENDAR_SESSIONS = WARMUP_SESSIONS + RESEARCH_SESSIONS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def nullable_bool(values: np.ndarray, valid: np.ndarray) -> pd.Series:
    out = pd.Series(pd.array([pd.NA] * len(values), dtype="boolean"))
    out.iloc[valid] = values[valid].astype(bool)
    return out


def _cent(values: np.ndarray) -> np.ndarray:
    return np.rint(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * 100).astype(np.int64)


def _outcomes(bars: pd.DataFrame, close: pd.Series, future5_complete: pd.Series) -> pd.DataFrame:
    """Reproduce the Phase24A raw OHLC outcome contract exactly."""
    future = pd.concat([bars.high.shift(-offset) for offset in range(1, 6)], axis=1)
    future.columns = [f"high_{offset}" for offset in range(1, 6)]
    high = future.to_numpy(dtype=float)
    entry = close.to_numpy(dtype=float)
    complete = future5_complete.to_numpy(dtype=bool) & np.isfinite(entry) & (entry > 0)
    close_cents = _cent(entry)
    high_cents = _cent(high)
    out = pd.DataFrame(index=bars.index)
    max_2_5 = np.full(len(bars), np.nan, dtype=float)
    max_1_5 = np.full(len(bars), np.nan, dtype=float)
    max_2_5[complete] = high[complete, 1:5].max(axis=1) / entry[complete] - 1.0
    max_1_5[complete] = high[complete, 0:5].max(axis=1) / entry[complete] - 1.0
    out["outcome_max_return_2_5d"] = max_2_5
    out["outcome_max_return_1_5d"] = max_1_5
    for threshold in (10, 20, 30):
        hit_2_5 = (high_cents[:, 1:5, None] * 100 >= close_cents[:, None, None] * (100 + threshold)).any(axis=(1, 2))
        hit_1_5 = (high_cents[:, 0:5, None] * 100 >= close_cents[:, None, None] * (100 + threshold)).any(axis=(1, 2))
        out[f"outcome_hit{threshold}_2_5d"] = nullable_bool(hit_2_5, complete).array
        out[f"outcome_hit{threshold}_1_5d"] = nullable_bool(hit_1_5, complete).array
    return out


def build_security(
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    ratio: float,
    cfg21: dict,
    cfg22: dict,
    event_dates=(),
    historical_boundaries=(),
    unknown_intervals=(),
    metadata: dict | None = None,
) -> pd.DataFrame:
    """Return all 310 calendar rows, including warm-up and missing bars."""
    if metadata is None:
        raise ValueError('PREDICTIVE_IDENTITY_UNVERIFIED: build_security requires metadata')
    report_audit(pd.DataFrame([metadata]))
    bars = bars.reindex(calendar)
    values = price_volume_values(bars, ratio, cfg22["feature_small_up_max"])
    # min(full consecutive run, 10) is equivalent to the bounded ten-session
    # expression; the overlap audit also verifies all frozen masked values.
    values["consecutive_small_up_days_10"] = values["consecutive_small_up_days"].clip(upper=10)
    intrinsic = intrinsic_validity(bars, values)
    valid, crossed, unproven = feature_continuity(
        calendar, intrinsic, historical_boundaries, unknown_intervals
    )
    raw_signals = fixed_signals(values, cfg21["rule_thresholds"], cfg22)
    eligible = eligibility(valid, "FULL_FEATURE_RESEARCH")
    label_crosses, label_unknown = interval_flags(
        calendar, event_dates, unknown_intervals, forward=5
    )
    future5 = pd.concat([bars.high.shift(-offset) for offset in range(1, 6)], axis=1)
    future5_complete = future5.notna().all(axis=1) & bars.close.notna() & bars.close.gt(0)
    label_valid = future5_complete & ~label_crosses & ~label_unknown

    payload = {
        "close_t": bars.close,
        "future5_complete": future5_complete.astype(bool),
        "LABEL_CROSSES_CA": pd.Series(label_crosses, index=calendar, dtype=bool),
        "LABEL_CA_PRESENCE_UNVERIFIED": pd.Series(label_unknown, index=calendar, dtype=bool),
        "label_valid": pd.Series(label_valid, index=calendar, dtype=bool),
    }
    for name in FEATURE_WINDOWS:
        payload[name] = values[name].where(valid[name])
        payload[name + "__valid"] = valid[name].astype(bool)
    payload["feature_valid_count"] = valid.sum(axis=1).astype("int64")
    long_names = [name for name, (old, _new) in FEATURE_WINDOWS.items() if old > 10]
    payload["long_feature_valid"] = valid[long_names].all(axis=1).astype(bool)
    for width in (5, 10, 20, 60):
        payload[f"ca_boundary_{width}"] = interval_flags(
            calendar, historical_boundaries, (), oldest=width
        )[0]
    for rule in "ABCDEFGHIJ":
        payload["signal_" + rule] = raw_signals[rule].astype(bool)
        payload["full_" + rule + "_strategy_eligible"] = eligible[rule].astype(bool)
        payload["full_signal_" + rule] = (raw_signals[rule] & eligible[rule]).astype(bool)
    payload["full_any_strategy_eligible"] = eligible.any(axis=1).astype(bool)
    payload["full_research_eligible"] = (label_valid & eligible.any(axis=1)).astype(bool)
    all_deps = sorted({d for ds in RULE_DEPENDENCIES.values() for d in ds})
    payload["full_any_feature_ca"] = crossed[all_deps].any(axis=1)
    payload["full_any_feature_unknown"] = unproven[all_deps].any(axis=1)
    payload["full_all_features_valid"] = valid[list(FEATURE_WINDOWS)].all(axis=1)
    out = pd.DataFrame(payload, index=calendar)
    outcome = _outcomes(bars, bars.close, future5_complete)
    out = pd.concat([out, outcome], axis=1)
    target = out["outcome_hit10_2_5d"].copy()
    target.loc[~label_valid] = pd.NA
    # Strictly t+2..t+5, with CA-invalid and incomplete labels censored.
    out["TARGET_2_5D_10"] = target.astype("boolean")
    # Preserve the legacy target name used by Phase23C for independent checks.
    legacy_target = out["outcome_hit10_1_5d"].copy()
    legacy_target.loc[~label_valid] = pd.NA
    out["TARGET_5D_10"] = legacy_target.astype("boolean")
    if metadata:
        for key, value in metadata.items():
            out[key] = value
    out.index.name = "date"
    return out


def _load_inputs():
    inv = pd.read_csv(AUDIT, dtype={"symbol": str})
    inv = predictive_universe(inv)
    inv["symbol"] = inv.symbol.astype(str).str.zfill(6)
    if inv.duplicated(["exchange", "symbol"]).any():
        raise AssertionError("DUPLICATE_UNIVERSE_AUDIT")
    all_calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(CALENDAR)["date"])).sort_values().drop_duplicates()
    if len(all_calendar) < CALENDAR_SESSIONS:
        raise AssertionError("OBSERVED_CALENDAR_SHORTER_THAN_310")
    calendar = all_calendar[-CALENDAR_SESSIONS:]
    research = calendar[-RESEARCH_SESSIONS:]
    truth = pd.read_csv(EVENT_TRUTH, dtype={"symbol": str})
    truth["symbol"] = truth.symbol.astype(str).str.zfill(6)
    if not truth.event_exists.all() or not truth.event_presence_qualified.all():
        raise AssertionError("CA_EVENT_TRUTH_NOT_FULLY_QUALIFIED")
    event_map = defaultdict(list)
    for row in truth.itertuples():
        event_map[(row.exchange, row.symbol)].append(pd.Timestamp(row.ex_date))
    boundary_rows = pd.read_csv(BOUNDARIES, dtype={"symbol": str})
    boundary_rows["symbol"] = boundary_rows.symbol.astype(str).str.zfill(6)
    boundary_map = defaultdict(list)
    for row in boundary_rows.itertuples():
        boundary_map[(row.exchange, row.symbol)].append(pd.Timestamp(row.effective_date))
    cert = pd.read_csv(CERTIFICATES, dtype={"symbol": str})
    cert["symbol"] = cert.symbol.astype(str).str.zfill(6)
    if cert.duplicated(["exchange", "symbol"]).any():
        raise AssertionError("DUPLICATE_PRESENCE_CERTIFICATE")
    required = {"full_history_query_success", "presence_source_qualified", "remaining_source_reason"}
    if not required.issubset(cert.columns):
        raise AssertionError("PRESENCE_CERTIFICATE_COLUMNS_MISSING")
    if not cert.full_history_query_success.all() or not cert.presence_source_qualified.all() or not cert.remaining_source_reason.eq("NONE").all():
        raise AssertionError("PRESENCE_CERTIFICATE_NOT_FULLY_QUALIFIED")
    cert_map = cert.set_index(["exchange", "symbol"])
    return inv, calendar, research, event_map, boundary_map, cert_map


def _validate_old120(data: pd.DataFrame) -> dict:
    old = pd.read_parquet(OLD_STOCK)
    old["symbol"] = old.symbol.astype(str).str.zfill(6)
    data_old = data[data.date >= pd.Timestamp("2026-04-09")].copy()
    feature_columns = list(FEATURE_WINDOWS) + [name + "__valid" for name in FEATURE_WINDOWS]
    compare = feature_columns + [
        "close_t", "future5_complete", "LABEL_CROSSES_CA", "LABEL_CA_PRESENCE_UNVERIFIED", "label_valid",
        *["full_signal_" + rule for rule in "ABCDEFGHIJ"], "full_any_strategy_eligible",
    ]
    left = data_old[KEYS + compare].sort_values(KEYS).reset_index(drop=True)
    right = old[KEYS + compare].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    outcomes = pd.read_parquet(OLD_OUTCOMES)
    outcomes["symbol"] = outcomes.symbol.astype(str).str.zfill(6)
    outcome_columns = [
        "outcome_max_return_2_5d", "outcome_max_return_1_5d",
        *[f"outcome_hit{n}_{h}_5d" for n in (10, 20, 30) for h in ("2", "1")],
    ]
    # The generated names are outcome_hit10_2_5d etc.; keep the explicit list
    # above readable while validating the actual columns.
    outcome_columns = [
        "outcome_max_return_2_5d", "outcome_max_return_1_5d",
        "outcome_hit10_2_5d", "outcome_hit10_1_5d", "outcome_hit20_2_5d",
        "outcome_hit20_1_5d", "outcome_hit30_2_5d", "outcome_hit30_1_5d",
    ]
    left_o = data_old[KEYS + outcome_columns].sort_values(KEYS).reset_index(drop=True)
    right_o = outcomes[KEYS + outcome_columns].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_o, right_o, check_dtype=False, check_exact=True)
    return {
        "phase23c_old120_feature_mask_signal_label_match": True,
        "phase24a_old120_outcome_match": True,
        "old120_rows": int(len(left)),
    }


def main() -> None:
    if (LOCAL / 'stock_days.parquet').exists():
        guard_cached_artifact(LOCAL / 'stock_days.parquet')
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    cfg21 = json.loads((ROOT / "config" / "phase21_research.json").read_text(encoding="utf-8"))
    cfg22 = json.loads((ROOT / "config" / "phase22_research.json").read_text(encoding="utf-8"))
    inv, calendar, research, event_map, boundary_map, cert_map = _load_inputs()
    parts = []
    source_hashes = []
    ratio_by_board = {"SSE_MAIN": .10, "SZSE_MAIN": .10, "STAR": .20, "CHINEXT": .20, "BSE": .30}
    for number, row in enumerate(inv.itertuples(index=False), 1):
        key = (row.exchange, row.symbol)
        if key not in cert_map.index:
            raise AssertionError(f"MISSING_PRESENCE_CERTIFICATE:{key}")
        raw_path = (ROOT / str(row.day_path)).resolve()
        bars, invalid_rows, observed_hash = read_prices(raw_path)
        if observed_hash.lower() != str(row.sha256).lower() or int(invalid_rows) != int(row.invalid_rows):
            raise AssertionError(f"RAW_SOURCE_CHECK_FAILED:{key}")
        source_hashes.append({
            "exchange": row.exchange, "symbol": row.symbol,
            "path": str(raw_path.relative_to(ROOT)),
            "expected_sha256": str(row.sha256), "observed_sha256": observed_hash,
            "expected_invalid_rows": int(row.invalid_rows), "observed_invalid_rows": int(invalid_rows),
        })
        bars = bars.loc[bars.index >= pd.Timestamp(row.eligible_first_date)]
        if pd.notna(row.delisting_date):
            bars = bars.loc[bars.index <= pd.Timestamp(row.delisting_date)]
        cert_reason = cert_map.loc[key, "remaining_source_reason"]
        metadata = {
            "symbol": row.symbol, "exchange": row.exchange, "board": row.board,
            "name": row.name, "current_listing_status": row.current_listing_status,
            "current_st": bool(row.current_st) if pd.notna(row.current_st) else False,
            "listing_date": row.listing_date, "delisting_date": row.delisting_date,
            "first_raw_bar_date": row.first_raw_bar_date,
            "eligible_first_date": row.eligible_first_date, "identity_source": row.identity_source,
            "presence_source_reason": cert_reason,
            "historical_st_status": "UNKNOWN",
        }
        sec = build_security(
            bars, calendar, ratio_by_board[row.board], cfg21, cfg22,
            event_dates=event_map[key], historical_boundaries=boundary_map[key],
            metadata=metadata,
        )
        if row.current_listing_status == "ACTIVE":
            sec.loc[pd.Timestamp(cfg21["as_of"]), "historical_st_status"] = "KNOWN_ST" if bool(row.current_st) else "KNOWN_NON_ST"
        sec = sec.loc[research].loc[sec.loc[research].close_t.notna()].copy()
        sec["market_pos"] = research.get_indexer(sec.index).astype(int)
        sec["month"] = sec.index.strftime("%Y-%m")
        sec["split"] = "PHASE25"
        parts.append(sec.reset_index())
        if number % 500 == 0:
            print(json.dumps({"processed_securities": number, "rows_so_far": sum(len(part) for part in parts)}), flush=True)
    data = pd.concat(parts, ignore_index=True)
    if data.duplicated(KEYS).any():
        raise AssertionError("PHASE25_DUPLICATE_KEYS")
    validation = _validate_old120(data)

    eligible = data["full_research_eligible"].astype(bool)
    complete = data["future5_complete"].astype(bool)
    coverage = []
    for month, group in data.groupby("month", sort=True):
        month_complete = int(group.future5_complete.sum())
        month_eligible = int(group.full_research_eligible.sum())
        coverage.append({
            "month": month, "raw_stock_days": int(len(group)),
            "label_complete_stock_days": month_complete,
            "research_eligible_stock_days": month_eligible,
            "coverage_pct": 100 * month_eligible / month_complete if month_complete else None,
            "raw_coverage_pct": 100 * month_eligible / len(group) if len(group) else None,
        })

    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_parquet(LOCAL / "stock_days.parquet", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "coverage_by_month.csv", index=False)
    pd.DataFrame({
        "date": calendar,
        "is_warmup": [True] * WARMUP_SESSIONS + [False] * RESEARCH_SESSIONS,
        "is_research_window": [False] * WARMUP_SESSIONS + [True] * RESEARCH_SESSIONS,
    }).to_csv(OUT / "observed_calendar.csv", index=False)
    dump_json(OUT / "source_hashes.json", source_hashes)
    health = {
        "bse_candidate_count": report_audit(data)["bse_candidate_count"],
        "status": "PHASE25_DATA_PREPARED",
        "scope": "RESEARCH_ONLY_NOT_TRADABLE_BACKTEST",
        "as_of": cfg21["as_of"], "observed_calendar_days": int(len(calendar)),
        "warmup_sessions": WARMUP_SESSIONS, "research_sessions": RESEARCH_SESSIONS,
        "warmup_start": str(calendar[0].date()), "warmup_end": str(calendar[WARMUP_SESSIONS - 1].date()),
        "research_start": str(research[0].date()), "research_end": str(research[-1].date()),
        "rows": int(len(data)), "securities": int(data.groupby(["exchange", "symbol"]).ngroups),
        "raw_stock_days": int(len(data)), "complete_label_stock_days": int(complete.sum()),
        "research_eligible_stock_days": int(eligible.sum()),
        "feature_windows": {name: list(window) for name, window in FEATURE_WINDOWS.items()},
        "source_hash_checks": {"count": len(source_hashes), "passed": len(source_hashes), "failed": 0},
        "identity_rules": "Phase21 universe_audit exchange+symbol+board; listing/delisting clipped; no member expansion",
        "st_rule": "UNKNOWN historically; current status only at as_of row, no backfill",
        "ca_truth": "Phase23C presence certificates fully qualified; event truth for label censoring; historical effective boundaries for past-only features; no publication or A/B parameters",
        "amount_used": False, "production_unchanged": True, "auto_trading": False,
        **validation,
        "output": str((LOCAL / "stock_days.parquet").relative_to(ROOT)),
    }
    dump_json(OUT / "data_health.json", health)
    print(json.dumps(_clean(health), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
