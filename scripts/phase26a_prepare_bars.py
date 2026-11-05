"""Prepare raw entry and forward bars for the frozen Phase25 OOF keys.

This is a read-only research preparation step.  It reads the OOF key order,
the Phase21 raw-file inventory, the Phase25 observed calendar, and the two
existing outcome artifacts for independent checks.  The only files written
are the three Phase26A artifacts named in :func:`main`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from scripts.run_phase21_research import ROOT, read_prices
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_phase21_research import ROOT, read_prices
from src.universe_policy import guard_cached_artifact


KEYS = ["exchange", "symbol", "date"]
OOF = ROOT / "data" / "research" / "phase25" / "oof_predictions.parquet"
PHASE25_STOCK_DAYS = ROOT / "data" / "research" / "phase25" / "stock_days.parquet"
PHASE24_OUTCOMES = ROOT / "data" / "research" / "phase24a" / "outcomes.parquet"
UNIVERSE_AUDIT = ROOT / "outputs" / "phase21" / "universe_audit.csv"
CALENDAR = ROOT / "outputs" / "phase25" / "observed_calendar.csv"
LOCAL_OUT = ROOT / "data" / "research" / "phase26a" / "entry_bars.parquet"
HEALTH_OUT = ROOT / "outputs" / "phase26a" / "bar_preparation.json"
HASH_OUT = ROOT / "outputs" / "phase26a" / "raw_source_hashes.json"

BOARD_RATIOS = {
    "SSE_MAIN": 0.10,
    "SZSE_MAIN": 0.10,
    "STAR": 0.20,
    "CHINEXT": 0.20,
    "BSE": 0.30,
}


def _clean(value):
    if value is pd.NA or value is pd.NaT:
        return None
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


def _dump_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise AssertionError(f"MISSING_KEY_COLUMNS:{missing}")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame[KEYS].isna().any().any():
        raise AssertionError("NULL_KEYS")
    return frame


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    oof = _normalise_keys(pd.read_parquet(OOF, columns=KEYS))
    if oof.duplicated(KEYS).any():
        raise AssertionError("OOF_DUPLICATE_KEYS")

    audit = pd.read_csv(UNIVERSE_AUDIT, dtype={"symbol": str})
    audit["symbol"] = audit["symbol"].astype(str).str.zfill(6)
    required_audit = {
        "exchange", "symbol", "board", "day_path", "sha256", "invalid_rows",
        "eligible_first_date", "delisting_date",
    }
    missing = sorted(required_audit - set(audit.columns))
    if missing:
        raise AssertionError(f"UNIVERSE_AUDIT_MISSING_COLUMNS:{missing}")
    if audit.duplicated(["exchange", "symbol"]).any():
        raise AssertionError("UNIVERSE_AUDIT_DUPLICATE_KEYS")
    audit["eligible_first_date"] = pd.to_datetime(audit["eligible_first_date"], errors="raise")
    audit["delisting_date"] = pd.to_datetime(audit["delisting_date"], errors="coerce")
    audit_map = audit.set_index(["exchange", "symbol"], drop=False)

    calendar_frame = pd.read_csv(CALENDAR)
    if "date" not in calendar_frame.columns:
        raise AssertionError("CALENDAR_DATE_MISSING")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_frame["date"], errors="raise")).normalize()
    calendar = calendar.sort_values().drop_duplicates()
    if len(calendar) < 2:
        raise AssertionError("CALENDAR_TOO_SHORT")
    if not oof.date.isin(calendar).all():
        raise AssertionError("OOF_DATE_OUTSIDE_PHASE25_CALENDAR")

    phase24 = _normalise_keys(
        pd.read_parquet(
            PHASE24_OUTCOMES,
            columns=KEYS + [
                "next_day_date", "outcome_next_day_gap", "outcome_next_day_one_price_up",
            ],
        )
    )
    if phase24.duplicated(KEYS).any():
        raise AssertionError("PHASE24_DUPLICATE_KEYS")
    phase24["next_day_date"] = pd.to_datetime(phase24["next_day_date"], errors="coerce").dt.normalize()

    phase25 = _normalise_keys(
        pd.read_parquet(
            PHASE25_STOCK_DAYS,
            columns=KEYS + [
                "future5_complete", "outcome_max_return_1_5d", "outcome_max_return_2_5d",
            ],
        )
    )
    if phase25.duplicated(KEYS).any():
        raise AssertionError("PHASE25_STOCK_DAYS_DUPLICATE_KEYS")
    return oof, audit_map, calendar, phase24, phase25


def _cent(values: np.ndarray) -> np.ndarray:
    return np.rint(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * 100).astype(np.int64)


def _same_float(left: np.ndarray, right: np.ndarray, atol: float = 1e-12) -> tuple[bool, int, int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    both_nan = np.isnan(left) & np.isnan(right)
    both_finite = np.isfinite(left) & np.isfinite(right)
    equal_finite = both_finite & np.isclose(left, right, rtol=0.0, atol=atol)
    equal = both_nan | equal_finite
    return bool(equal.all()), int(equal.sum()), int((~equal).sum())


def _same_nullable_bool(left: np.ndarray, right: pd.Series) -> tuple[bool, int, int]:
    left = np.asarray(left, dtype=object)
    right_values = right.astype("boolean").to_numpy(dtype=object)
    equal = np.zeros(len(left), dtype=bool)
    for i, (a, b) in enumerate(zip(left, right_values)):
        a_missing = a is None or pd.isna(a)
        b_missing = b is None or pd.isna(b)
        equal[i] = (a_missing and b_missing) or (not a_missing and not b_missing and bool(a) == bool(b))
    return bool(equal.all()), int(equal.sum()), int((~equal).sum())


def prepare() -> tuple[pd.DataFrame, dict, list[dict]]:
    oof, audit_map, calendar, phase24, phase25 = _load_inputs()
    row_count = len(oof)
    output = oof[KEYS].copy()
    for column in [
        "close_t", "next_open", "next_high", "next_low", "next_close", "next_volume",
        "future_high_2", "future_high_3", "future_high_4", "future_high_5",
    ]:
        output[column] = np.nan
    output["next_day_date"] = pd.NaT
    output["next_bar_present"] = False
    output["future2_5_complete"] = False
    output["nominal_limit_ratio"] = np.nan
    output["board"] = pd.Series(pd.NA, index=output.index, dtype="string")

    raw_hashes: list[dict] = []
    groups = oof.groupby(["exchange", "symbol"], sort=False).indices
    derived_gap = np.full(row_count, np.nan, dtype=float)
    derived_one_price_up = np.full(row_count, np.nan, dtype=object)
    derived_max_1_5 = np.full(row_count, np.nan, dtype=float)
    derived_max_2_5 = np.full(row_count, np.nan, dtype=float)
    source_missing_bar_rows = 0
    source_missing_close_rows = 0
    source_future2_5_incomplete_rows = 0

    for processed, (key, positions) in enumerate(groups.items(), 1):
        if key not in audit_map.index:
            raise AssertionError(f"MISSING_UNIVERSE_AUDIT:{key[0]}:{key[1]}")
        meta = audit_map.loc[key]
        if isinstance(meta, pd.DataFrame):
            raise AssertionError(f"DUPLICATE_UNIVERSE_AUDIT:{key[0]}:{key[1]}")
        board = str(meta.board)
        if board not in BOARD_RATIOS:
            raise AssertionError(f"UNKNOWN_BOARD_RATIO:{key[0]}:{key[1]}:{board}")

        raw_path = (ROOT / str(meta.day_path)).resolve()
        bars, invalid_rows, observed_hash = read_prices(raw_path)
        hash_match = observed_hash.lower() == str(meta.sha256).lower()
        invalid_match = int(invalid_rows) == int(meta.invalid_rows)
        record = {
            "exchange": key[0],
            "symbol": key[1],
            "path": str(raw_path.relative_to(ROOT)),
            "expected_sha256": str(meta.sha256),
            "observed_sha256": observed_hash,
            "expected_invalid_rows": int(meta.invalid_rows),
            "observed_invalid_rows": int(invalid_rows),
            "hash_match": hash_match,
            "invalid_rows_match": invalid_match,
        }
        raw_hashes.append(record)
        if not hash_match or not invalid_match:
            raise AssertionError(f"RAW_SOURCE_CHECK_FAILED:{key[0]}:{key[1]}")

        bars = bars.loc[bars.index >= pd.Timestamp(meta.eligible_first_date)]
        if pd.notna(meta.delisting_date):
            bars = bars.loc[bars.index <= pd.Timestamp(meta.delisting_date)]
        bars = bars.reindex(calendar)
        shifted = bars.shift(-1)
        future = {
            offset: bars["high"].shift(-offset).to_numpy(dtype=float)
            for offset in range(1, 6)
        }

        rows = oof.iloc[positions]
        date_positions = calendar.get_indexer(pd.DatetimeIndex(rows["date"]))
        if (date_positions < 0).any():
            raise AssertionError(f"OOF_DATE_NOT_ON_CALENDAR:{key[0]}:{key[1]}")
        close = bars["close"].to_numpy(dtype=float)[date_positions]
        next_values = shifted[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)[date_positions]
        future_values = {offset: values[date_positions] for offset, values in future.items()}

        next_present = np.isfinite(next_values).all(axis=1)
        close_valid = np.isfinite(close) & (close > 0)
        future2_complete = close_valid & np.isfinite(
            np.column_stack([future_values[offset] for offset in range(2, 6)])
        ).all(axis=1)
        future5_complete = close_valid & np.isfinite(
            np.column_stack([future_values[offset] for offset in range(1, 6)])
        ).all(axis=1)
        valid_next = next_present & close_valid

        output.loc[positions, "close_t"] = close
        output.loc[positions, "next_open"] = next_values[:, 0]
        output.loc[positions, "next_high"] = next_values[:, 1]
        output.loc[positions, "next_low"] = next_values[:, 2]
        output.loc[positions, "next_close"] = next_values[:, 3]
        output.loc[positions, "next_volume"] = next_values[:, 4]
        for offset in range(2, 6):
            output.loc[positions, f"future_high_{offset}"] = future_values[offset]
        next_dates = np.full(len(positions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        has_next_calendar = date_positions + 1 < len(calendar)
        next_dates[has_next_calendar] = calendar[date_positions[has_next_calendar] + 1].to_numpy(dtype="datetime64[ns]")
        output.loc[positions, "next_day_date"] = next_dates
        output.loc[positions, "next_bar_present"] = next_present
        output.loc[positions, "future2_5_complete"] = future2_complete
        output.loc[positions, "nominal_limit_ratio"] = BOARD_RATIOS[board]
        output.loc[positions, "board"] = board

        derived_gap[positions[valid_next]] = next_values[valid_next, 0] / close[valid_next] - 1.0
        close_cents = _cent(close)
        next_cents = _cent(next_values)
        one_price = (
            (next_cents[:, 0] == next_cents[:, 1])
            & (next_cents[:, 0] == next_cents[:, 2])
            & (next_cents[:, 0] == next_cents[:, 3])
            & (next_cents[:, 0] > close_cents)
        )
        derived_one_price_up[positions[valid_next]] = one_price[valid_next]
        high_1_5 = np.column_stack([future_values[offset] for offset in range(1, 6)])
        high_2_5 = np.column_stack([future_values[offset] for offset in range(2, 6)])
        derived_max_1_5[positions[future5_complete]] = high_1_5[future5_complete].max(axis=1) / close[future5_complete] - 1.0
        derived_max_2_5[positions[future2_complete]] = high_2_5[future2_complete].max(axis=1) / close[future2_complete] - 1.0

        source_missing_bar_rows += int((~next_present).sum())
        source_missing_close_rows += int((~close_valid).sum())
        source_future2_5_incomplete_rows += int((~future2_complete).sum())
        if processed % 1000 == 0:
            print(json.dumps({
                "processed_securities": processed,
                "securities_total": len(groups),
                "rows_so_far": int(sum(len(groups[group]) for group in list(groups)[:processed])),
            }), flush=True)

    expected_keys = oof[KEYS].reset_index(drop=True)
    actual_keys = output[KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_keys, expected_keys, check_dtype=False, check_exact=True)

    # Independent Phase24 checks are keyed and order-independent.
    phase24_ref = phase24.merge(oof[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if len(phase24_ref) != row_count:
        raise AssertionError(f"PHASE24_KEY_COVERAGE:{len(phase24_ref)}:{row_count}")
    phase24_ref = phase24_ref.set_index(KEYS).reindex(pd.MultiIndex.from_frame(oof[KEYS]))
    gap_ok, gap_equal, gap_failed = _same_float(
        derived_gap,
        phase24_ref["outcome_next_day_gap"].to_numpy(dtype=float),
    )
    one_ok, one_equal, one_failed = _same_nullable_bool(
        derived_one_price_up,
        phase24_ref["outcome_next_day_one_price_up"].reset_index(drop=True),
    )
    date_ok, date_equal, date_failed = _same_float(
        pd.to_datetime(output["next_day_date"]).astype("int64").to_numpy(dtype=float),
        pd.to_datetime(phase24_ref["next_day_date"].reset_index(drop=True)).astype("int64").to_numpy(dtype=float),
        atol=0.0,
    )
    if not gap_ok or not one_ok or not date_ok:
        raise AssertionError(f"PHASE24_OUTCOME_MISMATCH:gap={gap_failed}:one={one_failed}:date={date_failed}")

    # Phase25's max-return outcomes are only defined when its original
    # future5_complete flag is true.  Compare exactly those keyed reference
    # values; future2_5_complete intentionally remains a narrower raw-bar
    # completeness indicator.
    phase25_ref = phase25.merge(oof[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if len(phase25_ref) != row_count:
        raise AssertionError(f"PHASE25_KEY_COVERAGE:{len(phase25_ref)}:{row_count}")
    phase25_ref = phase25_ref.set_index(KEYS).reindex(pd.MultiIndex.from_frame(oof[KEYS]))
    max1_ok, max1_equal, max1_failed = _same_float(
        derived_max_1_5,
        phase25_ref["outcome_max_return_1_5d"].to_numpy(dtype=float),
    )
    # Phase25 censors both max-return fields when t+1 is missing, even when
    # the narrower t+2..t+5 raw window is complete.  Apply that source mask
    # only to this independent comparison; the emitted future2_5_complete
    # field remains the literal t+2..t+5 completeness check.
    max2_for_phase25 = derived_max_2_5.copy()
    max2_for_phase25[~phase25_ref["future5_complete"].to_numpy(dtype=bool)] = np.nan
    max2_ok, max2_equal, max2_failed = _same_float(
        max2_for_phase25,
        phase25_ref["outcome_max_return_2_5d"].to_numpy(dtype=float),
    )
    if not max1_ok or not max2_ok:
        raise AssertionError(f"PHASE25_OUTCOME_MISMATCH:max1={max1_failed}:max2={max2_failed}")

    output["next_day_date"] = pd.to_datetime(output["next_day_date"])
    output["next_bar_present"] = output["next_bar_present"].astype(bool)
    output["future2_5_complete"] = output["future2_5_complete"].astype(bool)
    output["board"] = output["board"].astype("string")

    health = {
        "status": "PASS",
        "phase": "26A",
        "rows": row_count,
        "securities": len(groups),
        "oof_key_order_preserved": True,
        "missing_bar_rows": source_missing_bar_rows,
        "missing_close_rows": source_missing_close_rows,
        "future2_5_incomplete_rows": source_future2_5_incomplete_rows,
        "calendar_sessions": len(calendar),
        "calendar_first": str(calendar[0].date()),
        "calendar_last": str(calendar[-1].date()),
        "raw_hash_checks": {
            "count": len(raw_hashes),
            "passed": int(sum(r["hash_match"] for r in raw_hashes)),
            "failed": int(sum(not r["hash_match"] for r in raw_hashes)),
            "invalid_rows_passed": int(sum(r["invalid_rows_match"] for r in raw_hashes)),
            "invalid_rows_failed": int(sum(not r["invalid_rows_match"] for r in raw_hashes)),
        },
        "alignment_checks": {
            "output_key_rows": row_count,
            "output_key_order_exact": True,
            "phase24_key_rows": len(phase24_ref),
            "phase24_next_day_date_equal": date_equal,
            "phase24_next_day_gap_equal": gap_equal,
            "phase24_one_price_up_equal": one_equal,
            "phase25_stock_days_key_rows": len(phase25_ref),
            "phase25_future5_complete_rows": int(phase25_ref["future5_complete"].sum()),
            "phase25_max_return_1_5d_equal": max1_equal,
            "phase25_max_return_2_5d_equal": max2_equal,
        },
        "missing_bar_policy": "Missing bars retain NaN fields and are not classified as trading halts.",
        "nominal_limit_ratio_policy": "Board proxy only: 0.10 for SSE_MAIN/SZSE_MAIN, 0.20 for STAR/CHINEXT, 0.30 for BSE; no legal limit-up judgment.",
        "entry_qualification_written": False,
        "true_limit_up_judgment_written": False,
        "amount_used": False,
        "outputs": {
            "entry_bars": str(LOCAL_OUT.relative_to(ROOT)),
            "bar_preparation": str(HEALTH_OUT.relative_to(ROOT)),
            "raw_source_hashes": str(HASH_OUT.relative_to(ROOT)),
        },
    }
    return output, health, raw_hashes


def main() -> None:
    guard_cached_artifact(OOF)
    guard_cached_artifact(PHASE25_STOCK_DAYS)
    guard_cached_artifact(PHASE24_OUTCOMES)
    output, health, raw_hashes = prepare()
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(LOCAL_OUT, index=False)
    _dump_json(HEALTH_OUT, health)
    _dump_json(HASH_OUT, raw_hashes)
    print(json.dumps(_clean(health), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
