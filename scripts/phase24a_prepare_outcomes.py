"""Prepare ex-post OHLCV outcomes for the Phase24A research audit.

This artifact is deliberately isolated from ranking inputs and strategy code.
It reads only the Phase23C keys/close/future-completeness columns and rebuilds
the outcome fields from the verified raw daily files.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

try:
    from scripts.run_phase21_research import ROOT, read_prices
except ModuleNotFoundError:  # Also support ``python scripts/phase24a_prepare_outcomes.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_phase21_research import ROOT, read_prices


OUT = ROOT / "outputs" / "phase24a"
LOCAL = ROOT / "data" / "research" / "phase24a"
STOCK_DAYS = ROOT / "data" / "research" / "phase23c" / "stock_days.parquet"
UNIVERSE_AUDIT = ROOT / "outputs" / "phase21" / "universe_audit.csv"
MARKET_CALENDAR = ROOT / "outputs" / "phase21" / "observed_market_calendar.csv"
KEYS = ["exchange", "symbol", "date"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_clean(value):
    """Convert pandas/numpy values to strict JSON values."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def safe_dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(_json_clean(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def safe_progress(value) -> None:
    print(json.dumps(_json_clean(value), ensure_ascii=False, allow_nan=False), flush=True)


def nullable_bool(values: np.ndarray, valid: np.ndarray) -> pd.Series:
    result = pd.Series(pd.array([pd.NA] * len(values), dtype="boolean"))
    result.iloc[valid] = values[valid].astype(bool)
    return result


def _cent_values(values: np.ndarray) -> np.ndarray:
    """Round prices exactly as the source price representation does."""
    return np.rint(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * 100).astype(np.int64)


def prepare() -> tuple[pd.DataFrame, dict]:
    required_stock = KEYS + ["close_t", "future5_complete"]
    stock = pd.read_parquet(STOCK_DAYS, columns=required_stock)
    stock["symbol"] = stock["symbol"].astype(str).str.zfill(6)
    stock["date"] = pd.to_datetime(stock["date"])
    if stock.duplicated(KEYS).any():
        raise AssertionError("PHASE23C_DUPLICATE_KEYS")

    audit = pd.read_csv(UNIVERSE_AUDIT, dtype={"symbol": str})
    audit["symbol"] = audit["symbol"].astype(str).str.zfill(6)
    required_audit = {
        "exchange", "symbol", "day_path", "sha256", "invalid_rows",
        "eligible_first_date", "delisting_date",
    }
    missing = sorted(required_audit - set(audit.columns))
    if missing:
        raise AssertionError(f"UNIVERSE_AUDIT_MISSING_COLUMNS:{missing}")
    audit_map = audit.set_index(["exchange", "symbol"], drop=False)

    calendar = pd.to_datetime(pd.read_csv(MARKET_CALENDAR)["date"])
    calendar = pd.DatetimeIndex(calendar).sort_values().drop_duplicates()
    if len(calendar) < 180:
        raise AssertionError("MARKET_CALENDAR_SHORTER_THAN_180")
    calendar = calendar[-180:]

    source_hashes = []
    source_hashes_by_key = {}
    parts = []
    processed = 0
    hash_failures = []
    invalid_failures = []

    # groupby(sort=False) gives the first-seen security order. Within each
    # group, iloc preserves the exact Phase23C row order for the final output.
    groups = stock.groupby(["exchange", "symbol"], sort=False, dropna=False).indices
    for key, positions in groups.items():
        if key not in audit_map.index:
            raise AssertionError(f"MISSING_UNIVERSE_AUDIT:{key[0]}:{key[1]}")
        meta = audit_map.loc[key]
        if isinstance(meta, pd.DataFrame):
            raise AssertionError(f"DUPLICATE_UNIVERSE_AUDIT:{key[0]}:{key[1]}")

        raw_path = (ROOT / str(meta.day_path)).resolve()
        bars, invalid_rows, observed_hash = read_prices(raw_path)
        hash_match = observed_hash.lower() == str(meta.sha256).lower()
        invalid_match = int(invalid_rows) == int(meta.invalid_rows)
        source_record = {
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
        source_hashes.append(source_record)
        source_hashes_by_key[key] = source_record
        if not hash_match:
            hash_failures.append(source_record)
        if not invalid_match:
            invalid_failures.append(source_record)
        if not hash_match or not invalid_match:
            raise AssertionError(f"RAW_SOURCE_CHECK_FAILED:{key[0]}:{key[1]}")

        lower = pd.Timestamp(meta.eligible_first_date)
        bars = bars.loc[bars.index >= lower]
        if pd.notna(meta.delisting_date):
            bars = bars.loc[bars.index <= pd.Timestamp(meta.delisting_date)]
        bars = bars.reindex(calendar)

        rows = stock.iloc[positions].copy().reset_index(drop=True)
        dates = pd.DatetimeIndex(rows["date"])
        if not dates.isin(calendar).all():
            raise AssertionError(f"PHASE23C_DATE_OUTSIDE_CALENDAR:{key[0]}:{key[1]}")
        if dates.duplicated().any():
            raise AssertionError(f"PHASE23C_DUPLICATE_SECURITY_DATE:{key[0]}:{key[1]}")

        # The Phase23C close is the fixed denominator. Rechecking it against
        # the raw file protects the outcome audit from accidental key drift.
        raw_close = bars["close"].reindex(dates).to_numpy(dtype=float)
        close = rows["close_t"].to_numpy(dtype=float)
        if not np.allclose(raw_close, close, equal_nan=False, rtol=0.0, atol=1e-12):
            raise AssertionError(f"PHASE23C_CLOSE_MISMATCH:{key[0]}:{key[1]}")
        close_valid = np.isfinite(close) & (close > 0)

        future = pd.concat(
            [bars["high"].shift(-offset) for offset in range(1, 6)],
            axis=1,
        )
        future.columns = [f"high_{offset}" for offset in range(1, 6)]
        future = future.reindex(dates)
        high_values = future.to_numpy(dtype=float)
        high_complete = np.isfinite(high_values).all(axis=1)
        complete5 = rows["future5_complete"].to_numpy(dtype=bool) & high_complete & close_valid

        next_bar = bars[["open", "high", "low", "close"]].shift(-1).reindex(dates)
        next_values = next_bar.to_numpy(dtype=float)
        next_complete = np.isfinite(next_values).all(axis=1) & close_valid
        next_open = next_values[:, 0]
        close_cents = _cent_values(close)
        high_cents = _cent_values(high_values)
        next_cents = _cent_values(next_values)

        result = rows[KEYS].copy()
        max_2_5 = np.full(len(rows), np.nan, dtype=float)
        max_1_5 = np.full(len(rows), np.nan, dtype=float)
        max_2_5[complete5] = high_values[complete5, 1:5].max(axis=1) / close[complete5] - 1.0
        max_1_5[complete5] = high_values[complete5, 0:5].max(axis=1) / close[complete5] - 1.0
        result["outcome_max_return_2_5d"] = max_2_5
        result["outcome_max_return_1_5d"] = max_1_5

        for threshold in (10, 20, 30):
            result[f"outcome_hit{threshold}_2_5d"] = nullable_bool(
                (high_cents[:, 1:5, None] * 100 >= close_cents[:, None, None] * (100 + threshold)).any(axis=(1, 2)),
                complete5,
            )
            result[f"outcome_hit{threshold}_1_5d"] = nullable_bool(
                (high_cents[:, 0:5, None] * 100 >= close_cents[:, None, None] * (100 + threshold)).any(axis=(1, 2)),
                complete5,
            )

        next_gap = np.full(len(rows), np.nan, dtype=float)
        next_gap[next_complete] = next_open[next_complete] / close[next_complete] - 1.0
        result["outcome_next_day_gap"] = next_gap
        gap_ge5 = next_cents[:, 0] * 100 >= close_cents * 105
        result["outcome_gap_ge5pct"] = nullable_bool(gap_ge5, next_complete)

        one_price_up = (
            (next_cents[:, 0] == next_cents[:, 1])
            & (next_cents[:, 0] == next_cents[:, 2])
            & (next_cents[:, 0] == next_cents[:, 3])
            & (next_cents[:, 0] > close_cents)
        )
        result["outcome_next_day_one_price_up"] = nullable_bool(one_price_up, next_complete)
        result["outcome_entry_assessment"] = np.where(
            ~next_complete,
            "NEXT_BAR_MISSING",
            np.where(one_price_up, "ONE_PRICE_UP_NO_FILL_PROOF", "INTRADAY_EXECUTABILITY_UNVERIFIED"),
        )

        date_positions = calendar.get_indexer(dates)
        next_dates = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
        has_calendar_next = date_positions + 1 < len(calendar)
        next_dates[has_calendar_next] = calendar[date_positions[has_calendar_next] + 1].to_numpy(dtype="datetime64[ns]")
        result["next_day_date"] = next_dates
        parts.append(result)

        processed += 1
        if processed % 1000 == 0:
            safe_progress({"processed_securities": processed, "outcome_rows_so_far": sum(len(part) for part in parts)})

    outcomes = pd.concat(parts, ignore_index=True)
    expected_keys = stock[KEYS].reset_index(drop=True)
    actual_keys = outcomes[KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_keys, expected_keys, check_dtype=False)

    # Outcome columns keep nullable booleans all the way into Parquet.
    boolean_columns = [
        c for c in outcomes.columns
        if c.startswith("outcome_hit") or c in ("outcome_gap_ge5pct", "outcome_next_day_one_price_up")
    ]
    for column in boolean_columns:
        outcomes[column] = outcomes[column].astype("boolean")
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(LOCAL / "outcomes.parquet", index=False)

    health = {
        "status": "PASS",
        "phase": "24A",
        "rows": int(len(outcomes)),
        "securities": int(len(groups)),
        "calendar_sessions_used": int(len(calendar)),
        "calendar_first": str(calendar[0].date()),
        "calendar_last": str(calendar[-1].date()),
        "input_stock_days": str(STOCK_DAYS.relative_to(ROOT)),
        "input_stock_days_sha256": file_sha256(STOCK_DAYS),
        "universe_audit": str(UNIVERSE_AUDIT.relative_to(ROOT)),
        "universe_audit_sha256": file_sha256(UNIVERSE_AUDIT),
        "observed_market_calendar": str(MARKET_CALENDAR.relative_to(ROOT)),
        "observed_market_calendar_sha256": file_sha256(MARKET_CALENDAR),
        "raw_hashchecks": {
            "count": int(len(source_hashes)),
            "passed": int(sum(bool(x["hash_match"]) for x in source_hashes)),
            "failed": int(len(hash_failures)),
            "invalid_rows_passed": int(sum(bool(x["invalid_rows_match"]) for x in source_hashes)),
            "invalid_rows_failed": int(len(invalid_failures)),
        },
        "source_hashes": source_hashes,
        "output": str((LOCAL / "outcomes.parquet").relative_to(ROOT)),
        "outcomes_not_used_for_ranking": True,
        "amount_read": False,
        "label_valid_written": False,
        "strategy_columns_written": False,
        "proxy_limitations": [
            "outcome_next_day_one_price_up is an OHLC equality proxy only.",
            "ONE_PRICE_UP_NO_FILL_PROOF does not establish a legal limit-up status or executable fill.",
            "INTRADAY_EXECUTABILITY_UNVERIFIED does not infer intraday order execution.",
        ],
    }
    safe_dump(OUT / "outcome_preparation.json", health)
    return outcomes, health


def main() -> None:
    outcomes, health = prepare()
    safe_progress({
        "status": health["status"],
        "rows": health["rows"],
        "securities": health["securities"],
        "calendar_first": health["calendar_first"],
        "calendar_last": health["calendar_last"],
        "raw_hashchecks": health["raw_hashchecks"],
        "output_rows_verified": len(outcomes),
    })


if __name__ == "__main__":
    main()
