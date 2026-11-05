"""Read-only TongDaXin (.day) discovery and structural verification.

This module never launches TDX and never writes beneath its installation root.
All artifacts are written to AStock_Lab only.
"""
from __future__ import annotations

import csv
import json
import math
import random
import struct
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DATE = date(2026, 9, 30)
RECORD = struct.Struct("<IIIIIfII")
RECORD_SIZE = RECORD.size
LEGACY_CSV_ROOT = Path(r"E:\2-5天\data\normalized\daily")


def classify_symbol(symbol: str, market: str | None = None) -> tuple[str, str]:
    """Classify from both market storage and code rules.

    Shanghai/Shenzhen folders contain industry/index codes beginning with 8.
    A 4/8 prefix is BSE only under the ``bj`` market directory.
    """
    if symbol.startswith(("688", "689")):
        return "SSE", "STAR"
    if symbol.startswith(("600", "601", "603", "605")):
        return "SSE", "SSE_MAIN"
    if symbol.startswith(("300", "301")):
        return "SZSE", "CHINEXT"
    if symbol.startswith(("000", "001", "002", "003")):
        return "SZSE", "SZSE_MAIN"
    if market == "bj" and symbol.startswith(("4", "8", "92")):
        return "BSE", "BSE"
    return "UNKNOWN", "OTHER_OR_UNCLASSIFIED"


def discover_root() -> tuple[Path | None, Path | None, Path | None]:
    """Check explicit common locations only; never recursively scan whole disks."""
    for root in (Path(r"C:\通达信"), Path(r"D:\通达信"), Path(r"E:\通达信"), Path(r"F:\通达信")):
        if not root.exists():
            continue
        exe = next((root / name for name in ("TdxW.exe", "tdxw.exe") if (root / name).is_file()), None)
        vipdoc = root / "vipdoc"
        return root, exe, vipdoc if vipdoc.is_dir() else None
    return None, None, None


def _record_is_valid(values: tuple[int, int, int, int, int, float, int, int]) -> tuple[bool, date | None]:
    raw_date, open_i, high_i, low_i, close_i, amount, volume, _reserved = values
    try:
        trade_date = datetime.strptime(str(raw_date), "%Y%m%d").date()
    except ValueError:
        return False, None
    price_values = (open_i, high_i, low_i, close_i)
    valid = (
        date(1989, 1, 1) <= trade_date <= date(2030, 12, 31)
        and all(value >= 0 for value in price_values)
        and low_i <= high_i
        and math.isfinite(amount)
        and amount >= 0
        and volume >= 0
    )
    return valid, trade_date


def parse_day_file(path: Path) -> dict:
    """Validate 32-byte TDX records, then parse first and final records only."""
    size = path.stat().st_size
    symbol = path.stem[2:] if len(path.stem) > 2 else path.stem
    market = path.parent.parent.name.lower()
    exchange, board = classify_symbol(symbol, market)
    basic = {"symbol": symbol, "exchange": exchange, "board": board, "file_path": str(path), "record_count": size // RECORD_SIZE}
    if size == 0:
        return {**basic, "latest_date": None, "status": "EMPTY_FILE"}
    if size % RECORD_SIZE:
        return {**basic, "latest_date": None, "status": "FORMAT_SIZE_MISMATCH"}
    with path.open("rb") as handle:
        first = RECORD.unpack(handle.read(RECORD_SIZE))
        handle.seek(-RECORD_SIZE, 2)
        last = RECORD.unpack(handle.read(RECORD_SIZE))
    first_valid, _ = _record_is_valid(first)
    last_valid, last_date = _record_is_valid(last)
    if not first_valid or not last_valid:
        return {**basic, "latest_date": None, "status": "INVALID_RECORD"}
    return {**basic, "latest_date": last_date.isoformat(), "status": "PARSED"}


def _all_day_files(vipdoc: Path) -> list[Path]:
    return sorted(path for market in vipdoc.iterdir() if market.is_dir() for path in (market / "lday").glob("*.day") if (market / "lday").is_dir())


def _read_all_records(path: Path) -> dict[str, tuple[float, float, float, float, float, int]]:
    result: dict[str, tuple[float, float, float, float, float, int]] = {}
    raw = path.read_bytes()
    for offset in range(0, len(raw), RECORD_SIZE):
        values = RECORD.unpack_from(raw, offset)
        valid, trade_date = _record_is_valid(values)
        if not valid or trade_date is None:
            continue
        _d, op, hi, lo, cl, amount, volume, _r = values
        result[trade_date.isoformat()] = (op / 100, hi / 100, lo / 100, cl / 100, amount, volume)
    return result


def _select_samples(records: list[dict]) -> list[dict]:
    chooser = random.Random(20261003)
    result: list[dict] = []
    for board in ("SSE_MAIN", "SZSE_MAIN", "CHINEXT", "STAR", "BSE"):
        eligible = [record for record in records if record["status"] == "PARSED" and record["board"] == board]
        result.extend(chooser.sample(eligible, min(10, len(eligible))))
    return result


def _crosscheck(samples: list[dict]) -> dict:
    rows: list[dict] = []
    price_matched = volume_matched = price_compared = volume_compared = 0
    volume_ratios: list[float] = []
    for sample in samples:
        path = Path(sample["file_path"])
        legacy = LEGACY_CSV_ROOT / f"{sample['symbol']}.csv"
        if not legacy.exists():
            rows.append({"symbol": sample["symbol"], "board": sample["board"], "sample_count": 1, "matched_count": 0, "mismatch_count": 0, "price_match_pct": None, "volume_match_pct": None, "amount_status": "NOT_COMPARED_LEGACY_MISSING", "status": "LEGACY_MISSING"})
            continue
        tdx = _read_all_records(path)
        old = pd.read_csv(legacy, usecols=lambda column: column in {"date", "open", "high", "low", "close", "volume", "amount"})
        old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.date.astype("string")
        matched = mismatched = p_comp = p_match = v_comp = v_match = 0
        for _, old_row in old.dropna(subset=["date"]).iterrows():
            values = tdx.get(str(old_row["date"]))
            if values is None:
                continue
            matched += 1
            tdx_prices = values[:4]
            old_prices = tuple(float(old_row[column]) for column in ("open", "high", "low", "close"))
            p_comp += 1
            if all(abs(left - right) <= max(0.01, abs(right) * 1e-6) for left, right in zip(tdx_prices, old_prices)):
                p_match += 1
            else:
                mismatched += 1
            if "volume" in old.columns and pd.notna(old_row.get("volume")):
                v_comp += 1
                if float(old_row["volume"]) > 0 and values[5] > 0:
                    volume_ratios.append(values[5] / float(old_row["volume"]))
                if abs(values[5] - float(old_row["volume"])) <= max(1.0, abs(float(old_row["volume"])) * 1e-6):
                    v_match += 1
        price_matched += p_match; price_compared += p_comp; volume_matched += v_match; volume_compared += v_comp
        rows.append({"symbol": sample["symbol"], "board": sample["board"], "sample_count": 1, "matched_count": matched, "mismatch_count": mismatched, "price_match_pct": round(100 * p_match / p_comp, 4) if p_comp else None, "volume_match_pct": round(100 * v_match / v_comp, 4) if v_comp else None, "amount_status": "NOT_COMPARED_LEGACY_HAS_NO_AMOUNT" if "amount" not in old.columns else "NOT_COMPARED_UNIT_NOT_VERIFIED", "status": "CHECKED"})
    median_ratio = float(pd.Series(volume_ratios).median()) if volume_ratios else None
    unit_note = "DATA_NOT_AVAILABLE" if median_ratio is None else "TDX_VOLUME_IS_100X_LEGACY_VOLUME" if abs(median_ratio - 100.0) < .001 else "TDX_VOLUME_IS_1X_LEGACY_VOLUME" if abs(median_ratio - 1.0) < .001 else "VOLUME_UNIT_RELATION_UNRESOLVED"
    summary = {"symbol": "__SUMMARY__", "board": "ALL", "sample_count": len(samples), "matched_count": sum(int(row["matched_count"]) for row in rows), "mismatch_count": sum(int(row["mismatch_count"]) for row in rows), "price_match_pct": round(100 * price_matched / price_compared, 4) if price_compared else None, "volume_match_pct": round(100 * volume_matched / volume_compared, 4) if volume_compared else None, "volume_unit_ratio_median": median_ratio, "volume_unit_interpretation": unit_note, "amount_status": "TDX_HAS_AMOUNT_FLOAT;_LEGACY_CSV_HAS_NO_AMOUNT", "status": "SUMMARY"}
    rows.append(summary)
    with (PROJECT_ROOT / "outputs" / "tdx_parse_crosscheck.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader(); writer.writerows(rows)
    return summary


def run() -> dict:
    docs, outputs = PROJECT_ROOT / "docs", PROJECT_ROOT / "outputs"
    root, exe, vipdoc = discover_root()
    if root is None or vipdoc is None:
        health = {"tdx_found": False, "tdx_root": "", "vipdoc_path": "", "expected_latest_trade_date": EXPECTED_DATE.isoformat(), "actual_latest_trade_date": "", "total_day_files": 0, "parsed_files": 0, "current_files": 0, "stale_files": 0, "broken_files": 0, "coverage_pct": 0.0, "price_adjustment": "RAW_OR_UNKNOWN", "status": "TDX_NOT_FOUND"}
        records: list[dict] = []
    else:
        files = _all_day_files(vipdoc)
        records = [parse_day_file(path) for path in files]
        parsed = [record for record in records if record["status"] == "PARSED"]
        latest = max((record["latest_date"] for record in parsed), default=None)
        current = sum(record["latest_date"] == EXPECTED_DATE.isoformat() for record in parsed)
        stale = sum(record["latest_date"] < EXPECTED_DATE.isoformat() for record in parsed)
        broken = len(records) - len(parsed)
        parse_pct = 100 * len(parsed) / len(records) if records else 0.0
        structure_verified = bool(parsed) and parse_pct >= 98.0
        status = "TDX_FORMAT_UNKNOWN" if not structure_verified else "TDX_STALE" if current == 0 else "TDX_INCOMPLETE" if current / len(records) < .98 else "TDX_SOURCE_CANDIDATE"
        health = {"tdx_found": True, "tdx_root": str(root), "tdx_exe_path": str(exe) if exe else "DATA_NOT_AVAILABLE", "vipdoc_path": str(vipdoc), "expected_latest_trade_date": EXPECTED_DATE.isoformat(), "actual_latest_trade_date": latest, "total_day_files": len(records), "parsed_files": len(parsed), "current_files": current, "stale_files": stale, "broken_files": broken, "coverage_pct": round(100 * current / len(records), 4) if records else 0.0, "parse_pct": round(parse_pct, 4), "price_adjustment": "RAW_OR_UNKNOWN", "status": status}
    with (outputs / "tdx_universe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "exchange", "board", "file_path", "latest_date", "record_count", "status"])
        writer.writeheader(); writer.writerows(records)
    samples = _select_samples(records)
    (outputs / "tdx_sample_parse.json").write_text(json.dumps({"sample_count": len(samples), "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    cross = _crosscheck(samples) if samples else {"sample_count": 0, "matched_count": 0, "mismatch_count": 0, "price_match_pct": None, "volume_match_pct": None}
    (outputs / "tdx_health.json").write_text(json.dumps({**health, "sample_parse": {"sample_count": len(samples), "by_board": dict(Counter(sample["board"] for sample in samples))}, "crosscheck": cross}, ensure_ascii=False, indent=2), encoding="utf-8")
    markets = {market.name: len(list((market / "lday").glob("*.day"))) for market in vipdoc.iterdir() if market.is_dir() and (market / "lday").is_dir()} if vipdoc else {}
    latest_file_time = max((path.stat().st_mtime for path in _all_day_files(vipdoc)), default=None) if vipdoc else None
    discovery = ["# TDX Local Data Discovery", "", f"- tdx_exe_path: `{health.get('tdx_exe_path', 'DATA_NOT_AVAILABLE')}`", f"- tdx_root: `{health['tdx_root'] or 'DATA_NOT_AVAILABLE'}`", f"- vipdoc_path: `{health['vipdoc_path'] or 'DATA_NOT_AVAILABLE'}`", f"- detected_markets: `{json.dumps(markets, ensure_ascii=False)}`", f"- file_counts: `{health['total_day_files']}`", f"- latest_file_time: `{datetime.fromtimestamp(latest_file_time).isoformat(timespec='seconds') if latest_file_time else 'DATA_NOT_AVAILABLE'}`", "", "## Format verification", "", f"- Verified record length: `{RECORD_SIZE}` bytes; all parsed files must be exact multiples of this length.", "- Field layout was corroborated from existing read-only TDX import/audit code and then checked against real files for valid calendar dates, nonnegative OHLC/amount/volume and low <= high.", f"- Result: parsed `{health['parsed_files']}/{health['total_day_files']}`; status `{health['status']}`.", "- price_adjustment: `RAW_OR_UNKNOWN`; this phase does not infer QFQ/HFQ."]
    (docs / "tdx_discovery.md").write_text("\n".join(discovery) + "\n", encoding="utf-8")
    return health


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
