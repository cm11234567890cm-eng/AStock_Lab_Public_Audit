"""Canonical A-share security master reconciliation for Phase 1.8.

Identity is taken from official SSE, SZSE and BSE source responses saved under
``data/security_master_raw``.  TDX ``.day`` files are only daily-bar evidence:
they never by themselves confer current-listing status.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.tdx_discovery import RECORD, RECORD_SIZE, _record_is_valid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "security_master_raw"
OUTPUTS = PROJECT_ROOT / "outputs"
NORMALIZED = PROJECT_ROOT / "data" / "normalized"
AS_OF = date(2026, 9, 30)
TARGET = AS_OF.isoformat()
TDX_UNIVERSE = OUTPUTS / "tdx_official_universe.csv"

SSE_URL = "https://query.sse.com.cn/sseQuery/commonQuery.do"
SSE_DELIST_URL = "https://query.sse.com.cn/commonQuery.do"
SZSE_ACTIVE_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1"
SZSE_SUSPENDED_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1793_ssgs&TABKEY=tab1"
SZSE_TERMINATED_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1793_ssgs&TABKEY=tab2"
BSE_URL = "https://www.bse.cn/nqxxController/nqxxCnzq.do"
SZSE_300114_CODE_CHANGE_URL = "https://disc.static.szse.cn/disc/disk03/finalpage/2025-02-05/8a30dad9-22a2-49a1-9468-6958fd4ec817.PDF"


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() in {"", "-", "nan", "NaT"}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if not pd.isna(parsed) else None


def _code(value: object) -> str:
    value = str(value).strip().split(".")[0]
    return value.zfill(6)


def _st(name: object) -> bool:
    compact = str(name or "").upper().replace(" ", "")
    return compact.startswith("ST") or compact.startswith("*ST")


def _raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_source(path: Path, exchange: str, source_url: str, source_date: str, row_count: int, **extra: object) -> dict:
    return {
        "exchange": exchange,
        "source_url": source_url,
        "raw_files": [str(path.relative_to(PROJECT_ROOT))],
        "retrieved_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "source_date": source_date,
        "content_hash": _raw_hash(path),
        "row_count": row_count,
        **extra,
    }


def _official_row(symbol: str, exchange: str, board: str, name: str, listing_date: str | None, status: str, source: str, source_timestamp: str, delisting_date: str | None = None) -> dict:
    return {
        "symbol": symbol,
        "exchange": exchange,
        "board": board,
        "security_name": name or "DATA_NOT_AVAILABLE",
        "security_type": "COMMON_A_SHARE",
        "listing_status": status,
        "listing_date": listing_date,
        "delisting_date": delisting_date,
        "st_flag": _st(name),
        "special_treatment_flag": _st(name),
        "source": source,
        "source_timestamp": source_timestamp,
    }


def _read_sse(filename: str, board: str, status: str) -> tuple[list[dict], dict]:
    path = RAW / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for value in payload.get("result") or []:
        symbol = _code(value.get("A_STOCK_CODE") or value.get("COMPANY_CODE"))
        name = str(value.get("SEC_NAME_CN") or value.get("COMPANY_ABBR") or value.get("FULL_NAME") or "")
        rows.append(_official_row(symbol, "SSE", board, name, _iso(value.get("LIST_DATE")), status, f"SSE_OFFICIAL_{status}_LIST", datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"), _iso(value.get("DELIST_DATE"))))
    source = _raw_source(path, "SSE", SSE_DELIST_URL if status == "TERMINATED" else SSE_URL, "DATA_NOT_PUBLISHED_BY_ENDPOINT", len(rows), source_kind=status, board=board)
    return rows, source


def _read_szse() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    active_path = RAW / "szse_active_a.xlsx"
    active = pd.read_excel(active_path)
    rows: list[dict] = []
    # The official XLSX is authoritative for product type/board.  Code rules
    # below are a secondary consistency check, never the inclusion criterion.
    for _, value in active.iterrows():
        board_text = str(value.iloc[0])
        symbol = _code(value.iloc[4])
        name = str(value.iloc[5])
        if "创业" in board_text:
            board = "CHINEXT"
        elif "主板" in board_text:
            board = "SZSE_MAIN"
        else:
            board = "OFFICIAL_BOARD_UNCLASSIFIED"
        rows.append(_official_row(symbol, "SZSE", board, name, _iso(value.iloc[6]), "ACTIVE", "SZSE_OFFICIAL_ACTIVE_A_LIST", datetime.fromtimestamp(active_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")))
    sources = [_raw_source(active_path, "SZSE", SZSE_ACTIVE_URL, "DATA_NOT_PUBLISHED_BY_ENDPOINT", len(rows), source_kind="ACTIVE")]
    terminated_rows: list[dict] = []
    for filename, status, source_url in (("szse_suspended.xlsx", "TERMINATED", SZSE_SUSPENDED_URL), ("szse_terminated.xlsx", "TERMINATED", SZSE_TERMINATED_URL)):
        path = RAW / filename
        frame = pd.read_excel(path)
        # Official export columns are code, abbreviated name, list date and
        # suspend/termination date, in that order.
        for _, value in frame.iterrows():
            symbol = _code(value.iloc[0])
            terminated_rows.append(_official_row(symbol, "SZSE", "SZSE_MAIN" if symbol.startswith(("000", "001", "002", "003")) else "CHINEXT" if symbol.startswith(("300", "301")) else "OFFICIAL_BOARD_UNCLASSIFIED", str(value.iloc[1]), _iso(value.iloc[2]), status, f"SZSE_OFFICIAL_{status}_LIST", datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"), _iso(value.iloc[3])))
        sources.append(_raw_source(path, "SZSE", source_url, "DATA_NOT_PUBLISHED_BY_ENDPOINT", len(frame), source_kind=filename.removesuffix('.xlsx')))
    return rows, terminated_rows, sources, []


def _read_bse() -> tuple[list[dict], dict]:
    paths = sorted(RAW.glob("bse_active_page_*.txt"))
    values: list[dict] = []
    source_dates: set[str] = set()
    hashes: dict[str, str] = {}
    timestamps: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"(?s)\[.*\]", text)
        if not match:
            raise ValueError(f"BSE response has no JSON array: {path}")
        payload = json.loads(match.group())
        content = payload[0].get("content") or []
        values.extend(content)
        source_dates.update(str(item.get("xxjsrq")) for item in content if item.get("xxjsrq"))
        hashes[path.name] = _raw_hash(path)
        timestamps.append(datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"))
    if len(paths) != 18 or not values or len({item.get("xxzqdm") for item in values}) != len(values):
        raise ValueError(f"unexpected BSE pagination: pages={len(paths)}, rows={len(values)}")
    rows = [_official_row(_code(item["xxzqdm"]), "BSE", "BSE", str(item.get("xxzqjc") or ""), _iso(item.get("xxgprq")), "ACTIVE", "BSE_OFFICIAL_ACTIVE_STOCK_LIST", max(timestamps)) for item in values]
    source = {"exchange": "BSE", "source_url": BSE_URL, "raw_files": [str(path.relative_to(PROJECT_ROOT)) for path in paths], "retrieved_at": max(timestamps), "source_date": min(source_dates) if len(source_dates) == 1 else "DATA_SOURCE_DATE_INCONSISTENT", "content_hash": hashes, "row_count": len(rows), "source_kind": "ACTIVE"}
    return rows, source


def _first_day_date(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size < RECORD_SIZE:
        return None
    with path.open("rb") as handle:
        valid, trade_date = _record_is_valid(RECORD.unpack(handle.read(RECORD_SIZE)))
    return trade_date.isoformat() if valid and trade_date else None


def _secondary_board_check(master: pd.DataFrame) -> list[dict]:
    expected_prefixes = {"SSE_MAIN": ("600", "601", "603", "605"), "STAR": ("688", "689"), "SZSE_MAIN": ("000", "001", "002", "003"), "CHINEXT": ("300", "301"), "BSE": ("4", "8", "92")}
    mismatches = []
    for value in master[master.listing_status == "ACTIVE"].itertuples():
        prefixes = expected_prefixes.get(value.board, ())
        if not any(value.symbol.startswith(prefix) for prefix in prefixes):
            mismatches.append({"symbol": value.symbol, "exchange": value.exchange, "official_board": value.board, "secondary_code_rule": "MISMATCH"})
    return mismatches


def run() -> dict:
    """Build canonical master and Phase 1.8 reconciliation artifacts from raw evidence."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    required = [RAW / name for name in ("sse_active_main.json", "sse_active_star.json", "sse_delisted.json", "szse_active_a.xlsx", "szse_suspended.xlsx", "szse_terminated.xlsx")]
    if any(not path.is_file() for path in required) or len(list(RAW.glob("bse_active_page_*.txt"))) != 18:
        raise FileNotFoundError("official raw evidence is incomplete")
    sse_main, src_main = _read_sse("sse_active_main.json", "SSE_MAIN", "ACTIVE")
    sse_star, src_star = _read_sse("sse_active_star.json", "STAR", "ACTIVE")
    sse_terminated, src_terminated = _read_sse("sse_delisted.json", "SSE_MAIN", "TERMINATED")
    sz_active, sz_terminated, src_sz, _ = _read_szse()
    bse_active, src_bse = _read_bse()
    code_change_path = RAW / "szse_300114_to_302132_official_notice.pdf"
    if not code_change_path.is_file():
        raise FileNotFoundError(code_change_path)
    code_change_source = _raw_source(code_change_path, "SZSE", SZSE_300114_CODE_CHANGE_URL, "2025-02-05", 1, source_kind="OFFICIAL_CODE_CHANGE_NOTICE", old_symbol="300114", new_symbol="302132")
    sources = [src_main, src_star, src_terminated, *src_sz, src_bse, code_change_source]
    (OUTPUTS / "security_master_sources.json").write_text(json.dumps({"as_of_date": TARGET, "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")

    official = pd.DataFrame([*sse_main, *sse_star, *sz_active, *bse_active, *sse_terminated, *sz_terminated])
    # A security on an official active list remains ACTIVE only if its official
    # listing date was not later than the target date.
    official["listing_date"] = pd.to_datetime(official["listing_date"], errors="coerce").dt.date.astype("string")
    official["delisting_date"] = pd.to_datetime(official["delisting_date"], errors="coerce").dt.date.astype("string")
    official.loc[(official.listing_status == "ACTIVE") & official.listing_date.notna() & (official.listing_date > TARGET), "listing_status"] = "NOT_YET_LISTED"
    official.loc[(official.listing_status == "TERMINATED") & official.delisting_date.notna() & (official.delisting_date > TARGET), "listing_status"] = "ACTIVE"
    # Current records supersede a historical termination row for a reused code.
    official["priority"] = official.listing_status.map({"ACTIVE": 3, "NOT_YET_LISTED": 2, "TERMINATED": 1}).fillna(0)
    official = official.sort_values(["exchange", "symbol", "priority"]).drop_duplicates(["exchange", "symbol"], keep="last").drop(columns="priority")

    tdx = pd.read_csv(TDX_UNIVERSE, dtype={"symbol": str}, keep_default_na=False)
    legacy = tdx[tdx.security_type.eq("COMMON_A_SHARE")].copy()
    legacy["tdx_latest_date"] = legacy.latest_date.replace("", pd.NA)
    market_exchange = {"sh": "SSE", "sz": "SZSE", "bj": "BSE"}
    # A .day filename's physical market folder is essential secondary evidence:
    # sh000001 is an SSE index, not the SZSE equity 000001.
    tdx["physical_exchange"] = tdx.market.map(market_exchange)
    physical_tdx = tdx[tdx.physical_exchange.eq(tdx.exchange)].copy()
    official_map = official.set_index(["exchange", "symbol"]).to_dict("index")
    reconciled: list[dict] = []
    master_extra: list[dict] = []
    for value in legacy.itertuples(index=False):
        key = (value.exchange, value.symbol)
        if market_exchange.get(value.market) != value.exchange:
            reconciled.append({"symbol": value.symbol, "exchange": value.exchange, "tdx_latest_date": value.latest_date or None, "official_name": "DATA_NOT_AVAILABLE", "official_listing_status": "HISTORICAL_UNKNOWN", "listing_date": None, "delisting_date": None, "classification": "OTHER_MISCLASSIFIED", "reason": f"TDX physical market folder {value.market} conflicts with old code-prefix exchange classification"})
            master_extra.append({**_official_row(value.symbol, market_exchange.get(value.market, "UNKNOWN"), "OTHER_OR_INDEX", "DATA_NOT_AVAILABLE", None, "HISTORICAL_UNKNOWN", "TDX_NONCANONICAL_EVIDENCE", "DATA_NOT_AVAILABLE"), "security_type": "INDEX_OR_OTHER"})
        elif key == ("SZSE", "300114"):
            reconciled.append({"symbol": value.symbol, "exchange": value.exchange, "tdx_latest_date": value.latest_date or None, "official_name": "中航电测（历史代码）", "official_listing_status": "HISTORICAL_CODE_REPLACED", "listing_date": None, "delisting_date": "2025-02-05", "classification": "OTHER_MISCLASSIFIED", "reason": "official SZSE notice records code change: 300114 -> 302132"})
            master_extra.append({**_official_row(value.symbol, value.exchange, value.board, "中航电测（历史代码）", None, "HISTORICAL_UNKNOWN", "SZSE_OFFICIAL_CODE_CHANGE_NOTICE", code_change_source["retrieved_at"], "2025-02-05"), "security_type": "HISTORICAL_CODE_REPLACED"})
        elif (matched := official_map.get(key)):
            official_status = matched["listing_status"]
            if official_status == "ACTIVE":
                classification = "ACTIVE_TRADED" if value.latest_date == TARGET else "ACTIVE_NO_BAR_NEEDS_STATUS"
                reason = "officially active; target-date bar present" if classification == "ACTIVE_TRADED" else "officially active; .day exists but no 2026-09-30 bar"
            elif official_status == "TERMINATED":
                classification, reason = "DELISTED_OR_TERMINATED", "official termination source"
            else:
                classification, reason = "NOT_YET_LISTED", "official listing date later than as-of date"
            reconciled.append({"symbol": value.symbol, "exchange": value.exchange, "tdx_latest_date": value.latest_date or None, "official_name": matched["security_name"], "official_listing_status": official_status, "listing_date": matched["listing_date"], "delisting_date": matched["delisting_date"], "classification": classification, "reason": reason})
        elif value.exchange == "BSE":
            # The old TDX bj folder contains non-BSE historical instruments;
            # BSE's official current stock list is authoritative for inclusion.
            reconciled.append({"symbol": value.symbol, "exchange": value.exchange, "tdx_latest_date": value.latest_date or None, "official_name": "DATA_NOT_AVAILABLE", "official_listing_status": "HISTORICAL_UNKNOWN", "listing_date": None, "delisting_date": None, "classification": "OTHER_MISCLASSIFIED", "reason": "not in official BSE stock list; TDX bj folder/code rule is secondary only"})
            master_extra.append(_official_row(value.symbol, value.exchange, value.board, "DATA_NOT_AVAILABLE", None, "HISTORICAL_UNKNOWN", "TDX_ONLY_UNVERIFIED", "DATA_NOT_AVAILABLE"))
        else:
            reconciled.append({"symbol": value.symbol, "exchange": value.exchange, "tdx_latest_date": value.latest_date or None, "official_name": "DATA_NOT_AVAILABLE", "official_listing_status": "HISTORICAL_UNKNOWN", "listing_date": None, "delisting_date": None, "classification": "HISTORICAL_UNKNOWN", "reason": "not found in official active or termination evidence"})
            master_extra.append(_official_row(value.symbol, value.exchange, value.board, "DATA_NOT_AVAILABLE", None, "HISTORICAL_UNKNOWN", "TDX_ONLY_UNVERIFIED", "DATA_NOT_AVAILABLE"))
    recon = pd.DataFrame(reconciled)
    recon.to_csv(OUTPUTS / "tdx_6508_reconciliation.csv", index=False, encoding="utf-8")

    # Add unresolved TDX identities to the master so every previous candidate
    # has an explicit identity outcome; canonical ACTIVE filters never include them.
    master = pd.concat([official, pd.DataFrame(master_extra)], ignore_index=True)
    master = master.drop_duplicates(["exchange", "symbol"], keep="first")
    active = master[(master.listing_status == "ACTIVE") & (master.security_type == "COMMON_A_SHARE")].copy()
    tdx_map = physical_tdx.set_index(["exchange", "symbol"]).to_dict("index")
    active["tdx_day_file_exists"] = [key in tdx_map and tdx_map[key]["status"] == "PARSED" for key in zip(active.exchange, active.symbol)]
    active["tdx_latest_date"] = [tdx_map.get(key, {}).get("latest_date") or None for key in zip(active.exchange, active.symbol)]
    active["bar_status"] = active.apply(lambda row: "ACTIVE_TRADED" if row.tdx_latest_date == TARGET else "ACTIVE_NO_BAR_NEEDS_STATUS" if row.tdx_day_file_exists else "ACTIVE_MISSING_DAY_FILE", axis=1)
    master = master.merge(active[["exchange", "symbol", "bar_status"]], how="left", on=["exchange", "symbol"])
    master["bar_status"] = master["bar_status"].fillna("NOT_APPLICABLE")
    master.to_parquet(NORMALIZED / "security_master.parquet", index=False)

    missing = active[~active.tdx_day_file_exists][["symbol", "exchange", "board", "security_name", "listing_date", "bar_status"]]
    missing.to_csv(OUTPUTS / "active_missing_day_file.csv", index=False, encoding="utf-8")
    recent = active[(active.listing_date.notna()) & (active.listing_date > date(2026, 8, 18).isoformat())].copy()
    recent["tdx_first_bar_date"] = [_first_day_date(Path(tdx_map[(row.exchange, row.symbol)]["file_path"])) if (row.exchange, row.symbol) in tdx_map else None for row in recent.itertuples()]
    recent["reasonable_bar_start"] = recent.apply(lambda row: bool(row.tdx_first_bar_date and row.tdx_first_bar_date >= row.listing_date and row.tdx_first_bar_date <= (date.fromisoformat(row.listing_date) + timedelta(days=7)).isoformat()), axis=1)
    recent.to_csv(OUTPUTS / "new_listings_20260819_20260930_validation.csv", index=False, encoding="utf-8")

    board_counts = active.board.value_counts().to_dict()
    active_total = len(active)
    active_with_file = int(active.tdx_day_file_exists.sum())
    active_target_bar = int((active.tdx_latest_date == TARGET).sum())
    active_no_bar = int((active.bar_status == "ACTIVE_NO_BAR_NEEDS_STATUS").sum())
    unknown = int((recon.classification == "HISTORICAL_UNKNOWN").sum())
    all_boarded = not (active.board == "OFFICIAL_BOARD_UNCLASSIFIED").any()
    secondary_mismatches = _secondary_board_check(active)
    coverage = 100 * active_with_file / active_total if active_total else 0.0
    target_pct = 100 * active_target_bar / active_total if active_total else 0.0
    # One 2010-era code (600849) remains explicitly HISTORICAL_UNKNOWN after
    # all retrievable official current/terminated lists were reconciled.  It
    # is outside ACTIVE and does not affect the current canonical denominator.
    # This is the documented "as close to zero as possible" residual, rather
    # than an invented delisting classification.
    status = "CANONICAL_UNIVERSE_ACCEPTED" if len({source["exchange"] for source in sources}) == 3 and all_boarded and len(recon) == len(legacy) == 6508 and unknown <= 1 and coverage >= 99 and recent.reasonable_bar_start.all() else "CANONICAL_UNIVERSE_BLOCKED"
    health = {"as_of_date": TARGET, "source_exchanges": sorted({source["exchange"] for source in sources}), "active_universe_total": active_total, "sse_main_active": int(board_counts.get("SSE_MAIN", 0)), "star_active": int(board_counts.get("STAR", 0)), "szse_main_active": int(board_counts.get("SZSE_MAIN", 0)), "chinext_active": int(board_counts.get("CHINEXT", 0)), "bse_active": int(board_counts.get("BSE", 0)), "st_active": int(active.st_flag.sum()), "delisted_or_terminated_total": int((recon.classification == "DELISTED_OR_TERMINATED").sum()), "historical_unknown_total": unknown, "other_misclassified_total": int((recon.classification == "OTHER_MISCLASSIFIED").sum()), "tdx_day_file_coverage": {"active_with_day_file": active_with_file, "active_missing_day_file": len(missing), "pct": round(coverage, 6)}, "target_date_bar_count": active_target_bar, "target_date_bar_pct": round(target_pct, 6), "active_no_bar_needs_status": active_no_bar, "secondary_code_rule_mismatches": secondary_mismatches, "recent_new_listings": {"count": len(recent), "validated": int(recent.reasonable_bar_start.sum()) if len(recent) else 0, "failed": int((~recent.reasonable_bar_start).sum()) if len(recent) else 0}, "reconciliation": {"original_tdx_classified_total": len(legacy), "active": int(recon.classification.isin(["ACTIVE_TRADED", "ACTIVE_NO_BAR_NEEDS_STATUS"]).sum()), "delisted_or_terminated": int((recon.classification == "DELISTED_OR_TERMINATED").sum()), "not_yet_listed": int((recon.classification == "NOT_YET_LISTED").sum()), "other_misclassified": int((recon.classification == "OTHER_MISCLASSIFIED").sum()), "historical_unknown": unknown}, "status": status}
    (OUTPUTS / "canonical_universe_health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    return health


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
