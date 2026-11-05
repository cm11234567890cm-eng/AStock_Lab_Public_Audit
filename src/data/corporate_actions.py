"""Corporate-action normalization with explicit provenance.

This module deliberately keeps the source event record separate from TDX daily
bars.  It reads a *copy produced inside AStock_Lab* of the read-only local TDX
``gbbq`` cache, and never opens TDX for writing or launches its runtime.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_PATH = PROJECT_ROOT / "data" / "normalized" / "corporate_actions.parquet"
DEFAULT_PARSED_GBBQ = PROJECT_ROOT / "data" / "cache" / "tdx_gbbq_parsed.csv"
DEFAULT_SECURITY_MASTER = PROJECT_ROOT / "data" / "normalized" / "security_master.parquet"
TDX_GBBQ_PATH = Path(r"D:\通达信\T0002\hq_cache\gbbq")
ARCHIVED_GBBQ_SNAPSHOT = PROJECT_ROOT / "data" / "corporate_actions_raw" / "tdx_gbbq_snapshot_20260818.bin"

CORPORATE_ACTION_COLUMNS = [
    "symbol", "exchange", "ex_date", "cash_dividend_per_10",
    "bonus_share_per_10", "transfer_share_per_10", "rights_share_per_10",
    "rights_price", "event_type", "source", "source_date", "retrieved_at",
    "raw_reference", "status",
    "reported_cash_dividend_per_10", "actual_cash_dividend_per_share",
    "ex_reference_cash_dividend_per_share", "participating_share_count",
    "total_share_count", "excluded_share_count", "differential_dividend_flag",
    "actual_bonus_transfer_ratio", "ex_reference_bonus_transfer_ratio",
    "cash_basis_status", "basis_source", "basis_raw_reference",
    "reported_rights_share_per_10", "ex_reference_rights_ratio",
    "rights_subscribed_share_count", "pre_rights_total_share_count",
]


class DividendBasisError(ValueError):
    """Official price-basis evidence is incomplete or internally inconsistent."""


def _finite_nonnegative(value: object, field: str) -> float:
    if value is None or pd.isna(value):
        raise DividendBasisError(f"{field} is unknown")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DividendBasisError(f"{field} must be finite and nonnegative")
    return result


def resolve_dividend_basis(event: dict) -> dict:
    """Return a new record, retaining reported entitlement and official evidence.

    Priority: explicit official virtual cash, official participating/total shares,
    then explicitly confirmed ordinary treatment. Never infer from market prices
    or assume that the absence of a differential flag means ordinary treatment.
    """
    result = dict(event)
    actual = _finite_nonnegative(event.get('actual_cash_dividend_per_share'), 'actual_cash_dividend_per_share')
    flag = event.get('differential_dividend_flag')
    if flag is None or pd.isna(flag) or type(flag).__name__ not in ('bool', 'bool_'):
        raise DividendBasisError('differential_dividend_flag must be explicitly boolean')
    if not event.get('basis_source') or not event.get('basis_raw_reference'):
        raise DividendBasisError('Official basis provenance is required')
    participating, total, excluded = (event.get(c) for c in ('participating_share_count','total_share_count','excluded_share_count'))
    has_counts = participating is not None and not pd.isna(participating) and total is not None and not pd.isna(total)
    ratio = None
    if has_counts:
        participating = _finite_nonnegative(participating, 'participating_share_count')
        total = _finite_nonnegative(total, 'total_share_count')
        if total == 0 or participating > total or not participating.is_integer() or not total.is_integer():
            raise DividendBasisError('Invalid official share counts')
        if excluded is not None and not pd.isna(excluded) and _finite_nonnegative(excluded, 'excluded_share_count') != total-participating:
            raise DividendBasisError('Excluded shares conflict with official participating shares')
        if not flag and participating != total:
            raise DividendBasisError('Ordinary flag conflicts with unequal share counts')
        ratio = participating/total
        result['calculated_virtual_cash_dividend_per_share'] = actual*ratio
        result['excluded_share_count'] = int(total-participating)
    official = event.get('official_virtual_cash_dividend_per_share')
    if official is not None and not pd.isna(official):
        virtual = _finite_nonnegative(official, 'official_virtual_cash_dividend_per_share')
        status = 'OFFICIAL_EXPLICIT'
    elif ratio is not None:
        virtual, status = actual*ratio, 'OFFICIAL_SHARE_COUNTS'
    elif not flag:
        virtual, status = actual, 'ORDINARY_CONFIRMED'
    else:
        raise DividendBasisError('Differential dividend has no official virtual value or share counts')
    if not flag and not math.isclose(virtual, actual, rel_tol=0, abs_tol=1e-12):
        raise DividendBasisError('Ordinary dividend cannot have a different virtual value')
    result['ex_reference_cash_dividend_per_share'] = virtual
    result['cash_basis_status'] = status
    actual_shares = event.get('actual_bonus_transfer_ratio')
    official_shares = event.get('official_virtual_bonus_transfer_ratio')
    if official_shares is not None and not pd.isna(official_shares):
        result['ex_reference_bonus_transfer_ratio'] = _finite_nonnegative(official_shares, 'official_virtual_bonus_transfer_ratio')
        if not flag and actual_shares is not None and not pd.isna(actual_shares):
            if not math.isclose(result['ex_reference_bonus_transfer_ratio'],
                                _finite_nonnegative(actual_shares,'actual_bonus_transfer_ratio'),rel_tol=0,abs_tol=1e-12):
                raise DividendBasisError('Ordinary share distribution has conflicting reference ratio')
    elif actual_shares is not None and not pd.isna(actual_shares):
        actual_shares = _finite_nonnegative(actual_shares, 'actual_bonus_transfer_ratio')
        if actual_shares == 0 or not flag:
            result['ex_reference_bonus_transfer_ratio'] = actual_shares
        elif ratio is not None:
            result['ex_reference_bonus_transfer_ratio'] = actual_shares*ratio
        else:
            raise DividendBasisError('Differential share distribution needs its own reference basis')
    return result


def actual_cash_distribution(event: dict, entitled_shares: float) -> float:
    """Gross shareholder entitlement only; not a total-return strategy engine."""
    return (_finite_nonnegative(event.get('actual_cash_dividend_per_share'), 'actual_cash_dividend_per_share')
            * _finite_nonnegative(entitled_shares, 'entitled_shares'))


def _as_number(value: object) -> float | None:
    """Preserve an absent source value as NULL; zero remains a real zero."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _event_type(cash: float | None, bonus: float | None, rights: float | None) -> str:
    parts = []
    if cash not in (None, 0.0):
        parts.append("CASH_DIVIDEND")
    if bonus not in (None, 0.0):
        parts.append("BONUS_OR_TRANSFER")
    if rights not in (None, 0.0):
        parts.append("RIGHTS_ISSUE")
    return "+".join(parts) if parts else "NO_ECONOMIC_CHANGE_RECORDED"


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_tdx_gbbq(
    parsed_path: Path = DEFAULT_PARSED_GBBQ,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    tdx_source_path: Path = TDX_GBBQ_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Normalize category-1 TDX gbbq records into the canonical schema.

    The TDX cache's category-1 layout exposes a combined ``songzhuangu`` field.
    It is retained as ``bonus_share_per_10`` with an explicit status and leaves
    ``transfer_share_per_10`` as NULL.  This avoids inventing a split which the
    local cache does not provide.  The adjustment engine understands that status
    as a known combined amount, not as an unknown numeric zero.
    """
    if not parsed_path.is_file():
        raise FileNotFoundError(f"TDX parsed gbbq cache unavailable: {parsed_path}")
    if not security_master_path.is_file():
        raise FileNotFoundError(f"Security master unavailable: {security_master_path}")

    raw = pd.read_csv(parsed_path, dtype={"code": "string"})
    master = pd.read_parquet(security_master_path, columns=["symbol", "exchange"])
    master["symbol"] = master["symbol"].astype("string").str.zfill(6)
    exchange_by_symbol = master.drop_duplicates("symbol").set_index("symbol")["exchange"].to_dict()
    source_mtime = datetime.fromtimestamp(tdx_source_path.stat().st_mtime).astimezone()
    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")

    records: list[dict] = []
    # pytdx documents category 1 as ex-rights/ex-dividend.  Other categories
    # describe listing/administrative changes and must not be double-counted as
    # an economic price transformation.
    for value in raw.loc[raw["category"] == 1].itertuples(index=False):
        symbol = str(value.code).zfill(6)
        exchange = exchange_by_symbol.get(symbol)
        if exchange not in {"SSE", "SZSE", "BSE"}:
            continue
        # TDX gbbq stores a YYYYMMDD integer, not a Unix timestamp.
        ex_date = datetime.strptime(str(int(value.datetime)), "%Y%m%d").date().isoformat()
        cash = _as_number(value.hongli_panqianliutong)
        rights_price = _as_number(value.peigujia_qianzongguben)
        bonus = _as_number(value.songgu_qianzongguben)
        rights = _as_number(value.peigu_houzongguben)
        records.append({
            "symbol": symbol,
            "exchange": exchange,
            "ex_date": ex_date,
            "cash_dividend_per_10": cash,
            "reported_cash_dividend_per_10": cash,
            "actual_cash_dividend_per_share": None if cash is None else cash / 10,
            "ex_reference_cash_dividend_per_share": None,
            "differential_dividend_flag": None,
            "cash_basis_status": "UNVERIFIED",
            "bonus_share_per_10": bonus,
            "transfer_share_per_10": None,
            "rights_share_per_10": rights,
            "rights_price": rights_price,
            "event_type": _event_type(cash, bonus, rights),
            "source": "TDX_LOCAL_GBBQ_CLIENT_CACHE",
            "source_date": source_mtime.date().isoformat(),
            "retrieved_at": retrieved_at,
            "raw_reference": f"{tdx_source_path}#record={int(value.raw_record_id)}",
            "status": "TDX_GBBQ_COMBINED_BONUS_TRANSFER",
        })
    result = pd.DataFrame(records, columns=CORPORATE_ACTION_COLUMNS)
    result["ex_date"] = pd.to_datetime(result["ex_date"], errors="coerce").dt.date
    result = result.dropna(subset=["ex_date"]).sort_values(["symbol", "ex_date", "raw_reference"], kind="stable")
    meta = {
        "source": "TDX_LOCAL_GBBQ_CLIENT_CACHE",
        "source_path": str(tdx_source_path),
        "source_sha256": _source_hash(tdx_source_path),
        "source_file_last_write": source_mtime.isoformat(timespec="seconds"),
        "parsed_cache": str(parsed_path),
        "category_1_records": int(len(result)),
        "covered_symbols": int(result["symbol"].nunique()),
        "latest_event_date": result["ex_date"].max().isoformat() if not result.empty else None,
    }
    if ARCHIVED_GBBQ_SNAPSHOT.is_file():
        meta["archived_snapshot"] = {
            "path": str(ARCHIVED_GBBQ_SNAPSHOT),
            "sha256": _source_hash(ARCHIVED_GBBQ_SNAPSHOT),
            "matches_source_sha256": _source_hash(ARCHIVED_GBBQ_SNAPSHOT) == meta["source_sha256"],
        }
    return result, meta


def get_actions(symbol: str, path: Path = DEFAULT_ACTION_PATH) -> pd.DataFrame:
    """Return all source events for one code, without applying an as-of filter."""
    if not path.is_file():
        raise FileNotFoundError(f"Corporate-action parquet unavailable: {path}")
    actions = pd.read_parquet(path)
    # Legacy records remain readable, but missing official reference fields must
    # stay NULL. The engine will fail closed instead of reusing nominal cash.
    for column in CORPORATE_ACTION_COLUMNS:
        if column not in actions:
            actions[column] = None
    actions["symbol"] = actions["symbol"].astype("string").str.zfill(6)
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="coerce").dt.date
    return actions.loc[actions["symbol"] == str(symbol).zfill(6)].sort_values("ex_date", kind="stable").reset_index(drop=True)
