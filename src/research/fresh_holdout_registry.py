"""Fresh data qualification only. No features, returns, targets or model imports."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data.trading_calendar import is_trading_day
from src.data.tdx_discovery import classify_symbol
from src.universe_policy import exclusion_reason, BSE_REASON

START = "2026-10-15"
KEY = ["exchange", "symbol", "date"]
RAW_COLUMNS = KEY + ["open", "high", "low", "close", "volume", "source_member"]
EVENT_TYPES = {1: "EX_RIGHTS_OR_DIVIDEND", 11: "CAPITAL_SCALING", 15: "READJUSTMENT"}
REGISTRY_COLUMNS = [
    "date", "raw_ready", "identity_ready", "ca_event_ready", "t5_complete",
    "eligible_stock_days", "coverage", "status",
]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(contract):
    assert contract["fresh_holdout_start"] == START, "FROZEN_START_CHANGED"
    assert contract["minimum_ready_days"] == 10, "FROZEN_MINIMUM_CHANGED"
    assert contract["minimum_daily_coverage"] == 0.70, "FROZEN_COVERAGE_FLOOR_CHANGED"
    assert contract["preparation_only"]
    assert not any(contract[k] for k in (
        "train_models", "evaluate_rankings", "amount_enabled", "production_changes"
    ))


def sessions(start, end):
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    assert start.year == 2026 and end.year == 2026, "CALENDAR_REVALIDATION_REQUIRED"
    return [x.date().isoformat() for x in pd.date_range(start, end)
            if is_trading_day(x.date())]


@lru_cache(maxsize=260)
def next_five(day):
    d = date.fromisoformat(str(day))
    assert d.year == 2026, "CALENDAR_REVALIDATION_REQUIRED"
    if d == date(2026, 12, 31):
        return []
    return sessions(d + timedelta(days=1), min(d + timedelta(days=20), date(2026, 12, 31)))[:5]


def namespace_valid(exchange, symbol, member):
    match = re.fullmatch(r"(sh|sz|bj)/lday/(sh|sz|bj)(\d{6})\.day", str(member).lower())
    if not match or match[1] != match[2] or match[3] != str(symbol):
        return False
    if exchange != {"sh": "SSE", "sz": "SZSE", "bj": "BSE"}[match[1]]:
        return False
    return classify_symbol(str(symbol), match[1])[0] == exchange


def normalize_raw(frame):
    frame = frame[RAW_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame["symbol"] = frame.symbol.astype(str).str.zfill(6)
    frame["source_member"] = frame.source_member.str.replace("\\", "/", regex=False)
    return frame.sort_values(KEY).reset_index(drop=True)


def raw_validity(frame):
    price = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    good = np.isfinite(price).all(axis=1) & price.gt(0).all(axis=1)
    good &= price.high.ge(price[["open", "close", "low"]].max(axis=1))
    good &= price.low.le(price[["open", "close", "high"]].min(axis=1))
    good &= frame.apply(lambda r: namespace_valid(r.exchange, r.symbol, r.source_member), axis=1)
    good &= ~frame.duplicated(KEY, keep=False)
    good &= frame.date.map(lambda d: is_trading_day(pd.Timestamp(d).date()))
    return good


def append_raw(frame, directory, completed_cutoff, provenance):
    """Immutable daily chunks. Revisions fail before any new chunk is written."""
    frame = normalize_raw(frame)
    assert len(frame) and frame.date.min() >= START
    assert frame.date.max() <= str(completed_cutoff), "UNFINISHED_SESSION"
    assert raw_validity(frame).all(), "INVALID_RAW"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.parquet"))
    latest = max((p.stem for p in existing), default="")
    pending, receipts = [], []
    for day, rows in frame.groupby("date", sort=True):
        rows = rows.reset_index(drop=True)
        path = directory / f"{day}.parquet"
        manifest = path.with_suffix(".json")
        if path.exists():
            assert manifest.exists(), "MISSING_RAW_MANIFEST"
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            assert sha256(path) == prior["sha256"], "RAW_HASH_CHANGED"
            pd.testing.assert_frame_equal(normalize_raw(pd.read_parquet(path)), rows, check_dtype=False)
            receipts.append(dict(date=day, action="REUSED_UNCHANGED", sha256=prior["sha256"]))
        else:
            assert day > latest, "HISTORICAL_DATE_INSERTION_REJECTED"
            assert not manifest.exists(), "ORPHAN_MANIFEST"
            pending.append((day, rows, path, manifest))
    # An absent market session is never inferred as a holiday.
    all_days = {p.stem for p in existing} | set(frame.date)
    assert set(sessions(START, max(all_days))).issubset(all_days), "MARKET_DATE_GAP"
    for day, rows, path, manifest in pending:
        with path.open("xb") as stream:
            rows.to_parquet(stream, index=False)
        record = dict(date=day, rows=len(rows), sha256=sha256(path), **provenance)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        receipts.append(dict(date=day, action="APPENDED", sha256=record["sha256"]))
    return receipts


def presence_schema(native):
    if not isinstance(native, dict) or str(native.get("ErrorId")) != "0" or native.get("Error"):
        return False
    # Explicit successful empty response used by the validated SDK1.2.2.
    if native.get("Value") == [] and "Date" not in native and "Type" not in native:
        return True
    dates, types = native.get("Date"), native.get("Type")
    if not isinstance(dates, list) or not isinstance(types, list) or len(dates) != len(types):
        return False
    # Economic values are not qualified, but contradictory array lengths cannot
    # certify event absence or a complete presence catalogue.
    if "Value" in native and (
        not isinstance(native["Value"], list) or len(native["Value"]) != len(dates)
    ):
        return False
    parsed = pd.to_datetime([str(d) for d in dates], format="%Y%m%d", errors="coerce")
    try:
        return not parsed.isna().any() and all(
            not isinstance(t, bool) and str(t).strip() in {"1", "11", "15"} for t in types)
    except (ValueError, TypeError, OverflowError):
        return False


def presence_rows(wrapper, source_path, source_hash):
    """Ignore native historical rows outside the explicitly requested interval."""
    start, end = wrapper["requested_start"], wrapper["requested_end"]
    assert START <= start <= end
    native = wrapper.get("native")
    qualified = presence_schema(native)
    cert = dict(exchange=wrapper["exchange"], symbol=wrapper["symbol"],
                coverage_start=start, coverage_end=end, qualified=qualified,
                source_path=source_path, source_sha256=source_hash,
                reason="EXPLICIT_SUCCESS" if qualified else "SOURCE_OR_SCHEMA_UNRESOLVED")
    events = []
    if qualified:
        for d, kind in zip(native.get("Date", []), native.get("Type", [])):
            ex_date = pd.to_datetime(str(d), format="%Y%m%d").date().isoformat()
            if start <= ex_date <= end:
                events.append(dict(exchange=wrapper["exchange"], symbol=wrapper["symbol"],
                    ex_date=ex_date, event_exists=True, event_type=EVENT_TYPES[int(kind)],
                    event_presence_qualified=True, parameter_basis_qualified=False,
                    purpose="LABEL_CLEANING_ONLY", source_path=source_path, source_sha256=source_hash))
    return cert, events


def covered(certificates, exchange, symbol, start, end):
    intervals = certificates.get((exchange, symbol), [])
    remaining = date.fromisoformat(start)
    for left, right in sorted(intervals):
        left, right = date.fromisoformat(left), date.fromisoformat(right)
        if right < remaining:
            continue
        if left > remaining:
            return False
        remaining = right + timedelta(days=1)
        if remaining > date.fromisoformat(end):
            return True
    return False


def build_registry(raw, identities, certificates, events, contract):
    """Only inspect bar existence/sanity and CA presence, never future returns."""
    validate_contract(contract)
    raw = normalize_raw(raw)
    assert len(raw) and raw.date.min() >= START
    raw["raw_valid"] = raw_validity(raw)
    assert not identities.duplicated(["exchange", "symbol"]).any()
    identity = identities.set_index(["exchange", "symbol"]).to_dict("index")
    certs = {}
    for r in certificates.to_dict("records"):
        if r["qualified"]:
            certs.setdefault((r["exchange"], r["symbol"]), []).append((r["coverage_start"], r["coverage_end"]))
    event_days = {}
    for r in events.to_dict("records"):
        if r["event_exists"] and r["event_presence_qualified"]:
            event_days.setdefault((r["exchange"], r["symbol"]), set()).add(r["ex_date"])
    bars = set(raw.loc[raw.raw_valid, KEY].itertuples(index=False, name=None))
    observed = set(raw.loc[raw.raw_valid, "date"])
    latest = raw.date.max()
    rows = []
    for r in raw.itertuples(index=False):
        ident = identity.get((r.exchange, r.symbol), {})
        status = ident.get("identity_status", "IDENTITY_PENDING")
        exchange_reason = exclusion_reason(r.exchange, r.symbol)
        common = status == "VERIFIED_COMMON_A_SHARE"
        denominator = status != "VERIFIED_NON_COMMON" and not exchange_reason
        listing = ident.get("eligible_first_date") or ident.get("listing_date")
        delisting = ident.get("delisting_date") or ""
        listing_ok = bool(common and listing and r.date >= listing and (not delisting or r.date <= delisting))
        st = bool(ident.get("current_st", False))
        preeligible = bool(r.raw_valid and common and listing_ok and not st and not exchange_reason)
        future = next_five(r.date)
        calendar_ready = len(future) == 5
        market_complete = calendar_ready and all(d in observed for d in future)
        complete = market_complete and all((r.exchange, r.symbol, d) in bars for d in future)
        observed_end = min(future[-1], latest) if future else latest
        ca_observed = covered(certs, r.exchange, r.symbol, r.date, observed_end)
        ca_full = complete and covered(certs, r.exchange, r.symbol, r.date, future[-1])
        # Pending/unknown windows are not reported as event-free. Weekend-dated
        # presence also conservatively censors the interval; never silently drop it.
        ca_cross = (any(r.date < d <= future[-1]
                        for d in event_days.get((r.exchange, r.symbol), set()))
                    if ca_full else None)
        eligible = preeligible and complete and ca_full and not ca_cross
        reasons = []
        if exchange_reason: reasons.append(exchange_reason)
        if not r.raw_valid: reasons.append("RAW_INVALID")
        if status == "IDENTITY_PENDING": reasons.append("IDENTITY_PENDING")
        if status == "VERIFIED_NON_COMMON": reasons.append("NON_COMMON_SECURITY")
        if common and not listing_ok: reasons.append("LISTING_BOUNDARY")
        if st: reasons.append("FROZEN_ST_RULE")
        if not calendar_ready: reasons.append("CALENDAR_PENDING")
        if not complete: reasons.append("T5_PENDING" if not market_complete else "MISSING_FUTURE_BAR")
        if preeligible and not ca_observed: reasons.append("OBSERVED_CA_SOURCE_PENDING")
        if complete and not ca_full: reasons.append("CA_WINDOW_SOURCE_PENDING")
        if ca_cross: reasons.append("LABEL_CROSSES_CA")
        rows.append(dict(date=r.date, exchange=r.exchange, symbol=r.symbol,
            raw_valid=bool(r.raw_valid), denominator=denominator, identity_status=status,
            listing_valid=listing_ok, frozen_st=st, prelabel_eligible=preeligible,
            t5_date=future[-1] if calendar_ready else None,
            calendar_ready=calendar_ready, market_t5_complete=market_complete,
            label_complete=complete, label_pending=not complete,
            ca_observed_ready=ca_observed, ca_window_ready=ca_full, label_crosses_ca=ca_cross,
            eligible=bool(eligible), exclusion_reason=reasons[0] if reasons else "",
            exclusion_reasons="|".join(reasons)))
    ledger = pd.DataFrame(rows)
    registry, audit = [], []
    for day in sessions(START, latest):
        part = ledger[ledger.date.eq(day)]
        denom = int(part.denominator.sum())
        count = int(part.eligible.sum())
        coverage = count / denom if denom else 0.0
        raw_ready = bool(len(part) and part.raw_valid.any())
        identity_ready = bool(denom and part.prelabel_eligible.sum()/denom >= contract["minimum_daily_coverage"])
        ca_ready = bool(denom and (part.prelabel_eligible & part.ca_observed_ready).sum()/denom >= contract["minimum_daily_coverage"])
        future = next_five(day)
        t5_complete = len(future) == 5 and all(d in observed for d in future)
        ready = raw_ready and identity_ready and ca_ready and t5_complete and coverage >= contract["minimum_daily_coverage"]
        registry.append(dict(date=day, raw_ready=raw_ready, identity_ready=identity_ready,
            ca_event_ready=ca_ready, t5_complete=t5_complete, eligible_stock_days=count,
            coverage=coverage, status="READY" if ready else "PENDING"))
        audit.append(dict(date=day, raw_stock_days=len(part), denominator_stock_days=denom,
            non_common_stock_days=int(part.identity_status.eq("VERIFIED_NON_COMMON").sum()),
            exchange_excluded_stock_days=int(part.exclusion_reason.eq(BSE_REASON).sum()),
            identity_pending=int(part.identity_status.eq("IDENTITY_PENDING").sum()),
            prelabel_eligible_stock_days=int(part.prelabel_eligible.sum()),
            prelabel_coverage=float(part.prelabel_eligible.sum()/denom) if denom else 0.0,
            ca_observed_pending_stock_days=int((part.prelabel_eligible & ~part.ca_observed_ready).sum()),
            t5_pending_stock_days=int((part.denominator & part.label_pending).sum()),
            missing_future_bar_stock_days=int((part.denominator & part.market_t5_complete & part.label_pending).sum())))
    return pd.DataFrame(registry, columns=REGISTRY_COLUMNS), ledger, pd.DataFrame(audit)


def preparation_status(registry):
    assert not registry.date.duplicated().any(), "DUPLICATE_REGISTRY_DATE"
    ready = int(registry.loc[registry.status.eq("READY"), "date"].nunique())
    return "FRESH_HOLDOUT_READY" if ready >= 10 else "FRESH_HOLDOUT_PREPARING"
