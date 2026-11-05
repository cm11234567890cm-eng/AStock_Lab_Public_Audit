"""Permanent predictive-universe admission. RAW/QA/adjustment data stay intact.

Call this before feature construction, fit, scan or ranking. Output boundaries
must assert admission again, so cached historical artifacts cannot bypass it.
This policy never grants production/trading permission.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

EXCLUDED_EXCHANGES = frozenset({"BSE"})
PREDICTIVE_EXCHANGES = frozenset({"SSE", "SZSE"})
POLICY_VERSION = "X1_SSE_SZSE_ONLY"
BSE_REASON = "BSE_PERMANENT_EXCLUSION"
IDENTITY_REASON = "PREDICTIVE_IDENTITY_UNVERIFIED"


def exclusion_reason(exchange, symbol=None):
    value = str(exchange).strip().upper()
    value = {"BJ": "BSE", "BEIJING STOCK EXCHANGE": "BSE", "北交所": "BSE",
             "SH": "SSE", "SZ": "SZSE"}.get(value, value)
    # Secondary identity consistency guard, not authority to admit a stock.
    # Reject a mislabeled Beijing code as well as a correctly labeled exchange.
    code = str(symbol).strip().upper()
    bare = code.split('.')[0]
    beijing_code = (code.endswith(".BJ") or code.startswith("BJ") or
                    (len(bare) == 6 and bare.isdigit() and bare.startswith(("4", "8", "92"))))
    if value in EXCLUDED_EXCHANGES or beijing_code:
        return BSE_REASON
    if value not in PREDICTIVE_EXCHANGES:
        return IDENTITY_REASON
    if len(bare) != 6 or not bare.isdigit():
        return IDENTITY_REASON
    suffix = '.SH' if value == 'SSE' else '.SZ'
    if code not in (bare, bare + suffix):
        return IDENTITY_REASON
    prefixes = ('6',) if value == 'SSE' else ('0', '3')
    return "" if bare.startswith(prefixes) else IDENTITY_REASON


def reasons(frame):
    if "exchange" not in frame:
        raise ValueError(IDENTITY_REASON + ": exchange column required")
    symbols = frame["symbol"] if "symbol" in frame else [None] * len(frame)
    return pd.Series([exclusion_reason(e, s) for e, s in zip(frame.exchange, symbols)],
                     index=frame.index, dtype="object")


def predictive_mask(frame):
    return reasons(frame).eq("")


def annotate_universe(frame):
    """Audit all inputs; denied rows are explicitly ineligible, never candidates."""
    out = frame.copy()
    why = reasons(out)
    out["exclusion_reason"] = why.where(why.ne(""), out.get("exclusion_reason", ""))
    out["eligible"] = why.eq("") & out.get("eligible", True)
    return out


def predictive_universe(frame):
    """Earliest admission for research, production, training and strategy scans."""
    return frame.loc[predictive_mask(frame)].copy()


class UniverseViolation(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(audit["status"] + ": bse_candidate_count=" + str(audit["bse_candidate_count"]))


def report_audit(frame):
    """Fail closed rather than silently filtering a contaminated final report."""
    why = reasons(frame)
    audit = dict(universe_policy=POLICY_VERSION, bse_candidate_count=int(why.eq(BSE_REASON).sum()),
                 identity_unverified_count=int(why.eq(IDENTITY_REASON).sum()), status="PASS")
    if audit["bse_candidate_count"]:
        audit["status"] = "BSE_EXCLUSION_VIOLATION"
    elif audit["identity_unverified_count"]:
        audit["status"] = IDENTITY_REASON
    if audit["status"] != "PASS":
        raise UniverseViolation(audit)
    return audit


def topk(ranked, k, rank_column="rank"):
    """No re-ranking of poisoned cached results: require a clean ranking first."""
    audit = report_audit(ranked)
    selected = ranked.loc[ranked[rank_column].le(k)].copy()
    selected["bse_candidate_count"] = audit["bse_candidate_count"]
    return selected


def final_recommendations(candidates):
    """Final defensive boundary only; does not grant execution permission."""
    audit = report_audit(candidates)
    out = candidates.copy()
    out["bse_candidate_count"] = audit["bse_candidate_count"]
    return out


def require_training_identity(matrix, metadata):
    if metadata is None or len(matrix) != len(metadata) or not matrix.index.equals(metadata.index):
        raise ValueError("TRAINING_IDENTITY_REQUIRED_AND_ALIGNED")
    return report_audit(metadata)


def guard_cached_artifact(path):
    """Check identity before a legacy runner can consume cached predictions."""
    path = Path(path)
    if path.suffix == ".parquet":
        identity = pd.read_parquet(path, columns=["exchange", "symbol"])
    else:
        identity = pd.read_csv(path, usecols=["exchange", "symbol"], dtype={"symbol": str})
    return report_audit(identity)


def write_predictive_report(path, text, candidates):
    """New reports carry the measured invariant; historical reports stay frozen."""
    audit = report_audit(candidates)
    path = Path(path)
    if path.exists() and path.name.startswith(("phase25", "phase26")):
        raise ValueError("FROZEN_HISTORICAL_REPORT_USE_NEW_STAGE")
    path.write_text(f"bse_candidate_count = {audit['bse_candidate_count']}\n\n" + text, encoding="utf-8")
