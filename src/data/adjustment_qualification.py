"""Independent qualification gates; interface limits are not causality failures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.adjustment_engine import adjust_prices_asof, IncompleteCorporateAction

PRICE_TOLERANCE = 0.005  # Fixed original half-cent bound, not tuned to results.
PRICE_FIELDS = ('open', 'high', 'low', 'close')


def compare_current_adjusted(symbol, raw, official, actions, method, as_of='2026-09-30'):
    """Compare on a declared common anchor, never estimate unknown event inputs.

    The caller must establish the official series' anchor and supply all events.
    Missing evidence produces NULL metrics, not a zero mismatch count or a pass.
    """
    result = dict(symbol=symbol, as_of_date=as_of, method=method,
                  price_match_pct=None, max_abs_diff=None, mismatch_count=None,
                  compared_values=0, tolerance=PRICE_TOLERANCE,
                  validation_kind='CURRENT_ADJUSTED_VALIDATION')
    effective = actions.loc[(actions['symbol'].astype(str) == str(symbol)) &
                            (pd.to_datetime(actions.ex_date) <= pd.Timestamp(as_of))]
    missing = effective.loc[~effective['basis_verified'].fillna(False)]
    if len(missing):
        return result | dict(status='UNRESOLVED_PARAMETER', unresolved_events=len(missing))
    raw = raw.loc[pd.to_datetime(raw.date) <= pd.Timestamp(as_of)].copy(deep=True)
    original = raw.copy(deep=True)
    try:
        calculated = adjust_prices_asof(symbol, raw, as_of, method, actions)
    except IncompleteCorporateAction:
        return result | dict(status='UNRESOLVED_PARAMETER', unresolved_events=len(actions))
    pd.testing.assert_frame_equal(raw, original)
    calculated['date'] = pd.to_datetime(calculated.date)
    reference = official.copy(deep=True)
    reference['date'] = pd.to_datetime(reference.date)
    reference = reference.loc[reference.date <= pd.Timestamp(as_of)]
    if (calculated.date.duplicated().any() or reference.date.duplicated().any()
            or set(calculated.date) != set(reference.date) or not len(reference)):
        return result | dict(status='DATE_COVERAGE_MISMATCH', unresolved_events=0)
    lhs = calculated.set_index('date').sort_index()[list(PRICE_FIELDS)].to_numpy(float)
    rhs = reference.set_index('date').sort_index()[list(PRICE_FIELDS)].to_numpy(float)
    if not (np.isfinite(lhs).all() and np.isfinite(rhs).all()):
        return result | dict(status='NONFINITE_PRICE', unresolved_events=0)
    diff = np.abs(lhs-rhs)
    mismatch = int((diff > PRICE_TOLERANCE).sum())
    return result | dict(status='PASS' if mismatch == 0 else 'PRICE_MISMATCH',
                         unresolved_events=0, compared_values=int(diff.size),
                         mismatch_count=mismatch, max_abs_diff=float(diff.max()),
                         price_match_pct=float((diff <= PRICE_TOLERANCE).mean()*100))


def acceptance_status(*, raw_price_pct, inventory_complete, unresolved_events,
                      unresolved_special, rights_passed, current_adjusted_passed,
                      synthetic_passed, real_passed, differential_pit_passed,
                      rights_pit_passed, lookahead_count, raw_unchanged,
                      historical_snapshot_available=False,
                      amount_status='AMOUNT_REPRESENTATION_PENDING'):
    """Historical snapshot availability and Amount are reported, not price gates."""
    checks = dict(
        RAW_VALIDATION=raw_price_pct == 100,
        SPECIAL_ACTION_CLOSURE=inventory_complete and unresolved_special == 0,
        EVENT_PARAMETERS=unresolved_events == 0,
        RIGHTS_VALIDATION=rights_passed > 0,
        CURRENT_ADJUSTED_VALIDATION=current_adjusted_passed,
        PIT_CAUSALITY_VALIDATION=all((synthetic_passed, real_passed,
                                     differential_pit_passed, rights_pit_passed)),
        NO_LOOKAHEAD=lookahead_count == 0,
        RAW_IMMUTABLE=raw_unchanged,
    )
    accepted = all(checks.values())
    return dict(status='ADJUSTMENT_BASIS_ACCEPTED' if accepted else 'ADJUSTMENT_BASIS_BLOCKED',
                TDXAIDATA_SOURCE_QUALIFIED=accepted, checks=checks,
                blockers=[name for name, passed in checks.items() if not passed],
                HISTORICAL_ADJUSTED_SNAPSHOT_NOT_AVAILABLE=not historical_snapshot_available,
                historical_snapshot_is_blocker=False, amount_status=amount_status,
                amount_is_adjustment_blocker=False)
