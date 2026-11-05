"""Point-in-time affine corporate-action adjustment engine.

The engine never mutates raw bars.  An action is visible only when its ex-date
is no later than the caller's research as-of date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd

from src.data.corporate_actions import get_actions

VALID_METHODS = {"NONE", "FRONT_PIT", "BACK_PIT"}
PRICE_COLUMNS = ("open", "high", "low", "close")


class IncompleteCorporateAction(ValueError):
    """A source event lacks a required numerical input and cannot be guessed."""


@dataclass(frozen=True)
class AffineTransform:
    """Represents adjusted_price = A * raw_price + B."""

    A: float = 1.0
    B: float = 0.0

    def compose_after(self, prior: "AffineTransform") -> "AffineTransform":
        """Return self(prior(price))."""
        return AffineTransform(self.A * prior.A, self.A * prior.B + self.B)


def _as_date(value: object) -> date:
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.date()


def _known(value: object, field: str, event: pd.Series) -> float:
    if value is None or pd.isna(value):
        # TDX category-1 uses a single known songzhuangu amount.  It is stored
        # in bonus_share_per_10 and explicitly leaves transfer NULL rather than
        # inventing a split.  Here only that explicit source convention maps the
        # missing transfer component to zero.
        if field == "transfer_share_per_10" and event.get("status") == "TDX_GBBQ_COMBINED_BONUS_TRANSFER":
            return 0.0
        raise IncompleteCorporateAction(f"{field} is unknown for {event.get('symbol')} {event.get('ex_date')}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise IncompleteCorporateAction(f"{field} must be finite and nonnegative")
    return number


def action_transform(event: pd.Series) -> AffineTransform:
    """Map a pre-ex-date raw price to the post-ex-date economic price basis.

    C, S and R are official reference-basis cash, bonus/transfer and effective
    rights per old share; rights are subscribed at Q:
    ex_reference = (P - C + R*Q) / (1 + S + R).
    Actual shareholder cash entitlement is never a fallback for C.
    """
    cash = _known(event.get("ex_reference_cash_dividend_per_share"), "ex_reference_cash_dividend_per_share", event)
    share_basis = event.get('ex_reference_bonus_transfer_ratio')
    if share_basis is not None and not pd.isna(share_basis):
        shares = _known(share_basis, 'ex_reference_bonus_transfer_ratio', event)
    else:
        bonus = _known(event.get("bonus_share_per_10"), "bonus_share_per_10", event) / 10.0
        transfer = _known(event.get("transfer_share_per_10"), "transfer_share_per_10", event) / 10.0
        shares = bonus + transfer
        flag = event.get('differential_dividend_flag')
        if shares and (flag is None or pd.isna(flag) or bool(flag)):
            raise IncompleteCorporateAction('Nonzero shares need a confirmed ordinary or explicit reference basis')
    reference_rights = event.get('ex_reference_rights_ratio')
    if reference_rights is not None and not pd.isna(reference_rights):
        rights = _known(reference_rights,'ex_reference_rights_ratio',event)
    else:
        rights = _known(event.get("rights_share_per_10"), "rights_share_per_10", event) / 10.0
        if rights:
            raise IncompleteCorporateAction('Effective subscribed rights ratio is required; offered ratio is not a fallback')
    rights_price = _known(event.get("rights_price"), "rights_price", event)
    denominator = 1.0 + shares + rights
    if denominator <= 0:
        raise IncompleteCorporateAction(f"non-positive share denominator for {event.get('symbol')} {event.get('ex_date')}")
    return AffineTransform(1.0 / denominator, (rights * rights_price - cash) / denominator)


def effective_actions(actions: pd.DataFrame, as_of_date: object) -> pd.DataFrame:
    """Filter strictly by ex-date; announced-but-future events never leak in."""
    as_of = _as_date(as_of_date)
    result = actions.copy()
    result["ex_date"] = pd.to_datetime(result["ex_date"], errors="coerce").dt.date
    return result.loc[result["ex_date"].notna() & (result["ex_date"] <= as_of)].sort_values("ex_date", kind="stable").reset_index(drop=True)


def annotate_history_boundary(actions: pd.DataFrame, *, listing_date: object,
                              first_raw_bar_date: object) -> pd.DataFrame:
    """Annotate a verified security-history boundary, not a caller's slice.

    Both dates must describe the security's complete history. A pre-boundary
    event is excluded by default; no exception is inferred from a file existing.
    This helper never changes event parameters or the input frame.
    """
    if pd.isna(listing_date) or pd.isna(first_raw_bar_date):
        raise ValueError('Both verified history dates are required')
    listing, first = _as_date(listing_date), _as_date(first_raw_bar_date)
    if pd.isna(listing) or pd.isna(first):
        raise ValueError('Both verified history dates are required')
    result = actions.copy(deep=True)
    result['listing_date'] = listing.isoformat()
    result['first_raw_bar_date'] = first.isoformat()
    dates = pd.to_datetime(result['ex_date'], errors='raise').dt.date
    result['pre_listing_event_flag'] = (dates < listing) | (dates < first)
    result['history_boundary_status'] = result['pre_listing_event_flag'].map(
        {True:'EXCLUDED_PRE_HISTORY_NO_OFFICIAL_EXCEPTION',False:'IN_HISTORY'})
    return result


def _history_eligible(actions, listing_date, first_raw_bar_date):
    if listing_date is None and first_raw_bar_date is None:
        if 'listing_date' in actions or 'first_raw_bar_date' in actions:
            if actions.empty:
                return actions
            if not {'listing_date','first_raw_bar_date'}.issubset(actions.columns):
                raise ValueError('Supply both listing_date and first_raw_bar_date')
            if actions[['listing_date','first_raw_bar_date']].isna().any().any():
                raise ValueError('Both verified history dates are required')
            if actions.listing_date.nunique()!=1 or actions.first_raw_bar_date.nunique()!=1:
                raise ValueError('Conflicting security history dates')
            listing_date=actions.listing_date.iloc[0]
            first_raw_bar_date=actions.first_raw_bar_date.iloc[0]
        else:
            # Legacy explicit synthetic/diagnostic records remain readable.
            # Qualified callers must supply both dates explicitly or on actions.
            # Never infer them from a partial window.
            return actions
    if listing_date is None or first_raw_bar_date is None:
        raise ValueError('Supply both listing_date and first_raw_bar_date')
    marked=annotate_history_boundary(actions,listing_date=listing_date,
                                     first_raw_bar_date=first_raw_bar_date)
    return marked.loc[~marked.pre_listing_event_flag].copy()


def build_adjustment_state(symbol: str, as_of_date: object, actions: pd.DataFrame | None = None,
                           *, listing_date: object = None, first_raw_bar_date: object = None) -> dict:
    """Build the front-PIT affine state for prices before all effective actions."""
    selected = get_actions(symbol) if actions is None else actions.loc[actions["symbol"].astype(str).str.zfill(6) == str(symbol).zfill(6)]
    selected = effective_actions(selected, as_of_date)
    selected = _history_eligible(selected, listing_date, first_raw_bar_date)
    state = AffineTransform()
    event_states: list[dict] = []
    for _, event in selected.iterrows():
        event_transform = action_transform(event)
        pre = state
        state = event_transform.compose_after(state)
        event_states.append({
            "ex_date": event["ex_date"].isoformat(),
            "pre_event_state": {"A": pre.A, "B": pre.B},
            "event": {"event_type": event.get("event_type"), "raw_reference": event.get("raw_reference"),
                      "actual_cash_dividend_per_share": event.get('actual_cash_dividend_per_share'),
                      "ex_reference_cash_dividend_per_share": event.get('ex_reference_cash_dividend_per_share'),
                      "cash_basis_status": event.get('cash_basis_status'),
                      "basis_raw_reference": event.get('basis_raw_reference')},
            "post_event_state": {"A": state.A, "B": state.B},
        })
    return {"symbol": str(symbol).zfill(6), "as_of_date": _as_date(as_of_date).isoformat(), "A": state.A, "B": state.B, "effective_event_count": len(event_states), "event_states": event_states}


def _copy_and_validate_bars(raw_bars: pd.DataFrame) -> pd.DataFrame:
    missing = {"date", *PRICE_COLUMNS}.difference(raw_bars.columns)
    if missing:
        raise ValueError(f"raw_bars missing columns: {sorted(missing)}")
    result = raw_bars.copy(deep=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date
    return result

def _apply_transform(frame: pd.DataFrame, mask: pd.Series, transform: AffineTransform) -> None:
    for column in PRICE_COLUMNS:
        frame.loc[mask, column] = frame.loc[mask, column].astype(float) * transform.A + transform.B


def adjust_prices_asof(
    symbol: str,
    raw_bars: pd.DataFrame,
    as_of_date: object,
    method: str,
    actions: pd.DataFrame | None = None,
    *, listing_date: object = None, first_raw_bar_date: object = None,
) -> pd.DataFrame:
    """Return a derived price frame for NONE, FRONT_PIT or BACK_PIT.

    ``FRONT_PIT`` maps only bars before an effective action to that action's
    post-event basis. ``BACK_PIT`` keeps pre-event bars raw and maps later bars
    to the earlier basis using the inverse affine transform.
    """
    method = method.upper()
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_METHODS)}")
    result = _copy_and_validate_bars(raw_bars)
    if method == "NONE":
        return result
    selected = get_actions(symbol) if actions is None else actions.loc[actions["symbol"].astype(str).str.zfill(6) == str(symbol).zfill(6)]
    selected = effective_actions(selected, as_of_date)
    selected = _history_eligible(selected, listing_date, first_raw_bar_date)
    if method == "FRONT_PIT":
        for _, event in selected.iterrows():
            _apply_transform(result, result["date"] < event["ex_date"], action_transform(event))
    else:  # BACK_PIT: reverse chronological inverse transforms.
        for _, event in selected.iloc[::-1].iterrows():
            transform = action_transform(event)
            inverse = AffineTransform(1.0 / transform.A, -transform.B / transform.A)
            _apply_transform(result, result["date"] >= event["ex_date"], inverse)
    return result
