import pandas as pd
import pytest

from src.data.adjustment_engine import (
    IncompleteCorporateAction,
    adjust_prices_asof,
    build_adjustment_state,
)


def _actions():
    return pd.DataFrame([
        {
            "symbol": "000001", "ex_date": "2025-08-01",
            "cash_dividend_per_10": 1.0, "bonus_share_per_10": 0.0,
            "actual_cash_dividend_per_share": 0.1,
            "ex_reference_cash_dividend_per_share": 0.1,
            "differential_dividend_flag": False,
            "transfer_share_per_10": None, "rights_share_per_10": 0.0,
            "rights_price": 0.0, "event_type": "CASH_DIVIDEND",
            "status": "TDX_GBBQ_COMBINED_BONUS_TRANSFER",
            "raw_reference": "fixture",
        },
    ])


def _bars():
    return pd.DataFrame([
        {"date": "2025-07-31", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
        {"date": "2025-08-01", "open": 9.9, "high": 9.9, "low": 9.9, "close": 9.9},
    ])


def test_future_ex_date_cannot_affect_earlier_asof():
    original = _bars()
    adjusted = adjust_prices_asof("000001", original, "2025-07-31", "FRONT_PIT", _actions())
    assert adjusted["close"].tolist() == [10.0, 9.9]
    assert original["close"].tolist() == [10.0, 9.9]  # no raw mutation


def test_ex_date_becomes_effective_on_the_ex_date():
    adjusted = adjust_prices_asof("000001", _bars(), "2025-08-01", "FRONT_PIT", _actions())
    assert adjusted.loc[0, "close"] == pytest.approx(9.9)
    assert adjusted.loc[1, "close"] == pytest.approx(9.9)
    state = build_adjustment_state("000001", "2025-08-01", _actions())
    assert state["effective_event_count"] == 1
    assert state["A"] == pytest.approx(1.0)
    assert state["B"] == pytest.approx(-0.1)


def test_affine_bonus_and_rights_round_trip():
    actions = _actions()
    actions.loc[0, ["cash_dividend_per_10", "bonus_share_per_10", "rights_share_per_10", "rights_price"]] = [1.0, 1.0, 1.0, 8.0]
    actions.loc[0, 'ex_reference_rights_ratio'] = 0.1
    bars = pd.DataFrame([
        {"date": "2025-07-31", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
        {"date": "2025-08-01", "open": 8.9166666667, "high": 8.9166666667, "low": 8.9166666667, "close": 8.9166666667},
    ])
    front = adjust_prices_asof("000001", bars, "2025-08-01", "FRONT_PIT", actions)
    back = adjust_prices_asof("000001", bars, "2025-08-01", "BACK_PIT", actions)
    assert front.loc[0, "close"] == pytest.approx(8.9166666667)
    assert back.loc[1, "close"] == pytest.approx(10.0)


def test_explicit_transfer_share_is_supported_separately_from_bonus():
    actions = _actions()
    actions.loc[0, ["cash_dividend_per_10", "bonus_share_per_10", "transfer_share_per_10", "rights_share_per_10", "rights_price"]] = [0.0, 0.0, 1.0, 0.0, 0.0]
    actions.loc[0, 'ex_reference_cash_dividend_per_share'] = 0.0
    adjusted = adjust_prices_asof("000001", _bars(), "2025-08-01", "FRONT_PIT", actions)
    assert adjusted.loc[0, "close"] == pytest.approx(10.0 / 1.1)


def test_unknown_component_is_not_silently_zeroed():
    actions = _actions()
    actions.loc[0, "ex_reference_cash_dividend_per_share"] = None
    with pytest.raises(IncompleteCorporateAction):
        adjust_prices_asof("000001", _bars(), "2025-08-01", "FRONT_PIT", actions)


def test_announced_october_event_is_not_effective_in_september():
    actions = _actions()
    actions.loc[0, 'ex_date'] = '2026-10-15'
    actions['announcement_date'] = '2026-09-20'
    for method in ('FRONT_PIT', 'BACK_PIT'):
        actual = adjust_prices_asof('000001', _bars(), '2026-09-30', method, actions)
        pd.testing.assert_frame_equal(actual, adjust_prices_asof('000001', _bars(), '2026-09-30', 'NONE'))
    assert build_adjustment_state('000001', '2026-09-30', actions)['effective_event_count'] == 0
    assert build_adjustment_state('000001', '2026-10-15', actions)['effective_event_count'] == 1
