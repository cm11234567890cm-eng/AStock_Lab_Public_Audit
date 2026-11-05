"""Phase 2.3 coverage and validity regressions using synthetic market data."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.research.coverage_recovery import (
    CORE_SHORT_FEATURE_SET,
    FEATURE_WINDOWS,
    RULE_DEPENDENCIES,
    bounded_streak,
    eligibility,
    exclusion_strings,
    feature_masks,
    interval_flags,
)
from src.research.target_discovery import labels


def _bars(n=90):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


def _finite_values(index):
    return pd.DataFrame(
        {name: np.ones(len(index), dtype=float) for name in FEATURE_WINDOWS},
        index=index,
    )


def test_t_minus_40_event_only_crosses_the_60_session_basis():
    bars = _bars()
    values = _finite_values(bars.index)
    t = bars.index[70]
    event = bars.index[30]  # exactly t-40 on this synthetic calendar

    _, ca, _, _ = feature_masks(bars, values, [event], [])

    assert not bool(ca.loc[t, "ret_5d"])
    assert not bool(ca.loc[t, "ret_10d"])
    assert bool(ca.loc[t, "position_60d"])
    assert bool(ca.loc[t, "distance_to_60d_high"])


def test_event_at_window_start_does_not_cross_the_basis():
    calendar = pd.date_range("2025-01-01", periods=12, freq="D")
    row = 7
    start = calendar[row - 3]

    event, _ = interval_flags(calendar, [start], [], oldest=3)
    next_event, _ = interval_flags(calendar, [calendar[row - 2]], [], oldest=3)

    assert not bool(event[row])
    assert bool(next_event[row])


def test_label_t_to_t_plus_5_can_cross_future_ca_without_changing_causal_masks():
    bars = _bars(90)
    values = _finite_values(bars.index)
    t = bars.index[65]
    future_ca = bars.index[70]  # t+5, visible to the label but not to features at t

    plain_labels = labels(bars, [])
    ca_labels = labels(bars, [(future_ca, 2.0, 0.0)])
    assert not bool(plain_labels.loc[t, "label_crosses_qualified_ca"])
    assert bool(ca_labels.loc[t, "label_crosses_qualified_ca"])

    plain_masks, _, _, _ = feature_masks(bars, values, [], [])
    future_masks, _, _, _ = feature_masks(bars, values, [future_ca], [])
    pdt.assert_frame_equal(plain_masks.loc[: bars.index[69]], future_masks.loc[: bars.index[69]])


def test_unknown_interval_propagates_to_overlapping_feature_and_not_disjoint_rows():
    bars = _bars()
    values = _finite_values(bars.index)
    t = bars.index[70]
    unknown = [(bars.index[66], bars.index[68])]

    _, unknown_flags = interval_flags(bars.index, [], unknown, oldest=5)
    assert bool(unknown_flags[70])
    assert not bool(unknown_flags[65])

    masks, _, unknown_frame, _ = feature_masks(bars, values, [], unknown)
    assert bool(unknown_frame.loc[t, "ret_5d"])
    assert not bool(masks.loc[t, "ret_5d"])


def test_core_set_is_derived_from_actual_lookback_and_excludes_long_dependencies():
    expected = {name for name, (oldest, _) in FEATURE_WINDOWS.items() if oldest <= 10}
    core = set(CORE_SHORT_FEATURE_SET)
    assert core == expected
    assert all(FEATURE_WINDOWS[name][0] <= 10 for name in core)
    assert {"prior_compression5", "recovery_from_drawdown_3", "breakout_20d"}.isdisjoint(core)

    index = pd.date_range("2025-01-01", periods=2, freq="D")
    validity = pd.DataFrame(
        {name: True for deps in RULE_DEPENDENCIES.values() for name in deps}, index=index
    )
    permitted = eligibility(validity, "CORE_SHORT_FEATURE_SET")
    assert permitted["J"].all()
    assert not permitted[["F", "G", "I"]].any().any()


def test_capped_ten_streak_threshold_matches_full_streak_on_long_sequence():
    n = 48
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100.0 * (1.01 ** np.arange(n))
    bars = pd.DataFrame(
        {
            "open": close / 1.005,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )

    capped = bounded_streak(bars)
    full = []
    run = 0
    for pos, value in enumerate(close):
        if pos and 0 < value / close[pos - 1] - 1 <= 0.015 and value > bars.open.iloc[pos]:
            run += 1
        else:
            run = 0
        full.append(run)
    full = pd.Series(full, index=dates)

    # The bounded feature has a ten-session warm-up; after that, the >=3
    # decision must agree with the uncapped historical streak, including a
    # streak much longer than ten sessions.
    assert (capped.iloc[10:] >= 3).equals((full.iloc[10:] >= 3))
    assert capped.iloc[-1] == 10
    assert full.iloc[-1] == n - 1


def test_missing_close_history_is_missing_and_cannot_become_bool_safe():
    bars = _bars(20)
    missing_row = bars.index[7]
    target = bars.index[10]
    bars.loc[missing_row, "close"] = np.nan
    values = _finite_values(bars.index)

    masks, _, _, missing = feature_masks(bars, values, [], [])

    assert bool(missing.loc[target, "ret_5d"])
    assert not bool(masks.loc[target, "ret_5d"])
    assert bool(missing.loc[target, "consecutive_small_up_days_10"])
    assert not bool(masks.loc[target, "consecutive_small_up_days_10"])


def test_exclusion_strings_keep_all_reasons_and_choose_first_primary():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    flags = pd.DataFrame(
        {
            "DATA_GAP": [True, False, True, False],
            "CORPORATE_ACTION": [True, True, False, False],
            "MISSING_HISTORY": [False, True, True, False],
        },
        index=index,
    )

    result = exclusion_strings(flags)

    assert result.loc[index[0], "exclusion_reasons"] == "DATA_GAP;CORPORATE_ACTION"
    assert result.loc[index[0], "primary_exclusion_reason"] == "DATA_GAP"
    assert result.loc[index[1], "exclusion_reasons"] == "CORPORATE_ACTION;MISSING_HISTORY"
    assert result.loc[index[1], "primary_exclusion_reason"] == "CORPORATE_ACTION"
    assert result.loc[index[2], "exclusion_reasons"] == "DATA_GAP;MISSING_HISTORY"
    assert result.loc[index[2], "primary_exclusion_reason"] == "DATA_GAP"
    assert result.loc[index[3], "exclusion_reasons"] == ""
    assert result.loc[index[3], "primary_exclusion_reason"] == "NONE"
