"""Phase 2.1 causal research regressions using only synthetic bars."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.research.target_discovery import event_mask, features, labels, signals


def _bars(n=160):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100.0 + np.arange(n, dtype=float) * 0.20 + np.sin(np.arange(n)) * 0.1
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + np.arange(n, dtype=float),
            "amount": (1000.0 + np.arange(n, dtype=float)) * close,
        },
        index=dates,
    )


def _thresholds():
    return {
        "A": {
            "ret20_min": -999.0,
            "close_vs_ma5_min": -999.0,
            "close_vs_ma5_max": 999.0,
        },
        "B": {"close_location_min": -999.0},
        "C": {"volume_ratio5_min": -999.0},
        "D": {"limit_proxy_count5_min": -999.0, "volume_ratio5_min": -999.0},
        "E": {
            "ret20_max": 999.0,
            "position60_max": 999.0,
            "close_location_min": -999.0,
            "volume_ratio5_min": -999.0,
        },
    }


def test_future_ohlcv_mutation_cannot_change_features_or_signals_before_cutoff():
    bars = _bars()
    cutoff = 100
    future = bars.copy(deep=True)
    future.iloc[cutoff + 1 :, future.columns.get_indexer(["open", "high", "low", "close", "volume"])] *= 17.0
    future.iloc[cutoff + 1 :, future.columns.get_indexer(["open", "high", "low", "close", "volume"])] += 31.0

    before = features(bars)
    after = features(future)
    pdt.assert_frame_equal(before.iloc[: cutoff + 1], after.iloc[: cutoff + 1])
    pdt.assert_frame_equal(
        signals(before, _thresholds()).iloc[: cutoff + 1],
        signals(after, _thresholds()).iloc[: cutoff + 1],
    )


def test_future_qualified_event_date_and_parameters_cannot_leak_before_ex_date():
    bars = _bars()
    ex_pos = 120
    ex = bars.index[ex_pos]
    changed_ex = bars.index[ex_pos + 2]
    baseline = features(bars, [(ex, 2.0, 1.0)])
    changed = features(bars, [(changed_ex, 7.0, -9.0)])

    pdt.assert_frame_equal(baseline.loc[baseline.index < ex], changed.loc[changed.index < ex])
    assert not baseline.loc[ex:].equals(changed.loc[ex:])


def test_current_event_transforms_historical_price_basis_only():
    bars = _bars()
    ex_pos = 100
    ex = bars.index[ex_pos]
    adjusted = features(bars, [(ex, 2.0, 1.0)])
    expected_previous = 2.0 * bars.close.iloc[ex_pos - 1] + 1.0
    expected_ret = bars.close.iloc[ex_pos] / expected_previous - 1.0
    np.testing.assert_allclose(adjusted.loc[ex, "ret_1d"], expected_ret)

    historical = 2.0 * bars.close.iloc[ex_pos - 4 : ex_pos] + 1.0
    expected_ma5 = (historical.sum() + bars.close.iloc[ex_pos]) / 5.0
    np.testing.assert_allclose(adjusted.loc[ex, "close_vs_ma5"], bars.close.iloc[ex_pos] / expected_ma5 - 1.0)


def test_labels_last_five_rows_are_censored_and_missing_bar_is_not_skipped():
    bars = _bars(100)
    missing_pos = 85
    bars.iloc[missing_pos] = np.nan
    result = labels(bars)
    full_horizon = [
        "ret_max_5d",
        "raw_ret_max_5d",
        "ret_close_5d",
        "max_adverse_5d",
        "TARGET_5D_10",
        "TARGET_2_5D_10",
    ]
    assert result.loc[bars.index[-5:], full_horizon].isna().all().all()
    assert result.loc[bars.index[missing_pos - 5], full_horizon].isna().all()


def test_target_2_5_is_the_five_day_target():
    result = labels(_bars(100))
    pdt.assert_series_equal(result["TARGET_2_5D_10"], result["TARGET_5D_10"], check_names=False)


def test_breakout_uses_prior_high_and_excludes_current_high():
    bars = _bars(100)
    bars.iloc[-21:-1, bars.columns.get_loc("close")] = 100.0
    bars.iloc[-21:-1, bars.columns.get_loc("high")] = 105.0
    bars.iloc[-1, bars.columns.get_loc("close")] = 110.0
    bars.iloc[-1, bars.columns.get_loc("high")] = 120.0
    bars.iloc[-1, bars.columns.get_loc("low")] = 109.0
    result = features(bars)
    assert result.iloc[-1]["breakout_20d"] == 1.0


def test_volume_ratio_denominator_excludes_today():
    bars = _bars(100)
    bars.iloc[-11:-1, bars.columns.get_loc("volume")] = 100.0
    bars.iloc[-1, bars.columns.get_loc("volume")] = 1000.0
    result = features(bars)
    np.testing.assert_allclose(result.iloc[-1]["volume_ratio_5"], 10.0)
    np.testing.assert_allclose(result.iloc[-1]["volume_ratio_10"], 10.0)


def test_amount_poison_is_ignored_by_ohlcv_features():
    bars = _bars()
    poisoned = bars.copy(deep=True)
    poisoned["amount"] = np.nan
    poisoned["Amount"] = -10**30
    pdt.assert_frame_equal(features(bars), features(poisoned))


def test_event_mask_is_exactly_five_days_before_and_sixty_after_event():
    calendar = pd.date_range("2025-01-01", periods=170, freq="D")
    event = calendar[100]
    mask = event_mask(calendar, [event])
    expected = np.zeros(len(calendar), dtype=bool)
    expected[95:161] = True
    np.testing.assert_array_equal(mask, expected)
    assert not mask[94]
    assert not mask[161]


def test_fixed_signals_ignore_forward_label_columns():
    frame = features(_bars())
    poisoned = frame.copy(deep=True)
    for name in ("ret_close_1d", "ret_max_5d", "raw_ret_max_5d", "TARGET_5D_10", "label_crosses_qualified_ca"):
        poisoned[name] = 10**9
    pdt.assert_frame_equal(signals(frame, _thresholds()), signals(poisoned, _thresholds()))


def test_known_cash_ten_to_nine_event_has_normalized_and_raw_forward_labels():
    bars = _bars(100)
    anchor_pos = 78
    ex_pos = anchor_pos + 2
    for pos in range(anchor_pos, anchor_pos + 6):
        bars.iloc[pos, bars.columns.get_loc("close")] = 10.0
        bars.iloc[pos, bars.columns.get_loc("high")] = 10.5
        bars.iloc[pos, bars.columns.get_loc("low")] = 9.5
    result = labels(bars, [(bars.index[ex_pos], 1.0, -1.0)])
    anchor = bars.index[anchor_pos]
    np.testing.assert_allclose(result.loc[anchor, "ret_max_5d"], 0.15)
    np.testing.assert_allclose(result.loc[anchor, "raw_ret_max_5d"], 0.05)
    assert bool(result.loc[anchor, "TARGET_5D_10"])
    assert not bool(result.loc[anchor, "raw_ret_max_5d"] >= 0.10)


def test_max_drawdown_uses_only_the_past_window():
    bars = _bars(100)
    pos = 80
    close_col = bars.columns.get_loc("close")
    bars.iloc[pos - 4 : pos + 1, close_col] = [100.0, 110.0, 90.0, 95.0, 100.0]
    result = features(bars)
    expected = 1.0 - 90.0 / 110.0
    np.testing.assert_allclose(result.iloc[pos]["max_drawdown_5"], expected)

    future = bars.copy(deep=True)
    future.iloc[pos + 1 :, close_col] = 1.0
    future_result = features(future)
    np.testing.assert_allclose(future_result.iloc[pos]["max_drawdown_5"], expected)


def test_missing_middle_market_bar_censors_entire_five_day_label():
    bars = _bars(100)
    bars.iloc[82] = np.nan
    result = labels(bars)
    assert result.iloc[80][['TARGET_5D_10', 'ret_close_5d', 'ret_max_5d']].isna().all()


def test_exact_ten_percent_cent_price_not_lost_to_float_division():
    bars = _bars(100)
    bars.loc[:, 'close'] = 10.0
    bars.loc[:, 'high'] = 11.0
    result = labels(bars)
    assert result.TARGET_5D_10.iloc[:-5].eq(1).all()
    bars.loc[:, 'high'] = 10.99
    assert labels(bars).TARGET_5D_10.iloc[:-5].eq(0).all()


def test_security_identity_uses_exchange_not_code_only():
    from scripts.run_phase21_research import board_for
    assert board_for('000001', 'SSE') is None  # Shanghai index, not Shenzhen stock.
    assert board_for('000001', 'SZSE') == 'SZSE_MAIN'
    assert board_for('689009', 'SSE') is None  # CDR excluded.
    assert board_for('430001', 'BSE') is None  # Cannot infer BSE from NEEQ-like code.
    assert board_for('920982', 'BSE', 'BSE') == 'BSE'


def test_summary_winner_and_label_agree_at_exact_price_threshold():
    from scripts.run_phase21_research import winner
    bars = _bars(100)
    bars.loc[:, 'close'] = 13.3
    bars.loc[:, 'high'] = 14.63
    result = labels(bars)
    result['close_t'] = bars.close
    complete = result.TARGET_5D_10.notna()
    np.testing.assert_array_equal(winner(result.loc[complete], 10), result.loc[complete, 'TARGET_5D_10'].astype(bool))
