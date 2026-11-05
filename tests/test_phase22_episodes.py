"""Phase 2.2 episode and causal-feature regressions using synthetic data only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.research.episodes import (
    additional_features,
    build_episodes,
    consecutive_groups,
    evaluate,
    new_signals,
)


PROJECT = Path(__file__).resolve().parents[1]


def _cfg():
    with (PROJECT / "config" / "phase22_research.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _bars(n=90):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100.0 + np.arange(n, dtype=float) * 0.25
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + np.arange(n, dtype=float) * 10.0,
            "amount": close * (1000.0 + np.arange(n, dtype=float) * 10.0),
        },
        index=dates,
    )


def test_additional_features_are_causal_to_future_bars_and_events():
    bars = _bars()
    cutoff = 70
    event_date = bars.index[cutoff + 5]
    baseline = additional_features(bars, [(event_date, 2.0, 5.0)])

    poisoned = bars.copy(deep=True)
    price_volume = poisoned.columns.get_indexer(["open", "high", "low", "close", "volume"])
    poisoned.iloc[cutoff + 1 :, price_volume] = 10**9
    changed = additional_features(poisoned, [(event_date, 2.0, 5.0)])

    pdt.assert_frame_equal(baseline.iloc[: cutoff + 1], changed.iloc[: cutoff + 1])
    future_event_changed = additional_features(bars, [(event_date, 0.2, -99.0)])
    pdt.assert_frame_equal(baseline.iloc[: cutoff + 1], future_event_changed.iloc[: cutoff + 1])


def test_additional_features_ignore_amount_and_other_non_ohlcv_columns():
    bars = _bars()
    poisoned = bars.copy(deep=True)
    poisoned["amount"] = -10**30
    poisoned["Amount"] = 10**30
    pdt.assert_frame_equal(additional_features(bars), additional_features(poisoned))


def test_additional_features_formulae_include_volatility_acceleration_and_shadows():
    bars = _bars()
    result = additional_features(bars).iloc[-1]
    close = bars["close"].to_numpy()
    volume = bars["volume"].to_numpy()
    returns = close[1:] / close[:-1] - 1.0

    np.testing.assert_allclose(result["volatility_5"], np.std(returns[-5:], ddof=0))
    np.testing.assert_allclose(
        result["volume_acceleration_3"], volume[-3:].mean() / volume[-6:-3].mean() - 1.0
    )
    np.testing.assert_allclose(
        result["price_acceleration_3"],
        (close[-1] / close[-4] - 1.0) - (close[-4] / close[-7] - 1.0),
    )
    width = bars["high"].iloc[-1] - bars["low"].iloc[-1]
    np.testing.assert_allclose(
        result["upper_shadow_ratio"],
        (bars["high"].iloc[-1] - max(bars["open"].iloc[-1], bars["close"].iloc[-1])) / width,
    )
    np.testing.assert_allclose(
        result["lower_shadow_ratio"],
        (min(bars["open"].iloc[-1], bars["close"].iloc[-1]) - bars["low"].iloc[-1]) / width,
    )


def test_additional_features_do_not_skip_missing_market_bars():
    bars = _bars()
    missing = bars.index[70]
    bars.loc[missing, ["open", "high", "low", "close", "volume"]] = np.nan
    result = additional_features(bars)
    assert result.iloc[72]["volatility_5"] != result.iloc[72]["volatility_5"]
    assert result.iloc[72]["volume_acceleration_3"] != result.iloc[72]["volume_acceleration_3"]


def test_new_signals_do_not_read_forward_label_columns():
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "breakout_20d": 0.0,
            "ret_20d": 0.1,
            "ret_3d": 0.1,
            "price_acceleration_3": 0.1,
            "volume_ratio_5": 2.0,
            "close_vs_ma10": 0.0,
            "prior_compression5": 0.5,
            "close_location": 0.8,
            "ret_1d": 0.1,
            "prior_volume_spike3": 3.0,
            "max_drawdown_10": 0.1,
            "recovery_from_drawdown_3": 0.1,
            "consecutive_small_up_days": 4.0,
            "ma10_slope": 0.1,
        },
        index=index,
    )
    labels = ["ret_max_5d", "TARGET_5D_10", "raw_ret_max_5d", "episode_id"]
    with_labels = frame.assign(**{name: 10**9 for name in labels})
    pdt.assert_frame_equal(new_signals(frame, _cfg()), new_signals(with_labels, _cfg()))


def test_consecutive_groups_split_gaps_and_exchange_identity():
    frame = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE", "SSE", "SZSE", "SZSE"],
            "symbol": ["000001"] * 5,
            "market_pos": [1, 2, 4, 1, 2],
        },
        index=range(5),
    )
    groups = consecutive_groups(frame, pd.Series(True, index=frame.index))
    assert groups.loc[0] == groups.loc[1]
    assert groups.loc[2] != groups.loc[1]
    assert groups.loc[3] != groups.loc[2]
    assert groups.loc[3] == groups.loc[4]
    assert groups.nunique() == 3


def _episode_frame(specs):
    rows = []
    first_date = pd.Timestamp("2025-02-01")
    for spec in specs:
        pos = spec["market_pos"]
        date = first_date + pd.Timedelta(days=len(rows))
        row = {
            "exchange": "SSE",
            "symbol": "600000",
            "market_pos": pos,
            "date": date,
            "hit10": False,
            "hit20": False,
            "hit30": False,
            "hit40": False,
            "first_hit_date10": date,
            "peak_date": date,
            "ret_max_5d": 0.2,
            "peak_return_5d": 0.2,
        }
        row.update(spec)
        for n in (10, 20, 30, 40):
            row[f"days_to_{n}pct"] = 1
            row[f"first_hit_pos{n}"] = pos
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_episodes_merges_continuous_winners_and_nests_20_30_on_common_id():
    frame = _episode_frame(
        [
            {"market_pos": 1, "hit10": True},
            {"market_pos": 2, "hit10": True, "hit20": True},
            {"market_pos": 3, "hit10": True, "hit20": True, "hit30": True},
            {"market_pos": 4, "hit10": False},
            {"market_pos": 5, "hit10": True, "hit30": True},
        ]
    )
    episodes, membership = build_episodes(frame)
    assert len(episodes) == 2
    assert membership.iloc[0] == membership.iloc[1] == membership.iloc[2]
    assert pd.isna(membership.iloc[3])
    assert membership.iloc[4] != membership.iloc[0]
    first = episodes.iloc[0]
    assert bool(first["episode_20"])
    assert bool(first["episode_30"])
    assert first["episode_id"] == membership.iloc[0]


def test_build_episodes_breaks_on_missing_market_pos():
    frame = _episode_frame(
        [
            {"market_pos": 1, "hit10": True},
            {"market_pos": 2, "hit10": True},
            {"market_pos": 4, "hit10": True},
        ]
    )
    episodes, membership = build_episodes(frame)
    assert len(episodes) == 2
    assert membership.iloc[0] == membership.iloc[1]
    assert membership.iloc[2] != membership.iloc[1]


def test_missing_calendar_position_is_rejected_not_invented():
    frame = _episode_frame([{"market_pos": np.nan, "hit10": True}])
    with pytest.raises(ValueError, match='MARKET_POSITION_REQUIRED'):
        build_episodes(frame)


def test_evaluate_captures_a_three_day_winner_episode_once():
    frame = _episode_frame(
        [
            {"market_pos": 1, "hit10": True, "hit20": True, "rule": True},
            {"market_pos": 2, "hit10": True, "hit20": True, "rule": True},
            {"market_pos": 3, "hit10": True, "hit20": True, "rule": True},
        ]
    )
    episodes, membership = build_episodes(frame)
    frame["episode_id"] = membership
    result = evaluate(frame, episodes, "rule")
    assert result["episodes10"] == 1
    assert result["caught10"] == 1
    assert result["episode_capture10"] == 100.0
    assert result["candidate_episodes"] == 1


def test_evaluate_episode_precision_denominator_includes_failed_signal_cluster():
    frame = _episode_frame(
        [
            {"market_pos": 1, "hit10": True, "rule": True},
            {"market_pos": 2, "hit10": True, "rule": True},
            {"market_pos": 4, "hit10": False, "rule": True},
            {"market_pos": 5, "hit10": False, "rule": True},
        ]
    )
    episodes, membership = build_episodes(frame)
    frame["episode_id"] = membership
    result = evaluate(frame, episodes, "rule")
    assert result["candidate_episodes"] == 2
    assert result["successful_candidate_episodes"] == 1
    assert result["episode_precision"] == 50.0


def test_evaluate_preserves_signed_negative_late_lead():
    frame = _episode_frame(
        [
            {"market_pos": 1, "hit10": True, "hit20": True, "rule": False},
            {"market_pos": 2, "hit10": True, "hit20": True, "rule": True},
        ]
    )
    episodes, membership = build_episodes(frame)
    frame["episode_id"] = membership
    result = evaluate(frame, episodes, "rule")
    assert result["episodes20"] == 1
    assert result["caught20"] == 1
    assert result["median_lead20"] == -1
    assert result["late_detections20"] == 1


def test_discovery_purges_boundary_episode_instead_of_split_reusing_it():
    from scripts.run_phase22_research import scope_data
    frame = _episode_frame([
        {'market_pos':74,'hit10':True,'split':'DISCOVERY'},
        {'market_pos':75,'hit10':True,'split':'PURGED_DISCOVERY'},
        {'market_pos':80,'hit10':True,'split':'VALIDATION'},
    ])
    frame['coverage']='LOW_RESEARCH_COVERAGE'
    episodes, ids = build_episodes(frame)
    frame['episode_id']=ids
    discovery, de, excluded = scope_data(frame,episodes,'DISCOVERY','LOW_RESEARCH_COVERAGE')
    validation, ve, _ = scope_data(frame,episodes,'VALIDATION','LOW_RESEARCH_COVERAGE')
    assert discovery.empty and de.empty and excluded==1
    assert len(validation)==1 and len(ve)==1


def test_future_metadata_cent_threshold_and_incomplete_horizon():
    from scripts.run_phase22_research import future_metadata
    bars=_bars(180)
    bars['close']=10.0;bars['high']=11.0
    result=future_metadata(bars,[],bars.index)
    assert result.days_to_10pct.iloc[60]==1
    assert result.first_hit_pos10.iloc[60]==1
    bars.iloc[62,bars.columns.get_loc('high')]=np.nan
    result=future_metadata(bars,[],bars.index)
    assert pd.isna(result.days_to_10pct.iloc[60])
    assert result.days_to_10pct.iloc[-5:].isna().all()


def test_empty_winner_inventory_has_valid_schema_and_false_candidates():
    frame=_episode_frame([{'market_pos':1,'hit10':False,'rule':True}])
    episodes,ids=build_episodes(frame);frame['episode_id']=ids
    result=evaluate(frame,episodes,'rule')
    assert result['candidate_episodes']==1 and result['episode_precision']==0
    assert result['caught10']==0 and result['episodes10']==0


def test_days_to_threshold_uses_earliest_qualifying_anchor_not_best_return():
    frame=_episode_frame([{'market_pos':1,'hit10':True,'ret_max_5d':.11},
                          {'market_pos':2,'hit10':True,'ret_max_5d':.25}])
    frame.loc[0,'days_to_10pct']=1;frame.loc[1,'days_to_10pct']=5
    episodes,_=build_episodes(frame)
    assert episodes.iloc[0].days_to_10pct==1
    assert episodes.iloc[0].peak_reference_days_to_10pct==5
