"""Small contract checks for the phase 2.5 OOF metrics."""

import numpy as np
import pandas as pd

from src.research.ml_ranking_metrics import evaluate_oof


def _frame():
    rows = []
    # One security has a continuous 10% run across two blocks.  The second
    # security supplies an unknown top slot and a market opportunity winner.
    for pos, block in [(1, "A"), (2, "A"), (3, "B")]:
        for symbol in ("600000", "600001"):
            is_winner = symbol == "600000"
            row = {
                "exchange": "SSE",
                "symbol": symbol,
                "date": pd.Timestamp("2026-01-02") + pd.Timedelta(days=pos - 1),
                "market_pos": pos,
                "block": block,
                "label_valid": not (symbol == "600001" and pos == 1),
                "outcome_max_return_1_5d": 0.30 if is_winner else 0.10,
                "tie_key": f"{pos}-{symbol}",
                "rank_M1": 1 if is_winner else 2,
                "rank_M2": 1 if is_winner else 2,
                "rank_R3_ALL": 1 if is_winner else 2,
                "rank_R3_GATE": 1 if is_winner else np.nan,
            }
            for signal in "ABCDEFGHIJ":
                row[f"full_signal_{signal}"] = is_winner
            for horizon in ("2_5d", "1_5d"):
                row[f"outcome_hit10_{horizon}"] = is_winner
                row[f"outcome_hit20_{horizon}"] = is_winner and pos == 2
                row[f"outcome_hit30_{horizon}"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_stock_day_and_daily_denominators_and_unknowns():
    out = evaluate_oof(_frame())
    row = out["metrics"].query(
        "method == 'M1' and block == 'A' and horizon == '2_5d' and k == 1 and threshold == 10"
    ).iloc[0]
    assert (row.selected, row.evaluable, row.hits) == (2, 2, 2)
    assert row.successful_days == 2 and row.days == 2

    # The lower ranked unknown row is never used to refill a missing gate slot.
    gate = out["metrics"].query(
        "method == 'R3_GATE' and block == 'A' and horizon == '2_5d' and k == 1 and threshold == 10"
    ).iloc[0]
    assert gate.selected == 2 and gate.evaluable == 2
    daily = out["daily"].query(
        "method == 'R3_GATE' and block == 'ALL' and horizon == '2_5d' and k == 1 and threshold == 10"
    )
    assert len(daily) == 3
    assert not out["daily"].duplicated(["method", "k", "threshold", "date"]).any()


def test_opportunity_keeps_gate_omission_in_denominator_and_uses_primary_tie():
    frame = _frame()
    frame.loc[frame.symbol.eq("600000"), "outcome_max_return_1_5d"] = 0.30
    frame.loc[frame.symbol.eq("600001"), "outcome_max_return_1_5d"] = 0.30
    frame.loc[frame.symbol.eq("600001"), "tie_key"] = "00"
    out = evaluate_oof(frame)
    opp = out["opportunity"].query("block == 'ALL'").set_index("method")
    assert opp.loc["R3_GATE", "unranked_days"] == 2
    assert opp.loc["R3_GATE", "winner_in_top100_pct"] == 100 / 3
    winners = out["winners"]
    assert winners.groupby("date").size().to_list() == [1, 2, 2]
    assert winners.groupby("date")["is_primary"].sum().eq(1).all()


def test_common_episode_crosses_block_and_20_member_is_not_borrowed_from_10():
    out = evaluate_oof(_frame())
    episodes = out["episodes"]
    across = episodes.query("method == 'M1' and block == 'ALL' and k == 3")
    ten = across.query("threshold == 10").iloc[0]
    twenty = across.query("threshold == 20").iloc[0]
    assert ten.total_episodes == 1 and ten.captured_episodes == 1
    assert twenty.total_episodes == 1 and twenty.captured_episodes == 1
    assert tuple(ten.episode_ids) == tuple(twenty.episode_ids)
    block_b = episodes.query("method == 'M1' and block == 'B' and k == 3 and threshold == 20").iloc[0]
    assert block_b.total_episodes == 1


def test_unknown_top_slot_is_not_backfilled_and_daily_rate_is_not_stock_day_rate():
    frame=_frame().loc[lambda d:d.market_pos.le(2)].copy()
    # The rank-1 row on day 1 is unknown; rank 2 is a known winner.
    frame.loc[frame.market_pos.eq(1)&frame.symbol.eq('600000'),'label_valid']=False
    frame.loc[frame.symbol.eq('600001'),'label_valid']=True
    frame.loc[frame.symbol.eq('600001'),'outcome_hit10_2_5d']=True
    out=evaluate_oof(frame)
    one=out['metrics'].query("method=='M1' and block=='ALL' and horizon=='2_5d' and k==1 and threshold==10").iloc[0]
    assert one.selected==2 and one.evaluable==1 and one.hits==1
    assert one.hit_rate_pct==100 and one.daily_success_lower_pct==50
    assert one.unresolved_days==1 and one.daily_success_upper_pct==100


def test_frozen_episode_horizon_remains_one_to_five_when_strict_target_differs():
    frame=_frame()
    for n in (10,20,30):frame[f'outcome_hit{n}_2_5d']=False
    out=evaluate_oof(frame)
    strict=out['metrics'].query("method=='M1' and block=='ALL' and horizon=='2_5d' and k==3 and threshold==10").iloc[0]
    legacy=out['episodes'].query("method=='M1' and block=='ALL' and k==3 and threshold==10").iloc[0]
    assert strict.hits==0
    assert legacy.horizon=='1_5d' and legacy.total_episodes==1 and legacy.captured_episodes==1
