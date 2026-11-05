import numpy as np
import pandas as pd
import pandas.testing as pdt
from src.research.objective_tradability import (classify_entries,scenario_masks,
    evaluate_selection,fallback_open_only,decile_table,opportunity_table)

def sample():
    rows=[]
    for date in pd.to_datetime(['2026-08-05','2026-08-06']):
        for r,op in enumerate([110.,105.,103.,100.,99.,108.,108.01,102.,101.,104.],1):
            row=dict(exchange='SSE',symbol=f'600{r:03d}',date=date,block='B1',rank_M1=r,rank_R3_ALL=11-r,
                score_M1=1-r*.05,close_t=100.,next_open=op,next_high=op+1,next_low=op-1,next_close=op,
                next_volume=1000.,nominal_limit_ratio=.1,next_bar_present=True,future2_5_complete=True,label_valid=True)
            for j in (2,3,4,5):row['future_high_'+str(j)]=120.
            for n in (10,20,30):row[f'outcome_hit{n}_2_5d']=n<=20
            rows.append(row)
    return pd.DataFrame(rows)

def test_gap_boundaries_exact_and_open_label_changes_denominator():
    x=classify_entries(sample())
    assert list(x.gap_bucket.iloc[:7])==['GAP_GT_8','GAP_5_8','GAP_3_5','GAP_0_3','NORMAL_ENTRY','GAP_5_8','GAP_GT_8']
    assert x.E2_status.iloc[1]=='SKIPPED_HIGH_GAP'
    assert not x.entry_hit10.iloc[0]  # 120 / 110 < 1.10
    assert x.entry_hit10.iloc[1]
    assert not x.entry_hit20.iloc[2]
    assert x.entry_hit20.iloc[3]

def test_one_price_unknown_is_not_declared_unexecutable():
    a=sample();a.loc[0,['next_open','next_high','next_low','next_close']]=110.
    x=classify_entries(a)
    assert x.one_price_up.iloc[0] and x.pattern_class.iloc[0]=='ONE_PRICE_LIMIT_UP'
    assert x.intraday_entry_status.iloc[0]=='INTRADAY_ENTRY_UNKNOWN'
    assert pd.isna(x.normal_continuous_auction_entry_confirmed.iloc[0])
    assert x.E1_entry_eligible.iloc[0] and not x.E3_entry_eligible.iloc[0]

def test_E3_later_opening_does_not_claim_fill_at_open():
    x=classify_entries(sample())
    assert x.opened_below_nominal_limit_during_day_proxy.iloc[0]
    assert not x.E3_entry_eligible.iloc[0]

def test_fallback_uses_open_only_and_never_high_low_labels_or_returns():
    a=sample();b=a.copy()
    for col in ['next_high','next_low','next_close','next_volume','future_high_2','future_high_3','future_high_4','future_high_5']:
        b[col]=99999.
    b['label_valid']=False;b['outcome_hit10_2_5d']=False
    pdt.assert_series_equal(fallback_open_only(a),fallback_open_only(b))
    selected=a.loc[fallback_open_only(a)]
    assert selected.groupby('date').rank_M1.apply(list).tolist()==[[3,4,5],[3,4,5]]

def test_future_CA_invalid_does_not_change_selection_or_deciles():
    a=classify_entries(sample());b=a.copy();b['label_valid']=False
    _,bins_a=decile_table(a);_,bins_b=decile_table(b)
    pdt.assert_frame_equal(bins_a,bins_b)
    pdt.assert_series_equal(fallback_open_only(a),fallback_open_only(b))

def test_skip_is_not_a_failure_and_daily_denominator_all_days():
    a=classify_entries(sample())
    # Day1 top1 skips; day2 top1 meets no-chase rule and hits.
    idx=a.index[a.date.eq(pd.Timestamp('2026-08-06'))&a.rank_M1.eq(1)][0]
    raw=sample();raw.loc[idx,'next_open']=100.
    a=classify_entries(raw)
    metrics,_=evaluate_selection(a,a.rank_M1.le(1),'M1',1)
    r=metrics.query("block=='ALL' and scenario=='E2' and threshold==10").iloc[0]
    assert (r.selected_candidates,r.entry_eligible_candidates,r.evaluable_candidates,r.hit_candidates)==(2,1,1,1)
    assert r.hit_rate_pct==100 and r.daily_success_proxy_pct==50 and r.high_gap_skipped==1

def test_CA_censor_retained_for_entry_scenarios_and_no_refill():
    a=sample();a['label_valid']=False;x=classify_entries(a)
    metrics,_=evaluate_selection(x,x.rank_M1.le(3),'M1',3)
    assert metrics.evaluable_candidates.eq(0).all()
    assert metrics.selected_candidates.eq(6).all()

def test_missing_next_bar_stays_unknown_not_official_suspension():
    a=sample();a.loc[0,['next_open','next_high','next_low','next_close','next_volume']]=np.nan
    x=classify_entries(a)
    assert x.pattern_class.iloc[0]=='SUSPENDED_OR_NO_BAR'
    assert not any(x.loc[0,c] for c in ['E1_entry_eligible','E2_entry_eligible','E3_entry_eligible'])
    assert x.intraday_entry_status.iloc[0]=='INTRADAY_ENTRY_UNKNOWN'

def test_deciles_do_not_bin_across_days_or_drop_censored_before_binning():
    x=classify_entries(sample());x.loc[0,'label_valid']=False
    table,bins=decile_table(x)
    top=table.query('decile==10 and block=="ALL"').iloc[0]
    assert top.rows==2 and top.evaluable==1
    assert bins.score_decile.iloc[0]==10 and bins.score_decile.iloc[9]==1

def test_opportunity_ratio_and_contains_target_use_strict_common_universe():
    x=classify_entries(sample());x['future_max_return_2_5d']=.10
    x.loc[x.rank_M1.eq(10),'future_max_return_2_5d']=.50
    x['outcome_hit20_2_5d']=x.rank_M1.eq(10);x['outcome_hit30_2_5d']=x.rank_M1.eq(10)
    daily,summary=opportunity_table(x)
    r=summary.query('block=="ALL" and method=="M1" and k==3').iloc[0]
    assert np.isclose(r.ratio_mean,.2) and r.contains20_daily_pct==0
    assert summary.query('block=="ALL" and method=="M1" and k==10').iloc[0].contains30_daily_pct==100
