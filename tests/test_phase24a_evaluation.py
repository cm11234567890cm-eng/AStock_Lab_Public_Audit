"""Independent denominator and no-ex-post-refill edge cases."""
import numpy as np
import pandas as pd
import pytest
from scripts.run_phase24a_ranking import evaluate_top,opportunity,fit_strategy_weights

def tables():
    dates=pd.DatetimeIndex(['2026-05-04','2026-05-05'])
    ranked=pd.DataFrame(dict(exchange=['SSE']*6,symbol=['600000','600001','600002']*2,
        date=np.repeat(dates,3),split='DISCOVERY',method='R1',rank=[1,2,3]*2))
    truth=ranked[['exchange','symbol','date']].copy();truth['label_valid']=True
    for h in ('2_5d','1_5d'):
        for n in (10,20,30):truth[f'outcome_hit{n}_{h}']=[True,True,False,True,False,False]
    return ranked,truth,dates

def test_stock_day_hit_and_daily_success_have_distinct_denominators():
    r,t,dates=tables();m,d,_=evaluate_top(r,t,{'DISCOVERY':dates})
    row=m[(m.k==3)&(m.threshold==10)&(m.horizon=='2_5d')].iloc[0]
    assert row.hit_slots==3 and row.evaluable_slots==6 and row.hit_rate_pct==50
    assert row.successful_days==2 and row.daily_success_lower_pct==100

def test_invalid_top_slot_is_unknown_and_never_replaced_by_lower_rank():
    r,t,dates=tables();t.loc[[0,3],'label_valid']=False
    m,d,top=evaluate_top(r,t,{'DISCOVERY':dates})
    row=m[(m.k==1)&(m.threshold==10)&(m.horizon=='2_5d')].iloc[0]
    assert row.selected_slots==2 and row.evaluable_slots==0 and row.hit_slots==0
    assert np.isnan(row.hit_rate_pct)
    assert row.unresolved_days==2 and row.daily_success_lower_pct==0 and row.daily_success_upper_pct==100
    assert len(top[top['rank']==1])==2

def test_market_maximum_without_feature_eligibility_remains_unranked():
    dates=pd.DatetimeIndex(['2026-05-04','2026-05-05','2026-05-06'])
    keys=pd.DataFrame(dict(exchange=['SSE']*3,symbol=['600000','600001','600002'],date=dates))
    truth=keys.assign(label_valid=True,outcome_max_return_1_5d=[.5,.4,.3])
    decisions=keys.assign(tie_key=['a','b','c'],candidate=[False,True,True],full_any_strategy_eligible=[False,True,True])
    rank=keys.iloc[1:].assign(rank=[20,2],tie_best_rank=[20,2],tie_worst_rank=[20,2],method='R2')
    summary,primary,_=opportunity(rank,truth,decisions,dates)
    assert summary['days']==3 and summary['unranked_days']==1
    assert summary['winner_rank_median']==20 and summary['winner_rank_p75']=='UNRANKED'
    assert summary['winner_in_top10_pct']==pytest.approx(100/3)
    assert summary['winner_in_top20_pct']==pytest.approx(200/3)
    assert len(primary)==3

def test_strategy_weight_fit_rejects_validation_before_reading_outcomes():
    with pytest.raises(AssertionError):fit_strategy_weights(pd.DataFrame({'split':['DISCOVERY','VALIDATION']}))
