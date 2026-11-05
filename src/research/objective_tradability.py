"""Read-only OOF objective and daily-bar entry-price sensitivity analysis.

No model, training, signal or feature generation is imported here. Actual
continuous-auction fills cannot be established by daily OHLCV observations.
"""
import numpy as np
import pandas as pd
from src.universe_policy import report_audit, predictive_mask

KEYS=['exchange','symbol','date']
METHODS=('M1','R3_ALL')
SCENARIOS=('CLOSE','E1','E2','E3')

def cents(x):
    return np.rint(np.nan_to_num(np.asarray(x,float),nan=0.)*100).astype(np.int64)

def classify_entries(frame):
    out=frame.copy()
    op=cents(out.next_open);cl=cents(out.close_t)
    present=out.next_open.notna()&out.next_open.gt(0)
    out['next_day_gap']=out.next_open/out.close_t-1
    out['gap_ge5']=present&(op*100>=cl*105)
    # The bucket boundaries use exact source cents, including exactly 8%.
    out['gap_bucket']=np.select([
        ~present,op<cl,op*100<cl*103,op*100<cl*105,op*100<=cl*108],
        ['SUSPENDED_OR_NO_BAR','NORMAL_ENTRY','GAP_0_3','GAP_3_5','GAP_5_8'],default='GAP_GT_8')
    nominal=(cl*(100+np.rint(out.nominal_limit_ratio.to_numpy()*100).astype('int64'))+50)//100
    out['nominal_limit_up_open_proxy']=present&(op==nominal)
    complete=out[['next_open','next_high','next_low','next_close']].notna().all(axis=1)
    one=complete&out.next_open.eq(out.next_high)&out.next_open.eq(out.next_low)&out.next_open.eq(out.next_close)
    out['one_price_day']=one
    out['one_price_up']=one&(op>cl)
    out['pattern_class']=np.select([~present,one&out.nominal_limit_up_open_proxy,
        out.nominal_limit_up_open_proxy,one&out.one_price_up,one],
        ['SUSPENDED_OR_NO_BAR','ONE_PRICE_LIMIT_UP','LIMIT_UP_OPEN','ONE_PRICE_UP_UNVERIFIED_LIMIT','ONE_PRICE_OTHER'],default='NORMAL_ENTRY')
    out['pattern_basis']='DAILY_OHLC_NOMINAL_BOARD_PROXY_NOT_OFFICIAL_LIMIT_STATUS'
    out['official_limit_up_status']='UNKNOWN_HISTORICAL_ST_AND_SPECIAL_LIMIT_NOT_QUALIFIED'
    out['intraday_entry_status']='INTRADAY_ENTRY_UNKNOWN'
    out['normal_continuous_auction_entry_confirmed']=pd.array([pd.NA]*len(out),dtype='boolean')
    out['opened_below_nominal_limit_during_day_proxy']=out.nominal_limit_up_open_proxy&out.next_low.lt(out.next_open)
    out['open_observed']=present
    out['E1_entry_eligible']=present
    out['E2_entry_eligible']=present&~out.gap_ge5
    # E3 is explicitly an EX-POST exclusion sensitivity. Today's full high/low
    # and volume are not known at the opening; this is not a causal fill model.
    out['E3_entry_eligible']=present&complete&~one&~out.nominal_limit_up_open_proxy&out.next_volume.gt(0)
    out['E1_status']=np.where(present,'OPEN_PRICE_BENCHMARK','SUSPENDED_OR_NO_BAR')
    out['E2_status']=np.select([~present,out.gap_ge5],['SUSPENDED_OR_NO_BAR','SKIPPED_HIGH_GAP'],default='NO_CHASE_OPEN_PRICE_BENCHMARK')
    out['E3_status']=np.where(out.E3_entry_eligible,'NONFLAGGED_DAILY_BAR_PROXY','INTRADAY_ENTRY_UNKNOWN_EXCLUDED')
    out.loc[~present,'E3_status']='SUSPENDED_OR_NO_BAR'
    peaks=out[[f'future_high_{j}' for j in (2,3,4,5)]].max(axis=1,skipna=False)
    out['future_max_return_2_5d']=peaks/out.close_t-1
    out['max_return_after_entry_2_5d']=peaks/out.next_open-1
    for n in (10,20,30):
        hit=cents(peaks)*100>=op*(100+n)
        out[f'entry_hit{n}']=pd.array(np.where(out.future2_5_complete&present,hit,None),dtype='boolean')
    return out

def scenario_masks(frame,scenario,threshold=10):
    if scenario=='CLOSE':
        allowed=pd.Series(True,index=frame.index)
        target=frame[f'outcome_hit{threshold}_2_5d']
    else:
        allowed=frame[scenario+'_entry_eligible'].astype(bool)
        target=frame[f'entry_hit{threshold}']
    known=allowed&frame.label_valid&target.notna()
    hits=known&target.fillna(False).astype(bool)
    return allowed,known,hits

def evaluate_selection(frame,selected,method,k,selection_name='FIXED_TOPK'):
    report_audit(frame)
    rows=[];daily=[]
    for block in ['ALL']+list(frame.block.unique()):
        scope=pd.Series(True,index=frame.index) if block=='ALL' else frame.block.eq(block)
        dates=pd.DatetimeIndex(sorted(frame.loc[scope,'date'].unique()))
        p=frame.loc[scope&selected]
        for scenario in SCENARIOS:
            for n in (10,20,30):
                allowed,known,hits=scenario_masks(p,scenario,n)
                day=pd.DataFrame(dict(date=p.date,eligible=allowed,evaluable=known,hits=hits,unknown=allowed&~known))
                day=day.groupby('date').sum().reindex(dates,fill_value=0)
                success=day.hits.gt(0);unresolved=~success&day.unknown.gt(0)
                rows.append(dict(bse_candidate_count=0,method=method,selection=selection_name,block=block,k=k,scenario=scenario,threshold=n,
                    selected_candidates=len(p),entry_eligible_candidates=int(allowed.sum()),evaluable_candidates=int(known.sum()),
                    label_censored_candidates=int((allowed&~known).sum()),hit_candidates=int(hits.sum()),
                    hit_rate_pct=100*hits.sum()/known.sum() if known.any() else np.nan,
                    entry_participation_pct=100*allowed.sum()/len(p) if len(p) else 0,
                    high_gap_skipped=int(p.gap_ge5.sum()) if scenario=='E2' else 0,
                    execution_unknown_excluded=int((~allowed&p.open_observed).sum()) if scenario=='E3' else 0,
                    no_bar_count=int((~p.open_observed).sum()) if scenario!='CLOSE' else 0,
                    days=len(dates),days_with_entry_candidates=int(day.eligible.gt(0).sum()),successful_days=int(success.sum()),
                    unresolved_label_days=int(unresolved.sum()),daily_success_proxy_pct=100*success.mean(),
                    daily_label_upper_proxy_pct=100*(success|unresolved).mean(),
                    actual_continuous_auction_success='NOT_IDENTIFIABLE_WITH_DAILY_BARS'))
                if block=='ALL':
                    for date,d in day.iterrows():daily.append(dict(method=method,selection=selection_name,k=k,scenario=scenario,
                        threshold=n,date=date,entry_eligible=int(d.eligible),evaluable=int(d.evaluable),hits=int(d.hits),unknown=int(d.unknown)))
    return pd.DataFrame(rows),pd.DataFrame(daily)

def fallback_open_only(frame,method='M1',k=3):
    """Opening-information-only scan. Never reads high/low, labels or returns.

    This may retain a subsequently observed one-price day: dropping it at the
    opening would require future knowledge. Actual fill remains unverified.
    """
    op=cents(frame.next_open);cl=cents(frame.close_t)
    nominal=(cl*(100+np.rint(frame.nominal_limit_ratio.to_numpy()*100).astype('int64'))+50)//100
    allowed=predictive_mask(frame)&frame.next_open.notna()&frame.next_open.gt(0)&(op*100<cl*105)&(op!=nominal)
    ordered=frame.loc[allowed].sort_values(['date','rank_'+method],kind='stable')
    picked=ordered.groupby('date',sort=False).head(k)
    result=pd.Series(False,index=frame.index);result.loc[picked.index]=True
    return result

def decile_table(frame):
    report_audit(frame)
    # Per-day cross-sectional deciles. Label invalidity does not change bins.
    size=frame.groupby('date').date.transform('size')
    decile=10-np.floor((frame.rank_M1-1)*10/size).astype(int)
    work=frame.assign(score_decile=decile)
    rows=[]
    for block in ['ALL']+list(work.block.unique()):
        scoped=work if block=='ALL' else work[work.block.eq(block)]
        for d,p in scoped.groupby('score_decile'):
            q=p[p.label_valid&p.future_max_return_2_5d.notna()]
            row=dict(block=block,decile=int(d),rows=len(p),evaluable=len(q),days=p.date.nunique(),
                     mean_future_max_return=float(q.future_max_return_2_5d.mean()),median_future_max_return=float(q.future_max_return_2_5d.median()),
                     score_min=float(p.score_M1.min()),score_max=float(p.score_M1.max()))
            for n in (10,20,30):row[f'hit{n}_pct']=100*q[f'outcome_hit{n}_2_5d'].astype(bool).mean()
            rows.append(row)
    return pd.DataFrame(rows),work[KEYS+['score_decile']]

def opportunity_table(frame):
    """Compare actual frozen top-K to market maximum, all on strict 2..5 target.

    Invalid labels are excluded only after top-K selection. Nonpositive market
    denominators have no meaningful upside ratio and remain explicitly missing.
    """
    report_audit(frame)
    rows=[]
    legal=frame[frame.label_valid&frame.future_max_return_2_5d.notna()]
    for date,market in legal.groupby('date'):
        peak=float(market.future_max_return_2_5d.max())
        for method in METHODS:
            for k in (3,5,10,20):
                p=market[market['rank_'+method].le(k)]
                best=float(p.future_max_return_2_5d.max()) if len(p) else np.nan
                for_unknown=frame[frame.date.eq(date)&frame['rank_'+method].le(k)]
                row=dict(date=date,block=market.block.iloc[0],method=method,k=k,evaluable=len(p),
                    selected=len(for_unknown),unknown=int((~for_unknown.label_valid).sum()),
                    best_return_in_topK=best,market_best_return=peak,
                    opportunity_capture_ratio=best/peak if peak>0 and np.isfinite(best) else np.nan)
                for n in (20,30):row[f'contains_any_{n}pct_winner']=bool(p[f'outcome_hit{n}_2_5d'].fillna(False).any())
                rows.append(row)
    daily=pd.DataFrame(rows);summary=[]
    for block in ['ALL']+list(daily.block.unique()):
        d=daily if block=='ALL' else daily[daily.block.eq(block)]
        for (method,k),g in d.groupby(['method','k']):
            summary.append(dict(block=block,method=method,k=int(k),days=len(g),ratio_evaluable_days=int(g.opportunity_capture_ratio.notna().sum()),
                ratio_mean=float(g.opportunity_capture_ratio.mean()),ratio_median=float(g.opportunity_capture_ratio.median()),
                ratio_p25=float(g.opportunity_capture_ratio.quantile(.25)),ratio_p75=float(g.opportunity_capture_ratio.quantile(.75)),
                mean_best_return=float(g.best_return_in_topK.mean()),median_best_return=float(g.best_return_in_topK.median()),
                contains20_daily_pct=100*g.contains_any_20pct_winner.mean(),contains30_daily_pct=100*g.contains_any_30pct_winner.mean()))
    return daily,pd.DataFrame(summary)
