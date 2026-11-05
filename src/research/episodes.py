"""Phase2.2: causal OHLCV features and ex-post, explicitly separate event accounting."""
import numpy as np
import pandas as pd
from src.universe_policy import predictive_universe, report_audit
from src.research.target_discovery import windows, divide

NEW_FEATURES=['volatility_5','volatility_10','range_compression_5','range_compression_10',
    'volume_acceleration_3','volume_acceleration_5','price_acceleration_3','price_acceleration_5',
    'higher_low_count_5','higher_low_count_10','close_above_prev_high_count_5',
    'distance_from_5d_low','distance_from_10d_low','recovery_from_drawdown_3','recovery_from_drawdown_5',
    'body_ratio','upper_shadow_ratio','lower_shadow_ratio','consecutive_small_up_days']


def additional_features(bars, qualified_events=(), small_up_max=.015):
    """61 past market observations, adjusted with only events effective by row t."""
    if 'exchange' in bars:
        report_audit(bars)
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:raise ValueError('INDEX_ORDER')
    dates=bars.index.to_numpy(dtype='datetime64[D]')
    grid=np.lib.stride_tricks.sliding_window_view(np.r_[np.full(60,np.datetime64('NaT','D')),dates],61)
    mats={k:windows(bars[k]) for k in ('open','high','low','close','volume')}
    for ex,A,B in sorted(qualified_events):
        mask=(dates[:,None]>=np.datetime64(ex,'D'))&(grid<np.datetime64(ex,'D'))
        for k in ('open','high','low','close'):mats[k]=np.where(mask,A*mats[k]+B,mats[k])
        mats['volume']=np.where(mask,mats['volume']/A,mats['volume'])
    o,h,l,c,v=(mats[k] for k in ('open','high','low','close','volume'))
    r=divide(c[:,1:],c[:,:-1])-1
    ranges=divide(h[:,1:]-l[:,1:],c[:,:-1]);out={}
    for n in (5,10):
        out[f'volatility_{n}']=np.std(r[:,-n:],axis=1,ddof=0)
        out[f'range_compression_{n}']=divide(ranges[:,-n:].mean(axis=1),ranges[:,-2*n:-n].mean(axis=1))
        d=np.diff(l[:,-n-1:],axis=1)
        out[f'higher_low_count_{n}']=np.where(np.isfinite(d).all(axis=1),(d>0).sum(axis=1),np.nan)
        out[f'distance_from_{n}d_low']=divide(c[:,-1],l[:,-n:].min(axis=1))-1
    for n in (3,5):
        out[f'volume_acceleration_{n}']=divide(v[:,-n:].mean(axis=1),v[:,-2*n:-n].mean(axis=1))-1
        out[f'price_acceleration_{n}']=(divide(c[:,-1],c[:,-1-n])-1)-(divide(c[:,-1-n],c[:,-1-2*n])-1)
        ddnow=1-divide(c[:,-1],c[:,-20:].max(axis=1))
        ddold=1-divide(c[:,-1-n],c[:,-20-n:-n].max(axis=1))
        out[f'recovery_from_drawdown_{n}']=ddold-ddnow
    comp=c[:,-5:]-h[:,-6:-1]
    out['close_above_prev_high_count_5']=np.where(np.isfinite(comp).all(axis=1),(comp>0).sum(axis=1),np.nan)
    width=h[:,-1]-l[:,-1]
    out['body_ratio']=divide(abs(c[:,-1]-o[:,-1]),width)
    out['upper_shadow_ratio']=divide(h[:,-1]-np.maximum(o[:,-1],c[:,-1]),width)
    out['lower_shadow_ratio']=divide(np.minimum(o[:,-1],c[:,-1])-l[:,-1],width)
    small=(r>0)&(r<=small_up_max)&(c[:,1:]>o[:,1:])
    count=np.cumprod(small[:,::-1].astype(int),axis=1).sum(axis=1).astype(float)
    out['consecutive_small_up_days']=np.where(np.isfinite(r[:,-1]),count,np.nan)
    out['prior_runup_20']=divide(h[:,-21:-1].max(axis=1),l[:,-60:-20].min(axis=1))-1
    result=pd.DataFrame(out,index=bars.index).replace([np.inf,-np.inf],np.nan)
    result['prior_compression5']=result.range_compression_5.shift(1)
    vr=divide(v[:,-1],v[:,-6:-1].mean(axis=1))
    result['prior_volume_spike3']=pd.Series(vr,index=bars.index).shift(1).rolling(3,min_periods=3).max()
    return result


def new_signals(f,cfg):
    if 'exchange' in f:
        f=predictive_universe(f)
    F,G,H,I,J=(cfg[k] for k in 'FGHIJ')
    out={}
    out['F']=(f.breakout_20d==0)&(f.ret_20d>0)&(f.ret_3d>0)&(f.price_acceleration_3>F['price_acceleration_3_min'])&(f.volume_ratio_5>=F['volume_ratio_5_min'])&(f.close_vs_ma10>F['close_vs_ma10_min'])
    out['G']=(f.prior_compression5<G['prior_compression_max'])&(f.volume_ratio_5>=G['volume_ratio_5_min'])&(f.close_location>=G['close_location_min'])&(f.ret_1d>0)
    out['H']=(f.prior_volume_spike3>=H['prior_volume_spike_min'])&(f.breakout_20d==0)&(f.ret_3d>0)&(f.price_acceleration_3>0)&(f.close_location>=H['close_location_min'])
    out['I']=(f.max_drawdown_10>=I['drawdown_10_min'])&(f.recovery_from_drawdown_3>=I['recovery_3_min'])&(f.ret_3d>0)&(f.close_location>=I['close_location_min'])
    out['J']=(f.consecutive_small_up_days>=J['small_up_days_min'])&(f.close_vs_ma10>0)&(f.ma10_slope>0)&f.volume_ratio_5.between(J['volume_ratio_5_min'],J['volume_ratio_5_max'])
    return pd.DataFrame(out,index=f.index).fillna(False).astype(bool)


def archetypes(f,cfg):
    """Multi-label structural proxies across t-3..t. No claim to observe accumulation."""
    if 'exchange' in f:
        f=predictive_universe(f)
    a=cfg['archetypes']
    out={
      'EARLY_TREND_ACCELERATION':(f.ret_20d>0)&(f.price_acceleration_3>0)&(f.breakout_20d==0)&(f.breakout_60d==0),
      'DEEP_PULLBACK_REVERSAL':(f.prior_runup_20>=a['prior_runup_min'])&(f.max_drawdown_10>=a['deep_drawdown_min'])&(f.recovery_from_drawdown_3>0),
      'LOW_BASE_EXPANSION':(f.position_60d<=a['low_position_max'])&(f.ret_20d.abs()<=a['low_ret20_abs_max'])&(f.volume_ratio_5>=a['expansion_min']),
      'LIMIT_UP_IGNITION':(f.limit_up_today==1)&(f.limit_up_count_5==1),
      'GAP_IGNITION':f.gap_pct>=a['gap_min'],
      'VOLUME_FIRST_PRICE_LATER':(f.prior_volume_spike3>=a['prior_volume_min'])&(f.breakout_20d==0),
      'HIGH_MOMENTUM_CONTINUATION':(f.ret_20d>=a['high_ret20_min'])&(f.position_60d>=a['high_position_min'])&(f.price_acceleration_3>0),
      'VOLATILITY_COMPRESSION':f.range_compression_5<=a['compression_max'],
      'MULTI_DAY_STEALTH_ACCUMULATION':(f.consecutive_small_up_days>=a['stealth_small_up_min'])&(f.volume_ratio_5<=1)&(f.close_vs_ma10>0)}
    return pd.DataFrame(out,index=f.index).fillna(False).astype(bool)


def consecutive_groups(frame,mask):
    """Group only selected rows; explicit market_pos prevents merging across missing dates."""
    if frame.market_pos.isna().any() or not np.isfinite(frame.market_pos).all():
        raise ValueError('MARKET_POSITION_REQUIRED')
    x=frame.loc[mask].sort_values(['exchange','symbol','market_pos']).copy()
    if x.empty:return pd.Series(dtype='int64',index=x.index)
    changed=x.exchange.ne(x.exchange.shift())|x.symbol.ne(x.symbol.shift())|x.market_pos.diff().ne(1)
    return changed.cumsum().astype('int64')


def build_episodes(frame):
    """One common10% episode identity, threshold20/30/40 memberships as nested subsets.

    Continuous winner runs can span an extended rally: this is an operational
    clustering definition, not a claim that distinct economic events are known.
    """
    ids=consecutive_groups(frame,frame.hit10)
    membership=pd.Series(pd.NA,index=frame.index,dtype='Int64');membership.loc[ids.index]=ids
    columns=['episode_id','symbol','exchange','episode_start_date','first_detectable_date',
        'episode_end_date','start_pos','end_pos','peak_reference_date','peak_date','peak_return_5d',
        'acceleration_date','member_stock_days']
    for n in (10,20,30,40):columns.extend([f'episode_{n}',f'days_to_{n}pct',f'peak_reference_days_to_{n}pct',f'first_detectable_{n}',f'first_target_pos{n}'])
    if ids.empty:return pd.DataFrame(columns=columns),membership
    # Vectorized aggregation avoids repeatedly scanning a wide300k-row frame.
    x=frame.loc[ids.index].copy();x['_eid']=ids
    x=x.sort_values(['_eid','market_pos'])
    first=x.drop_duplicates('_eid').set_index('_eid')
    last=x.drop_duplicates('_eid',keep='last').set_index('_eid')
    best=x.sort_values(['_eid','ret_max_5d','market_pos'],ascending=[True,False,True]).drop_duplicates('_eid').set_index('_eid')
    out=pd.DataFrame(index=first.index)
    out['episode_id']=first.index
    out['symbol']=first.symbol;out['exchange']=first.exchange
    out['episode_start_date']=first.date;out['first_detectable_date']=first.date
    out['episode_end_date']=last.date;out['start_pos']=first.market_pos;out['end_pos']=last.market_pos
    out['peak_reference_date']=best.date;out['peak_date']=best.peak_date;out['peak_return_5d']=best.ret_max_5d
    out['acceleration_date']=first.first_hit_date10;out['member_stock_days']=x.groupby('_eid').size()
    for n in (10,20,30,40):
        eligible=x[x[f'hit{n}']].drop_duplicates('_eid').set_index('_eid')
        out[f'episode_{n}']=out.index.isin(eligible.index)
        out[f'days_to_{n}pct']=eligible[f'days_to_{n}pct'].reindex(out.index)
        out[f'peak_reference_days_to_{n}pct']=best[f'days_to_{n}pct']
        out[f'first_detectable_{n}']=eligible.date.reindex(out.index)
        out[f'first_target_pos{n}']=eligible[f'first_hit_pos{n}'].reindex(out.index)
    return out[columns].reset_index(drop=True),membership


def evaluate(frame,episodes,rule):
    report_audit(frame)
    """Stock-day denominator, predicted signal episodes, true episodes are distinct."""
    mask=frame[rule].astype(bool);hits=frame.hit10.astype(bool)
    groups=consecutive_groups(frame,mask)
    successful=int(frame.loc[groups.index,'hit10'].groupby(groups).any().sum()) if len(groups) else 0
    result=dict(bse_candidate_count=0,candidate_stock_days=int(mask.sum()),stock_day_winners=int((mask&hits).sum()),
        stock_day_precision=100*(mask&hits).sum()/mask.sum() if mask.sum() else np.nan,
        stock_day_capture=100*(mask&hits).sum()/hits.sum() if hits.sum() else np.nan,
        candidate_episodes=int(groups.nunique()),successful_candidate_episodes=successful,
        episode_precision=100*successful/groups.nunique() if groups.nunique() else np.nan)
    for n in (10,20,30,40):
        truth=episodes.loc[episodes[f'episode_{n}'].astype(bool)].set_index('episode_id')
        selected=frame.loc[mask&frame[f'hit{n}']&frame.episode_id.isin(truth.index)]
        detection=selected.groupby('episode_id').market_pos.min()
        lead=truth.loc[detection.index,f'first_target_pos{n}']-detection
        result[f'episodes{n}']=len(truth);result[f'caught{n}']=len(detection)
        result[f'episode_capture{n}']=100*len(detection)/len(truth) if len(truth) else np.nan
        result[f'median_lead{n}']=lead.median()
        result[f'late_detections{n}']=int((lead<=0).sum())
    return result
