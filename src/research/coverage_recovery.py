"""Phase23 research-only validity. No tuning, production gates or order interfaces."""
import numpy as np
import pandas as pd

# Inclusive oldest/newest input offsets in market sessions. A return over N
# sessions needs N+1 prices. These are actual expressions, not name suffixes.
FEATURE_WINDOWS = {}
def register(names, oldest, newest=0):
    for name in names.split(): FEATURE_WINDOWS[name] = (oldest, newest)
for n in (1,3,5,10,20): register(f'ret_{n}d',n)
for n in (5,10,20): register(f'close_vs_ma{n}',n-1)
for n in (5,10):
    register(f'ma{n}_slope volume_ratio_{n} up_days_{n} limit_up_count_{n} volatility_{n} higher_low_count_{n}',n)
    register(f'max_drawdown_{n} distance_from_{n}d_low',n-1)
    register(f'range_compression_{n}',2*n)
for n in (20,60):
    register(f'distance_to_{n}d_high',n-1)
    register(f'breakout_{n}d',n)
register('close_location body_ratio upper_shadow_ratio lower_shadow_ratio',0)
register('gap_pct range_pct volume_change_1d limit_up_today',1)
register('position_60d',59)
register('ma5_above_ma10',9)
register('ma10_above_ma20',19)
for n in (3,5):
    register(f'volume_acceleration_{n}',2*n-1)
    register(f'price_acceleration_{n}',2*n)
    register(f'recovery_from_drawdown_{n}',19+n)
register('close_above_prev_high_count_5',5)
register('consecutive_small_up_days',60)
register('consecutive_small_up_days_10',10)
register('prior_runup_20',59,1)
register('prior_compression5',11,1)
register('prior_volume_spike3',8,1)
CORE_SHORT_FEATURE_SET = tuple(k for k,(a,b) in FEATURE_WINDOWS.items() if a<=10)

RULE_DEPENDENCIES = {
 'A':'ret_20d ma5_above_ma10 ma10_above_ma20 close_vs_ma20 ret_1d close_vs_ma5',
 'B':'breakout_20d close_location ret_1d',
 'C':'breakout_20d close_location ret_1d volume_ratio_5',
 'D':'limit_up_count_5 ret_20d close_vs_ma5 volume_ratio_5',
 'E':'ret_20d position_60d ret_3d close_location volume_ratio_5',
 'F':'breakout_20d ret_20d ret_3d price_acceleration_3 volume_ratio_5 close_vs_ma10',
 'G':'prior_compression5 volume_ratio_5 close_location ret_1d',
 'H':'prior_volume_spike3 breakout_20d ret_3d price_acceleration_3 close_location',
 'I':'max_drawdown_10 recovery_from_drawdown_3 ret_3d close_location',
 # min(full consecutive run,10)>=3 iff full run>=3. Threshold is unchanged.
 'J':'consecutive_small_up_days_10 close_vs_ma10 ma10_slope volume_ratio_5',
}
RULE_DEPENDENCIES={k:v.split() for k,v in RULE_DEPENDENCIES.items()}

def interval_flags(calendar, bad_events, unknown_intervals, oldest=0, newest=0, forward=0):
    """Events cross a price basis only for start < ex_date <= end.

    No feature mask examines future dates. Forward is exclusively label QA.
    Unknown intervals describe source completeness, never invented CA events.
    """
    dates=pd.DatetimeIndex(calendar).to_numpy(dtype='datetime64[ns]'); pos=np.arange(len(dates))
    start=dates[np.maximum(pos-oldest,0)]
    end=dates[np.minimum(np.maximum(pos-newest+forward,0),len(dates)-1)]
    event=np.zeros(len(dates),bool); unknown=event.copy()
    if len(bad_events):
        ex=pd.DatetimeIndex(bad_events).to_numpy(dtype='datetime64[ns]')
        event=((start[:,None]<ex)&(ex<=end[:,None])).any(axis=1)
    for a,b in unknown_intervals:
        unknown|=(start<np.datetime64(pd.Timestamp(b),'ns'))&(end>=np.datetime64(pd.Timestamp(a),'ns'))&(start<end)
    return event,unknown

def feature_masks(bars, values, bad_events, unknown_intervals):
    masks={};ca={};unknown={};missing={}
    # Values already contain exact-expression NaNs. Explicit close-history checks
    # also stop bool comparisons/streak counts from silently treating NaN as false.
    close=bars.close;window_cache={}
    for name,(old,new) in FEATURE_WINDOWS.items():
        if (old,new) not in window_cache:
            e,u=interval_flags(bars.index,bad_events,unknown_intervals,old,new)
            history=close.shift(new).rolling(old-new+1,min_periods=old-new+1).count().eq(old-new+1).to_numpy()
            window_cache[(old,new)]=(e,u,history)
        e,u,history=window_cache[(old,new)]
        finite=np.isfinite(values[name].to_numpy(dtype=float))
        m=~(history&finite)
        masks[name]=~e&~u&~m;ca[name]=e;unknown[name]=u;missing[name]=m
    return tuple(pd.DataFrame(x,index=bars.index) for x in (masks,ca,unknown,missing))

def bounded_streak(bars, events=(), small_up_max=.015):
    from src.research.target_discovery import windows,divide
    dates=bars.index.to_numpy(dtype='datetime64[D]')
    grid=np.lib.stride_tricks.sliding_window_view(np.r_[np.full(10,np.datetime64('NaT','D')),dates],11)
    c=windows(bars.close,11);o=windows(bars.open,11)
    for ex,a,b in sorted(events):
        mask=(dates[:,None]>=np.datetime64(ex,'D'))&(grid<np.datetime64(ex,'D'))
        c=np.where(mask,a*c+b,c);o=np.where(mask,a*o+b,o)
    r=divide(c[:,1:],c[:,:-1])-1
    small=(r>0)&(r<=small_up_max)&(c[:,1:]>o[:,1:])
    count=np.cumprod(small[:,::-1],axis=1).sum(axis=1).astype(float)
    return pd.Series(np.where(np.isfinite(r[:,-1]),count,np.nan),index=bars.index)

def eligibility(validity, version):
    out={}
    for rule,deps in RULE_DEPENDENCIES.items():
        permitted=version=='FULL_FEATURE_RESEARCH' or all(d in CORE_SHORT_FEATURE_SET for d in deps)
        out[rule]=validity[deps].all(axis=1)&permitted
    return pd.DataFrame(out,index=validity.index)

def exclusion_strings(flags):
    """All reasons overlap; primary is deterministic and partitions the rows."""
    values=np.full(len(flags),'',object);primary=np.full(len(flags),'NONE',object)
    for reason in flags:
        mask=flags[reason].to_numpy(bool)
        primary[(primary=='NONE')&mask]=reason
        values[mask]=np.where(values[mask]=='',reason,values[mask]+';'+reason)
    return pd.DataFrame({'exclusion_reasons':values,'primary_exclusion_reason':primary},index=flags.index)
