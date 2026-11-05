"""Causal price/volume features, separate forward labels, and fixed research rules."""
import numpy as np
import pandas as pd
from src.universe_policy import predictive_universe, report_audit
from numpy.lib.stride_tricks import sliding_window_view

FEATURES = ([f'ret_{n}d' for n in (1,3,5,10,20)] +
    [f'close_vs_ma{n}' for n in (5,10,20)] + ['ma5_slope','ma10_slope',
    'distance_to_20d_high','distance_to_60d_high','breakout_20d','breakout_60d',
    'close_location','gap_pct','range_pct','volume_ratio_5','volume_ratio_10',
    'volume_change_1d','up_days_5','up_days_10','limit_up_today','limit_up_count_5',
    'limit_up_count_10','max_drawdown_5','max_drawdown_10'])
RULE_NAMES = {'A':'Strong Trend Pullback','B':'Breakout Confirmation',
    'C':'Volume Expansion Breakout','D':'Limit-up Follow-through (nominal-price proxy)',
    'E':'Low-position Reversal'}


def divide(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    out=np.full(a.shape,np.nan)
    np.divide(a,b,out=out,where=np.isfinite(b)&(b!=0))
    return out


def windows(values,n=61):
    return sliding_window_view(np.pad(np.asarray(values,dtype=float),(n-1,0),constant_values=np.nan),n).copy()


def features(bars, qualified_events=(), nominal_limit=.10):
    """Each row t transforms only historical window cells using events ex_date<=t.

    Events are previously verified (date,A,B), not inferred from price jumps.
    Volume is put on the same share basis (1/A for old volume). Amount is never read.
    Missing bars stay NaN; they are neither zero returns nor official suspension.
    """
    if 'exchange' in bars:
        report_audit(bars)
    dates=bars.index.to_numpy(dtype='datetime64[D]')
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError('Expected unique sorted observed-market dates')
    mats={k:windows(bars[k].to_numpy()) for k in ('open','high','low','close','volume')}
    date_grid=np.pad(dates,(60,0),constant_values=np.datetime64('NaT'))
    date_grid=sliding_window_view(date_grid,61)
    events=sorted(qualified_events,key=lambda e:e[0])
    for ex,A,B in events:
        ex=np.datetime64(ex,'D')
        mask=(dates[:,None]>=ex)&(date_grid<ex)
        for k in ('open','high','low','close'):
            mats[k]=np.where(mask,A*mats[k]+B,mats[k])
        mats['volume']=np.where(mask,mats['volume']/A,mats['volume'])
    o,h,l,c,v=(mats[k] for k in ('open','high','low','close','volume'))
    now=c[:,-1];out={}
    for n in (1,3,5,10,20):out[f'ret_{n}d']=divide(now,c[:,-1-n])-1
    for n in (5,10,20):out[f'close_vs_ma{n}']=divide(now,c[:,-n:].mean(axis=1))-1
    for n in (5,10):out[f'ma{n}_slope']=divide(c[:,-n:].mean(axis=1),c[:,-n-1:-1].mean(axis=1))-1
    for n in (20,60):
        out[f'distance_to_{n}d_high']=divide(now,np.max(h[:,-n:],axis=1))-1
        prevhigh=np.max(h[:,-n-1:-1],axis=1)
        out[f'breakout_{n}d']=np.where(np.isfinite(prevhigh)&np.isfinite(now),(now>prevhigh).astype(float),np.nan)
    out['close_location']=divide(now-l[:,-1],h[:,-1]-l[:,-1])
    out['gap_pct']=divide(o[:,-1],c[:,-2])-1
    out['range_pct']=divide(h[:,-1]-l[:,-1],c[:,-2])
    for n in (5,10):
        out[f'volume_ratio_{n}']=divide(v[:,-1],v[:,-n-1:-1].mean(axis=1))
        changes=np.diff(c[:,-n-1:],axis=1)
        out[f'up_days_{n}']=np.where(np.isfinite(changes).all(axis=1),(changes>0).sum(axis=1),np.nan)
        past=c[:,-n:];peak=np.maximum.accumulate(past,axis=1)
        out[f'max_drawdown_{n}']=np.max(1-divide(past,peak),axis=1)
    out['volume_change_1d']=divide(v[:,-1],v[:,-2])-1
    # Raw-date nominal limit price. This is explicitly a research price-pattern
    # proxy, NOT a guessed historical ST flag or an execution qualification.
    rawclose=bars.close.to_numpy(float); prev=np.r_[np.nan,rawclose[:-1]]
    for ex,A,B in events:prev=np.where(dates==np.datetime64(ex,'D'),A*prev+B,prev)
    cents=np.floor(prev*(1+nominal_limit)*100+.5)
    proxy=np.where(np.isfinite(prev)&np.isfinite(rawclose),
        ((np.rint(rawclose*100)==cents)&(bars.high.to_numpy()==rawclose)).astype(float),np.nan)
    out['limit_up_today']=proxy
    for n in (5,10):out[f'limit_up_count_{n}']=windows(proxy,n).sum(axis=1)
    min60=np.min(l[:,-60:],axis=1);max60=np.max(h[:,-60:],axis=1)
    out['position_60d']=divide(now-min60,max60-min60)
    out['ma5_above_ma10']=(c[:,-5:].mean(axis=1)>c[:,-10:].mean(axis=1))
    out['ma10_above_ma20']=(c[:,-10:].mean(axis=1)>c[:,-20:].mean(axis=1))
    result=pd.DataFrame(out,index=bars.index)
    result.replace([np.inf,-np.inf],np.nan,inplace=True)
    return result


def labels(bars, qualified_events=()):
    """Forward values are labels only; suffix unavailable bars stay censored.

    Also retains literal RAW labels. Comparable labels reverse ONLY future
    qualified transformations so highs have the same economic basis as close[t].
    """
    count=len(bars);dates=bars.index.to_numpy(dtype='datetime64[D]')
    future={k:np.column_stack([bars[k].shift(-j).to_numpy(float) for j in range(1,6)])
            for k in ('high','low','close')}
    future_dates=np.column_stack([np.r_[dates[j:],np.full(j,np.datetime64('NaT'))][:count] for j in range(1,6)])
    A=np.ones((count,5));B=np.zeros((count,5));has_event=np.zeros((count,5),dtype=bool)
    for ex,a,b in sorted(qualified_events,key=lambda e:e[0]):
        mask=(dates[:,None]<np.datetime64(ex,'D'))&(future_dates>=np.datetime64(ex,'D'))
        A=np.where(mask,a*A,A);B=np.where(mask,a*B+b,B);has_event|=mask
    fixed={k:divide(value-B,A) for k,value in future.items()}
    out={};entry=bars.close.to_numpy(float)
    for h in (1,2,3,5):
        complete=np.isfinite(future['high'][:,:h]).all(axis=1)&np.isfinite(entry)
        r=divide(np.max(fixed['high'][:,:h],axis=1),entry)-1
        raw=divide(np.max(future['high'][:,:h],axis=1),entry)-1
        out[f'ret_max_{h}d']=np.where(complete,r,np.nan)
        out[f'future_max_{h}d']=np.where(complete,np.max(fixed['high'][:,:h],axis=1),np.nan)
        out[f'raw_ret_max_{h}d']=np.where(complete,raw,np.nan)
        out[f'ret_close_{h}d']=np.where(complete,divide(fixed['close'][:,h-1],entry)-1,np.nan)
        out[f'max_adverse_{h}d']=np.where(complete,divide(np.min(fixed['low'][:,:h],axis=1),entry)-1,np.nan)
        if h!=1:
            # Integer-cent input ratios still need robust IEEE equality at10%.
            # Comparison uses cross multiplication, not an increased tolerance.
            peak=np.max(fixed['high'][:,:h],axis=1)
            raw_peak=np.max(future['high'][:,:h],axis=1)
            raw_hit=np.rint(raw_peak*100)*100 >= np.rint(entry*100)*110
            hit=np.where(has_event[:,:h].any(axis=1),peak*100>=entry*110,raw_hit)
            out[f'TARGET_{h}D_10']=np.where(complete,hit,np.nan)
    out['TARGET_2_5D_10']=out['TARGET_5D_10'].copy()
    out['label_crosses_qualified_ca']=has_event.any(axis=1)
    return pd.DataFrame(out,index=bars.index)


def signals(f, thresholds):
    if 'exchange' in f:
        f=predictive_universe(f)
    a,b,c,d,e=(thresholds[k] for k in 'ABCDE')
    result={}
    result['A']=(f.ret_20d>a['ret20_min']) & f.ma5_above_ma10 & f.ma10_above_ma20 & (f.close_vs_ma20>0) & (f.ret_1d<0) & f.close_vs_ma5.between(a['close_vs_ma5_min'],a['close_vs_ma5_max'])
    result['B']=(f.breakout_20d==1)&(f.close_location>=b['close_location_min'])&(f.ret_1d>0)
    result['C']=result['B']&(f.volume_ratio_5>=c['volume_ratio5_min'])
    result['D']=(f.limit_up_count_5>=d['limit_proxy_count5_min'])&(f.ret_20d>0)&(f.close_vs_ma5>=0)&(f.volume_ratio_5>=d['volume_ratio5_min'])
    result['E']=(f.ret_20d<e['ret20_max'])&(f.position_60d<e['position60_max'])&(f.ret_3d>0)&(f.close_location>=e['close_location_min'])&(f.volume_ratio_5>=e['volume_ratio5_min'])
    result=pd.DataFrame(result,index=f.index).fillna(False).astype(bool)
    result['UNION']=result.any(axis=1)
    return result


def event_mask(calendar, event_dates, lookback=60, forward=5):
    """Explicit research selection mask, never a feature or a deployable signal."""
    dates=pd.DatetimeIndex(calendar);out=np.zeros(len(dates),dtype=bool)
    starts=dates[np.maximum(np.arange(len(dates))-lookback,0)]
    ends=dates[np.minimum(np.arange(len(dates))+forward,len(dates)-1)]
    for ex in event_dates:
        ex=pd.Timestamp(ex);out|=(starts<=ex)&(ends>=ex)
    return out
