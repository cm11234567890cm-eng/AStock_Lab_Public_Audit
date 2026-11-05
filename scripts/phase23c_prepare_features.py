"""Prepare immutable, announcement-free feature values and intrinsic masks."""
import json
import numpy as np
import pandas as pd
from scripts.run_phase21_research import ROOT,read_prices,sha
from scripts.run_phase23_research import dump
from src.research.coverage_recovery import FEATURE_WINDOWS
from src.research.label_censoring import price_volume_values,intrinsic_validity,fixed_signals
from src.universe_policy import guard_cached_artifact

OUT=ROOT/'outputs/phase23c';LOCAL=ROOT/'data/research/phase23c'

def main():
    guard_cached_artifact(ROOT/'data/research/phase22/stock_days.parquet')
    LOCAL.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/phase22_research.json').read_text())
    oldcfg=json.loads((ROOT/'config/phase21_research.json').read_text())
    p22=pd.read_parquet(ROOT/'data/research/phase22/stock_days.parquet')
    cols=['exchange','symbol','date']+[n for n in FEATURE_WINDOWS if n!='consecutive_small_up_days_10']
    raw=p22[cols].copy();del p22
    # Full streak is capped at 60 by the frozen expression; min(run,10) is
    # mathematically identical to the explicit ten-session implementation.
    raw['consecutive_small_up_days_10']=raw.consecutive_small_up_days.clip(upper=10)
    inv=pd.read_csv(ROOT/'outputs/phase21/universe_audit.csv',dtype={'symbol':str})
    qualified=pd.read_csv(ROOT/'outputs/phase21/qualified_event_parameters.csv',dtype={'symbol':str})
    adjusted=set(zip(qualified.exchange,qualified.symbol))
    groups=raw.groupby(['exchange','symbol']).groups
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'outputs/phase21/observed_market_calendar.csv').date)).sort_values()[-180:]
    masks=[];checks=[];source=[]
    for num,r in enumerate(inv.itertuples(),1):
        key=(r.exchange,r.symbol);idx=groups[key];p=raw.loc[idx].set_index('date')
        bars,invalid,digest=read_prices(ROOT/r.day_path)
        assert digest==r.sha256 and invalid==r.invalid_rows
        bars=bars.loc[bars.index>=pd.Timestamp(r.eligible_first_date)]
        if pd.notna(r.delisting_date):bars=bars.loc[bars.index<=pd.Timestamp(r.delisting_date)]
        bars=bars.reindex(cal)
        ratio={'SSE_MAIN':.10,'SZSE_MAIN':.10,'STAR':.20,'CHINEXT':.20,'BSE':.30}[r.board]
        # Only the 16 stocks with qualified CA adjustments need a full numeric
        # rebuild; all other cached Phase22 calculations used an empty A/B list.
        if key in adjusted or num<=25:
            computed=price_volume_values(bars,ratio,cfg['feature_small_up_max']).loc[p.index]
            if key not in adjusted:
                for n in FEATURE_WINDOWS:np.testing.assert_allclose(computed[n].astype(float),p[n].astype(float),equal_nan=True)
            for n in FEATURE_WINDOWS:
                raw.loc[idx,n]=computed[n].to_numpy();p[n]=computed[n]
            checks.append(dict(exchange=r.exchange,symbol=r.symbol,recomputed=True,qualified_adjustments_removed=key in adjusted,
                raw_cache_replay_match=True if key not in adjusted else None))
        vals=p.reindex(cal)
        valid=intrinsic_validity(bars,vals).loc[p.index].add_suffix('__valid')
        valid.index=idx;masks.append(valid)
        source.append(dict(exchange=r.exchange,symbol=r.symbol,path=r.day_path,sha256=digest))
        if num%1000==0:print(json.dumps(dict(feature_intrinsic_securities=num)),flush=True)
    mask=pd.concat(masks).sort_index()
    assert raw.index.equals(mask.index)
    # Numeric booleans are cast in fixed_signals, never inferred from NaN.
    rule=fixed_signals(raw,oldcfg['rule_thresholds'],cfg)
    for r in 'ABCDEFGHIJ':raw['signal_'+r]=rule[r]
    raw.to_parquet(LOCAL/'ohlcv_feature_values.parquet',index=False)
    pd.concat([raw[['exchange','symbol','date']],mask],axis=1).to_parquet(LOCAL/'intrinsic_validity.parquet',index=False)
    pd.DataFrame(checks).to_csv(OUT/'raw_feature_replays.csv',index=False)
    dump(OUT/'price_source_hashes.json',source)
    dump(OUT/'feature_preparation_health.json',dict(rows=len(raw),securities=len(inv),qualified_parameter_stocks_recomputed=len(adjusted),
        independent_unadjusted_replays=sum(x['qualified_adjustments_removed'] is False for x in checks),
        feature_uses_CA_parameters=False,feature_uses_CA_publication=False,amount_used=False,
        numeric_values_sha256=sha(LOCAL/'ohlcv_feature_values.parquet')))
    print('OHLCV_FEATURES_PREPARED',len(raw),flush=True)

if __name__=='__main__':main()
