"""Offline episode discovery; frozen rules, temporal purge and coverage strata."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.run_phase21_research import ROOT, read_prices, sha, winner
from src.research.target_discovery import FEATURES
from src.research.episodes import (NEW_FEATURES, additional_features, new_signals,
    archetypes, consecutive_groups, build_episodes, evaluate)
from src.universe_policy import guard_cached_artifact

OUT=ROOT/'outputs/phase22';LOCAL=ROOT/'data/research/phase22'


def future_metadata(bars,events,calendar):
    """Outcome-only metadata. Not passed to feature/signal functions."""
    dates=bars.index.to_numpy(dtype='datetime64[D]');n=len(bars)
    high=np.column_stack([bars.high.shift(-j).to_numpy() for j in range(1,6)])
    future_dates=np.column_stack([np.r_[dates[j:],np.full(j,np.datetime64('NaT','D'))] for j in range(1,6)])
    A=np.ones(high.shape);B=np.zeros(high.shape);cross=np.zeros(high.shape,dtype=bool)
    for ex,a,b in sorted(events):
        mask=(dates[:,None]<np.datetime64(ex,'D'))&(future_dates>=np.datetime64(ex,'D'))
        A=np.where(mask,a*A,A);B=np.where(mask,a*B+b,B);cross|=mask
    comparable=(high-B)/A;entry=bars.close.to_numpy()
    complete=np.isfinite(high).all(axis=1)&np.isfinite(entry)
    pos=np.arange(n)-60;peak=np.argmax(np.where(np.isfinite(comparable),comparable,-np.inf),axis=1)
    out=pd.DataFrame(index=bars.index)
    out['peak_date']=pd.to_datetime(np.where(complete,future_dates[np.arange(n),peak],np.datetime64('NaT','D')))
    for k in (10,20,30,40):
        raw=np.rint(high*100)*100>=np.rint(entry[:,None]*100)*(100+k)
        hit=np.where(cross,comparable*100>=entry[:,None]*(100+k),raw)
        exists=complete&hit.any(axis=1);offset=hit.argmax(axis=1)+1
        out[f'days_to_{k}pct']=np.where(exists,offset,np.nan)
        out[f'first_hit_pos{k}']=np.where(exists,pos+offset,np.nan)
        out[f'first_hit_date{k}']=pd.to_datetime(np.where(exists,future_dates[np.arange(n),offset-1],np.datetime64('NaT','D')))
    return out


def build(config):
    guard_cached_artifact(ROOT/'data/research/phase21/stock_days.parquet')
    old=ROOT/'data/research/phase21/stock_days.parquet';before=sha(old)
    data=pd.read_parquet(old)
    inventory=pd.read_csv(ROOT/'outputs/phase21/universe_audit.csv',dtype={'symbol':str})
    ca=pd.read_csv(ROOT/'outputs/phase21/qualified_event_parameters.csv',dtype={'symbol':str})
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'outputs/phase21/observed_market_calendar.csv').date)).sort_values()[-180:]
    research=cal[-120:];parts=[];groups=data.groupby(['exchange','symbol']).groups
    for num,r in enumerate(inventory.itertuples()):
        oldpart=data.loc[groups[(r.exchange,r.symbol)]].set_index('date')
        bars,invalid,digest=read_prices(ROOT/r.day_path)
        assert invalid==r.invalid_rows and digest==r.sha256
        lower=pd.Timestamp(r.eligible_first_date);bars=bars.loc[bars.index>=lower]
        if not pd.isna(r.delisting_date):bars=bars.loc[bars.index<=pd.Timestamp(r.delisting_date)]
        bars=bars.reindex(cal)
        events=[(x.ex_date,x.A,x.B) for x in ca.itertuples() if x.symbol==r.symbol and x.exchange==r.exchange and pd.Timestamp(x.ex_date)>=lower]
        extra=additional_features(bars,events,config['feature_small_up_max'])
        full=pd.concat([oldpart.reindex(cal),extra],axis=1)
        candidates=new_signals(full,config)
        for k in 'FGHIJ':full['signal_'+k]=candidates[k]
        meta=future_metadata(bars,events,cal)
        for k in meta:full[k]=meta[k]
        full=full.loc[oldpart.index].copy()
        full['market_pos']=research.get_indexer(full.index)
        assert (full.market_pos>=0).all()
        parts.append(full.reset_index(names='date'))
        if num%1000==0:print(json.dumps({'phase22_features_securities':num}),flush=True)
    data=pd.concat(parts,ignore_index=True).sort_values(['exchange','symbol','market_pos']).reset_index(drop=True)
    assert len(data)==662319 and not data.duplicated(['exchange','symbol','date']).any()
    for n in (10,20,30,40):data[f'hit{n}']=winner(data,n)&data.clean_research
    assert np.array_equal(data.loc[data.clean_research,'hit10'],data.loc[data.clean_research,'TARGET_5D_10'].astype(bool))
    for n in (10,20,30,40):
        assert np.array_equal(data.loc[data.clean_research,f'hit{n}'],data.loc[data.clean_research,f'days_to_{n}pct'].notna())
    data['signal_OLD5']=data[['signal_'+k for k in 'ABCDE']].any(axis=1)
    data['signal_NEW10']=data[['signal_'+k for k in 'ABCDEFGHIJ']].any(axis=1)
    assert (data.signal_OLD5==data.signal_UNION).all()
    monthly=pd.read_csv(ROOT/'outputs/phase21/monthly_coverage.csv')
    monthly['coverage_status']=np.where(monthly.clean_pct_of_mature>=config['coverage_threshold_pct'],'HIGH_RESEARCH_COVERAGE','LOW_RESEARCH_COVERAGE')
    data['month']=data.date.dt.strftime('%Y-%m')
    data['coverage']=data.month.map(monthly.set_index('month').coverage_status)
    discovery=config['discovery_sessions'];purge=config['purge_discovery_sessions'];end=discovery+config['validation_sessions']
    assert end+5==len(research) and purge>=5
    data['split']=np.select([data.market_pos<discovery-purge,data.market_pos<discovery,data.market_pos<end],['DISCOVERY','PURGED_DISCOVERY','VALIDATION'],default='UNLABELED_TAIL')
    assert sha(old)==before
    changed=[r.day_path for r in inventory.itertuples() if sha(ROOT/r.day_path)!=r.sha256]
    assert not changed,'RAW_CHANGED'
    data.to_parquet(LOCAL/'stock_days.parquet',index=False)
    monthly.to_csv(OUT/'coverage_by_month.csv',index=False)
    return data,monthly,research,before


def scope_data(clean,episodes,split,coverage):
    mask=pd.Series(True,index=clean.index)
    if split!='ALL_DATES':mask&=clean.split.eq(split)
    if coverage!='ALL_COVERAGE':mask&=clean.coverage.eq(coverage)
    if split=='ALL_DATES':mask&=clean.split.isin(['DISCOVERY','VALIDATION'])
    frame=clean.loc[mask].copy()
    counts=frame[frame.episode_id.notna()].groupby('episode_id').size()
    complete=episodes.set_index('episode_id').member_stock_days
    eligible=counts.index[counts.to_numpy()==complete.reindex(counts.index).to_numpy()]
    eps=episodes[episodes.episode_id.isin(eligible)]
    boundary=frame.episode_id.notna()&~frame.episode_id.isin(eligible)
    return frame.loc[~boundary],eps,int(boundary.sum())


def performance(clean,episodes):
    rows=[]
    for split in ('ALL_DATES','DISCOVERY','VALIDATION'):
        # Never publish a mixed-coverage overall performance conclusion.
        for coverage in ('HIGH_RESEARCH_COVERAGE','LOW_RESEARCH_COVERAGE'):
            f,e,boundary=scope_data(clean,episodes,split,coverage)
            for r in list('ABCDEFGHIJ')+['OLD5','NEW10']:
                row=evaluate(f,e,'signal_'+r)
                row.update(split=split,coverage=coverage,rule=r,stock_days=len(f),boundary_winner_days_purged=boundary)
                rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(OUT/'strategy_metrics.csv',index=False)
    return result


def describe_episodes(clean,episodes):
    rows=[];detections=[];indexed=episodes.set_index('episode_id')
    for n in (10,20,30,40):
        e=episodes[episodes[f'episode_{n}']]
        rows.append(dict(threshold=n,coverage='ALL_COVERAGE_INVENTORY_ONLY',episode_count=len(e),unique_symbols=e.groupby(['exchange','symbol']).ngroups,
            median_peak_return=None,median_days_to_threshold=None))
        for cov,group in e.groupby('episode_coverage'):
            rows.append(dict(threshold=n,coverage=cov,episode_count=len(group),unique_symbols=group.groupby(['exchange','symbol']).ngroups,
                median_peak_return=group.peak_return_5d.median() if cov!='CROSS_COVERAGE_EPISODE' else None,
                median_days_to_threshold=group[f'days_to_{n}pct'].median() if cov!='CROSS_COVERAGE_EPISODE' else None))
        for rule in list('ABCDEFGHIJ')+['OLD5','NEW10']:
            selected=clean[clean['signal_'+rule]&clean[f'hit{n}']]
            first=selected.sort_values('market_pos').groupby('episode_id').first()
            for eid,r in first.iterrows():
                target=indexed.loc[eid,f'first_target_pos{n}']
                detections.append(dict(episode_id=int(eid),threshold=n,rule=rule,symbol=r.symbol,
                    first_detection_date=r.date,first_detection_lead_days=target-r.market_pos,
                    episode_coverage=indexed.loc[eid,'episode_coverage'],signal_coverage=r.coverage,split=r.split))
    pd.DataFrame(rows).to_csv(OUT/'episode_summary.csv',index=False)
    pd.DataFrame(detections).to_csv(OUT/'first_detections.csv',index=False)
    return rows


def profiles(data,clean,episodes,config,research):
    # One anchor per20% episode: earliest qualifying20% date, NOT best-return date.
    targets=clean[clean.hit20].sort_values('market_pos').groupby('episode_id',sort=False).head(1).copy()
    # Inventory includes all episodes, but precursor comparisons must not borrow
    # capture labels across discovery/validation or coverage boundaries.
    ep=episodes.set_index('episode_id')
    end_discovery=config['discovery_sessions']-config['purge_discovery_sessions']
    start_validation=config['discovery_sessions'];end_validation=start_validation+config['validation_sessions']
    ep_split=pd.Series(np.select([ep.end_pos<end_discovery,
        (ep.start_pos>=start_validation)&(ep.end_pos<end_validation)],
        ['DISCOVERY','VALIDATION'],default='BOUNDARY_OR_PURGED'),index=ep.index)
    targets['split']=targets.episode_id.map(ep_split)
    targets['coverage']=targets.episode_id.map(ep.episode_coverage)
    oldcaught=set(clean.loc[clean.hit20&clean.signal_OLD5,'episode_id'])
    targets['profile_group']=np.where(targets.episode_id.isin(oldcaught),'CAPTURED_WINNER','MISSED_WINNER')
    normals=clean[clean.ret_max_5d<.05].copy()
    normals['stable_hash']=[hashlib.sha256(f'{r.exchange}:{r.symbol}:{r.date}:phase22'.encode()).hexdigest() for r in normals.itertuples()]
    controls=[]
    # Fixed date/board matched controls, no feature/outcome threshold fitting.
    pools={k:g.sort_values('stable_hash') for k,g in normals.groupby(['date','board'])}
    for key,g in targets.groupby(['date','board']):
        if key in pools:
            chosen=pools[key].head(len(g)).copy()
            chosen['split']=g.iloc[:len(chosen)].split.to_numpy()
            chosen['coverage']=g.iloc[:len(chosen)].coverage.to_numpy()
            controls.append(chosen)
    normal=pd.concat(controls).drop(columns='stable_hash') if controls else normals.iloc[:0].drop(columns='stable_hash')
    normal['profile_group']='NORMAL';anchors=pd.concat([targets,normal],ignore_index=True)
    anchors['profile_id']=np.arange(len(anchors));lookup=data.set_index(['exchange','symbol','date'])
    fields=FEATURES+NEW_FEATURES+['prior_runup_20','prior_compression5','prior_volume_spike3','position_60d']
    snapshots=[]
    for lag in (3,2,1,0):
        dates=[research[int(p)-lag] if p>=lag else pd.NaT for p in anchors.market_pos]
        index=pd.MultiIndex.from_arrays([anchors.exchange,anchors.symbol,dates],names=['exchange','symbol','date'])
        s=lookup.reindex(index).reset_index()
        safe=s.CA_UNVERIFIED_WINDOW.eq(False)&s.CA_COVERAGE_UNKNOWN_WINDOW.eq(False)
        s.loc[~safe,fields]=np.nan
        snap=anchors[['profile_id','episode_id','profile_group','symbol','exchange','date','split','coverage']].copy()
        snap['lag']=lag;snap['feature_date']=dates;snap['feature_window_verified']=safe.to_numpy()
        for field in fields:snap[field]=s[field].to_numpy()
        snapshots.append(snap)
    snapshots=pd.concat(snapshots,ignore_index=True)
    snapshots.to_parquet(LOCAL/'precursor_profiles.parquet',index=False)
    stats=[]
    for key,g in snapshots.groupby(['split','coverage','profile_group','lag']):
        for feature in fields:
            x=g[feature].dropna()
            stats.append(dict(split=key[0],coverage=key[1],group=key[2],lag=key[3],feature=feature,
                n=len(x),mean=x.mean(),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75)))
    pd.DataFrame(stats).to_csv(OUT/'precursor_feature_statistics.csv',index=False)
    flags=archetypes(snapshots,config)
    flags['profile_id']=snapshots.profile_id.to_numpy()
    anyflags=flags.groupby('profile_id').any()
    targets=anchors[anchors.profile_group.ne('NORMAL')].copy().set_index('profile_id')
    for c in config['archetypes']['priority']:targets[c]=anyflags[c]
    targets['archetype']='UNKNOWN'
    for c in reversed(config['archetypes']['priority']):targets.loc[targets[c],'archetype']=c
    targets['archetype_is_causal_explanation']=False
    targets[['episode_id','symbol','exchange','date','split','coverage','profile_group','archetype']+config['archetypes']['priority']].to_csv(OUT/'episode20_archetypes.csv',index=False)
    return targets,stats,len(normal)


def feature_validation(clean):
    rows=[]
    for split in ('DISCOVERY','VALIDATION'):
        for coverage in ('HIGH_RESEARCH_COVERAGE','LOW_RESEARCH_COVERAGE'):
            g=clean[(clean.split==split)&(clean.coverage==coverage)]
            for name in FEATURES+NEW_FEATURES:
                x=g[name].dropna();y=g.loc[x.index,'hit10'];n1=int(y.sum());n0=len(y)-n1
                auc=(x.rank().loc[y].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
                rows.append(dict(split=split,coverage=coverage,feature=name,n=len(x),winners=n1,auc=auc,signed_rank_biserial=2*auc-1))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'feature_time_validation.csv',index=False)
    return frame


def main():
    guard_cached_artifact(ROOT/'data/research/phase21/stock_days.parquet')
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'config/phase22_research.json').read_text())
    assert not config['amount_enabled']
    frozen=sha(ROOT/'config/phase22_research.json');gate=sha(ROOT/'scripts/phase2_readiness.py')
    import sys
    if '--reuse-built' in sys.argv:
        checkpoint=ROOT/'outputs/phase22_research_health.json'
        previous=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.is_file() else None
        if previous:assert previous['config_sha256']==frozen
        parent_hash=sha(ROOT/'data/research/phase21/stock_days.parquet')
        if previous:assert parent_hash==previous['parent_phase21_sha256']
        data=pd.read_parquet(LOCAL/'stock_days.parquet')
        parent=pd.read_parquet(ROOT/'data/research/phase21/stock_days.parquet')
        keys=['exchange','symbol','date']
        pd.testing.assert_frame_equal(data[parent.columns].sort_values(keys).reset_index(drop=True),
            parent.sort_values(keys).reset_index(drop=True),check_dtype=False)
        monthly=pd.read_csv(OUT/'coverage_by_month.csv')
        research=pd.DatetimeIndex(sorted(data.date.unique()))
        assert len(research)==120 and len(data)==len(parent)
    else:
        data,monthly,research,parent_hash=build(config)
    clean=data[data.clean_research].copy()
    episodes,ids=build_episodes(clean);clean['episode_id']=ids
    coverage_group=clean[clean.episode_id.notna()].groupby('episode_id').coverage
    count=coverage_group.nunique();first=coverage_group.first()
    episodes['episode_coverage']=episodes.episode_id.map(first.where(count==1,'CROSS_COVERAGE_EPISODE'))
    # Global identity remains fixed across all strategy comparisons and splits.
    data['episode_id']=ids.reindex(data.index)
    episodes.to_csv(OUT/'winner_episodes.csv',index=False)
    clean[['symbol','exchange','date','episode_id','hit10','hit20','hit30','hit40']].to_parquet(LOCAL/'episode_membership.parquet',index=False)
    metrics=performance(clean,episodes)
    summary=describe_episodes(clean,episodes)
    archetype_rows,_,normals=profiles(data,clean,episodes,config,research)
    feature_validation(clean)
    assert sha(ROOT/'scripts/phase2_readiness.py')==gate and sha(ROOT/'config/phase22_research.json')==frozen
    health=dict(status='PENDING_FINAL_VERIFICATION',scope='RESEARCH_ONLY_NOT_TRADABLE_BACKTEST',
        run_time=datetime.now(timezone.utc).isoformat(),config_sha256=frozen,parent_phase21_sha256=parent_hash,
        research_days=120,discovery_assigned_days=80,discovery_evaluated_days=75,validation_days=35,
        discovery_start=str(research[0].date()),discovery_end=str(research[79].date()),
        validation_start=str(research[80].date()),validation_end=str(research[114].date()),
        raw_stock_days=len(data),clean_stock_days=len(clean),episode_summary=summary,
        control_profiles=normals,missed_archetypes=archetype_rows[archetype_rows.profile_group=='MISSED_WINNER'].archetype.value_counts().to_dict(),
        validation_high_coverage_months=0,validation_representative=False,
        prior_phase21_outcomes_already_seen=True,pristine_holdout=False,
        production_gate_unchanged=True,RAW_modified=False,amount_used=False,model_trained=False,
        commit_created=False,pushed=False)
    (ROOT/'outputs/phase22_research_health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(health,ensure_ascii=False,indent=2,default=str),flush=True)


if __name__=='__main__':main()
