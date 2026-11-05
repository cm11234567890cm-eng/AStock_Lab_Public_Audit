"""Offline Phase23 coverage recovery. Only writes new Phase23 artifacts."""
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd

from scripts.run_phase21_research import ROOT, read_prices, sha, winner
from scripts.run_phase22_research import scope_data
from src.research.target_discovery import features, signals
from src.research.episodes import additional_features, new_signals, build_episodes, evaluate, archetypes
from src.research.coverage_recovery import (FEATURE_WINDOWS, CORE_SHORT_FEATURE_SET,
    RULE_DEPENDENCIES, interval_flags, feature_masks, bounded_streak, eligibility, exclusion_strings)
from src.universe_policy import guard_cached_artifact, report_audit

OUT=ROOT/'outputs/phase23'; LOCAL=ROOT/'data/research/phase23'
VERSIONS=('FULL_FEATURE_RESEARCH','CORE_SHORT_RESEARCH')

def dump(path,obj):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        if isinstance(v,(np.integer,np.bool_)):return v.item()
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def source_inventory():
    events=defaultdict(set);members=set();full=set();gap=set();sources=[]
    def record(p):sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    p=ROOT/'data/cache/tdx_gbbq_parsed.csv';record(p)
    for r in pd.read_csv(p,dtype={'code':str}).itertuples():
        if r.market not in (0,1,2):continue
        key=({0:'SZSE',1:'SSE',2:'BSE'}[r.market],r.code.zfill(6));members.add(key)
        if r.category in (1,4,11,12,13,14,15):events[key].add(pd.Timestamp(str(int(r.datetime))))
    for p in sorted((ROOT/'data/tdxaidata_raw/gap_corporate_actions').glob('*.json')):
        record(p);o=json.loads(p.read_text(encoding='utf-8'))
        if str(o.get('ErrorId'))!='0' or o.get('schema_valid') is not True:continue
        if (o.get('requested_start'),o.get('requested_end'))!=('2026-08-25','2026-09-30'):continue
        key=(o['exchange'],o['symbol']);gap.add(key)
        events[key].update(pd.Timestamp(d) for d in o['Date'])
    p=ROOT/'outputs/tdxaidata_100_sample.csv';record(p)
    for r in pd.read_csv(p,dtype={'symbol':str}).itertuples():
        p=ROOT/f'data/tdxaidata_raw/probe100/{r.symbol}.divid.native.json';record(p)
        o=json.loads(p.read_text(encoding='utf-8'))
        assert str(o['ErrorId'])=='0' and len(o['Date'])==len(o['Type'])==len(o['Value'])
        key=(r.exchange,r.symbol);full.add(key);events[key].update(pd.Timestamp(d) for d in o['Date'])
    p=ROOT/'outputs/phase21/qualified_event_parameters.csv';record(p)
    qualified=defaultdict(list)
    for r in pd.read_csv(p,dtype={'symbol':str}).itertuples():
        qualified[(r.exchange,r.symbol)].append((r.ex_date,r.A,r.B))
        events[(r.exchange,r.symbol)].add(pd.Timestamp(r.ex_date))
    for p in (ROOT/'data/test_fixtures/rights_validation').glob('*.json'):
        record(p);o=json.loads(p.read_text());events[(o['exchange'],o['symbol'])].add(pd.Timestamp(o['ex_date']))
    return events,qualified,members,full,gap,sources

def unknown_ranges(key,first,members,full,gap):
    if key in full:return []
    ranges=[]
    if key not in members:ranges.append(('1900-01-01','2026-08-18'))
    ranges.append(('2026-08-19','2026-08-24'))
    if key not in gap:ranges.append(('2026-08-25','2026-09-30'))
    return [(max(pd.Timestamp(a),first),pd.Timestamp(b)) for a,b in ranges if first<=pd.Timestamp(b)]

def attrition(data):
    flags=pd.DataFrame(index=data.index)
    flags['CA_UNQUALIFIED_EVENT_OLD_WINDOW']=data.CA_UNVERIFIED_WINDOW
    flags['CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW']=data.CA_COVERAGE_UNKNOWN_WINDOW
    flags['LABEL_DATASET_TAIL']=data.market_pos>=115
    flags['MISSING_FUTURE_BAR']=~data.future5_complete&~flags.LABEL_DATASET_TAIL
    # This is the exact legacy observed-stock-day denominator. ST is deliberately
    # unknown historically and allowed by the frozen research contract.
    flags['ST_RULE']=False;flags['LISTING_BOUNDARY']=False;flags['IDENTITY']=False
    strings=exclusion_strings(flags)
    ledger=pd.concat([data[['exchange','symbol','date','month','future5_complete','clean_research']],flags,strings],axis=1)
    assert np.array_equal(ledger.primary_exclusion_reason.eq('NONE'),data.clean_research)
    ledger.to_parquet(LOCAL/'legacy_exclusion_ledger.parquet',index=False)
    rows=[]
    for month,g in ledger.groupby('month'):
        ca=g.CA_UNQUALIFIED_EVENT_OLD_WINDOW|g.CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW
        row=dict(month=month,raw_stock_days=len(g),label_complete_stock_days=int(g.future5_complete.sum()),
          clean_stock_days=int(g.clean_research.sum()),raw_coverage_pct=100*g.clean_research.mean(),
          legacy_clean_pct_of_label_complete=100*g.clean_research.sum()/g.future5_complete.sum(),
          excluded_by_CA=int(ca.sum()),excluded_by_ST_rule=0,excluded_by_listing_boundary=0,
          excluded_by_missing_bar=int(g.MISSING_FUTURE_BAR.sum()),excluded_by_identity=0,
          excluded_by_other=int(g.LABEL_DATASET_TAIL.sum()),
          ca_unqualified_event_stock_days=int(g.CA_UNQUALIFIED_EVENT_OLD_WINDOW.sum()),
          ca_source_unknown_stock_days=int(g.CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW.sum()),
          ca_event_and_unknown_overlap=int((g.CA_UNQUALIFIED_EVENT_OLD_WINDOW&g.CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW).sum()),
          ca_event_only=int((g.CA_UNQUALIFIED_EVENT_OLD_WINDOW&~g.CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW).sum()),
          ca_unknown_only=int((~g.CA_UNQUALIFIED_EVENT_OLD_WINDOW&g.CA_SOURCE_COVERAGE_UNKNOWN_OLD_WINDOW).sum()))
        counts=g.primary_exclusion_reason.value_counts()
        for reason in flags:row['primary_'+reason]=int(counts.get(reason,0))
        row['largest_primary_exclusion_reason']=counts.drop('NONE',errors='ignore').idxmax()
        rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(ROOT/'outputs/phase23_coverage_attrition.csv',index=False)
    ledger.groupby(['month','primary_exclusion_reason']).size().rename('stock_days').to_csv(OUT/'primary_exclusion_counts.csv')
    return result

def build(data,config):
    report_audit(data)
    inv=pd.read_csv(ROOT/'outputs/phase21/universe_audit.csv',dtype={'symbol':str})
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'outputs/phase21/observed_market_calendar.csv').date)).sort_values()[-180:]
    events,qualified,members,full,gap,sources=source_inventory()
    groups=data.groupby(['exchange','symbol']).groups;parts=[];grid=[];source_rows=[]
    replay_rows=[]
    for num,r in enumerate(inv.itertuples()):
        key=(r.exchange,r.symbol);p=data.loc[groups[key]].copy().set_index('date')
        bars,invalid,digest=read_prices(ROOT/r.day_path)
        assert digest==r.sha256 and invalid==r.invalid_rows
        lower=pd.Timestamp(r.eligible_first_date)
        b=bars.loc[bars.index>=lower]
        if pd.notna(r.delisting_date):b=b.loc[b.index<=pd.Timestamp(r.delisting_date)]
        b=b.reindex(cal)
        good=[e for e in qualified[key] if lower<=pd.Timestamp(e[0])<=cal[-1]]
        gooddates={pd.Timestamp(e[0]) for e in good}
        bad=sorted(d for d in events[key] if lower<=d<=cal[-1] and d not in gooddates)
        unknown=unknown_ranges(key,lower,members,full,gap)
        vals=p.reindex(cal).copy()
        vals['consecutive_small_up_days_10']=bounded_streak(b,good,config['feature_small_up_max'])
        assert np.array_equal((vals.consecutive_small_up_days>=config['J']['small_up_days_min']).loc[p.index],
                              (vals.consecutive_small_up_days_10>=config['J']['small_up_days_min']).loc[p.index])
        validity,ca,unk,missing=feature_masks(b,vals,bad,unknown)
        le,lu=interval_flags(cal,bad,unknown,forward=5)
        p['label_ca_unqualified']=pd.Series(le,index=cal).loc[p.index]
        p['label_ca_coverage_unknown']=pd.Series(lu,index=cal).loc[p.index]
        p['label_valid']=p.future5_complete&~p.label_ca_unqualified&~p.label_ca_coverage_unknown
        extra={}
        for name in FEATURE_WINDOWS:
            extra[name]=vals[name].where(validity[name]).loc[p.index]
            extra[name+'__valid']=validity[name].loc[p.index]
        p=pd.concat([p.drop(columns=[n for n in FEATURE_WINDOWS if n in p]),pd.DataFrame(extra,index=p.index)],axis=1)
        p['core_all_features_valid']=validity[list(CORE_SHORT_FEATURE_SET)].all(axis=1).loc[p.index]
        p['full_all_features_valid']=validity.all(axis=1).loc[p.index]
        for version in VERSIONS:
            tag='full' if version==VERSIONS[0] else 'core'
            eligible=eligibility(validity,version).loc[p.index]
            for rule in 'ABCDEFGHIJ':
                p[f'{tag}_{rule}_strategy_eligible']=eligible[rule]
                p[f'{tag}_signal_{rule}']=p[f'signal_{rule}']&eligible[rule]
            p[f'{tag}_any_strategy_eligible']=eligible.any(axis=1)
            p[f'{tag}_research_eligible']=p.label_valid&eligible.any(axis=1)
            deps=set(d for rule,ds in RULE_DEPENDENCIES.items() for d in ds
                if version==VERSIONS[0] or all(x in CORE_SHORT_FEATURE_SET for x in ds))
            p[f'{tag}_any_feature_ca']=ca[sorted(deps)].any(axis=1).loc[p.index]
            p[f'{tag}_any_feature_unknown']=unk[sorted(deps)].any(axis=1).loc[p.index]
            p[f'{tag}_any_feature_missing']=missing[sorted(deps)].any(axis=1).loc[p.index]
        for k in (10,20,30,40):p[f'hit{k}']=winner(p,k)&p.label_valid
        # TARGET is unusable when its own target window is invalid. Outcomes
        # remain audit-only; never feed these columns into feature eligibility.
        for col in ('TARGET_5D_10','TARGET_2_5D_10'):
            p.loc[~p.label_valid,col]=np.nan
        for h in (2,3):
            e,u=interval_flags(cal,bad,unknown,forward=h)
            ok=~pd.Series(e|u,index=cal).loc[p.index]
            p.loc[~ok,f'TARGET_{h}D_10']=np.nan
        parts.append(p.reset_index())
        research=cal[-120:]
        absent=b.loc[research].close.isna();before=research<lower
        after=np.zeros(len(research),bool) if pd.isna(r.delisting_date) else research>pd.Timestamp(r.delisting_date)
        for month in sorted(set(research.strftime('%Y-%m'))):
            monthmask=research.strftime('%Y-%m')==month
            grid.append(dict(month=month,expected_security_days=int(monthmask.sum()),
                absent_bar_days=int((absent&monthmask).sum()),listing_boundary_days=int(((before|after)&monthmask).sum()),
                missing_inside_listing_days=int((absent&~before&~after&monthmask).sum())))
        source_rows.append(dict(exchange=r.exchange,symbol=r.symbol,full_native=key in full,
            gbbq_member=key in members,gap_member=key in gap,unknown_ranges=str(unknown),
            unqualified_events=';'.join(str(d.date()) for d in bad)))
        if num<25:
            # Direct real-security replay and future poisoning, not cached claims.
            ratio={'SSE_MAIN':.10,'SZSE_MAIN':.10,'STAR':.20,'CHINEXT':.20,'BSE':.30}[r.board]
            f=features(b,good,ratio);x=additional_features(b,good,config['feature_small_up_max'])
            rawvals=pd.concat([f,x],axis=1)
            for name in rawvals:
                np.testing.assert_allclose(rawvals[name].loc[p.index].astype(float),vals[name].loc[p.index].astype(float),equal_nan=True)
            cut=130;poison=b.copy();poison.iloc[cut+1:]=1e9
            future=[e for e in good if pd.Timestamp(e[0])<=cal[cut]]+[(str(cal[cut+2].date()),.27,41.)]
            pd.testing.assert_frame_equal(f.iloc[:cut+1],features(poison,future,ratio).iloc[:cut+1])
            pd.testing.assert_frame_equal(x.iloc[:cut+1],additional_features(poison,future,config['feature_small_up_max']).iloc[:cut+1])
            pv=vals.copy();pv.iloc[cut+1:]=np.nan
            futurebad=bad+[cal[cut+2]]
            changed=feature_masks(poison,pv,futurebad,unknown)[0]
            pd.testing.assert_frame_equal(validity.iloc[:cut+1],changed.iloc[:cut+1])
            replay_rows.append(dict(exchange=r.exchange,symbol=r.symbol,feature_replay=True,future_poison=True))
        if num%750==0:print(json.dumps({'phase23_securities':num}),flush=True)
    result=pd.concat(parts,ignore_index=True)
    assert len(result)==len(data) and not result.duplicated(['exchange','symbol','date']).any()
    result.to_parquet(LOCAL/'stock_days.parquet',index=False)
    pd.DataFrame(grid).groupby('month').sum().to_csv(OUT/'pre_universe_missing_bar_audit.csv')
    pd.DataFrame(source_rows).to_csv(OUT/'ca_source_inventory.csv',index=False)
    pd.DataFrame(replay_rows).to_csv(OUT/'independent_real_replays.csv',index=False)
    dump(OUT/'ca_source_hashes.json',sources)
    # Verify every raw price file and CA source at the end as well.
    assert all(sha(ROOT/r.day_path)==r.sha256 for r in inv.itertuples())
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in sources)
    return result,len(replay_rows)

def coverage(data):
    rows=[];remaining=[]
    for version in VERSIONS:
        tag='full' if version==VERSIONS[0] else 'core'
        for month,g in data.groupby('month'):
            mature=g.future5_complete;valid=g[f'{tag}_research_eligible']
            rows.append(dict(version=version,month=month,raw_stock_days=len(g),label_complete_stock_days=int(mature.sum()),
                label_valid_stock_days=int(g.label_valid.sum()),research_eligible_stock_days=int(valid.sum()),
                coverage_pct=100*valid.sum()/mature.sum(),coverage_pct_of_raw=100*valid.mean(),
                all_features_and_label_valid=int((g.label_valid&g[f'{tag}_all_features_valid']).sum()),
                all_features_coverage_pct=100*(g.label_valid&g[f'{tag}_all_features_valid']).sum()/mature.sum(),
                label_valid_coverage_ceiling_pct=100*g.label_valid.sum()/mature.sum()))
        flags=pd.DataFrame(index=data.index)
        flags['LABEL_UNQUALIFIED_CA']=data.label_ca_unqualified
        flags['LABEL_CA_SOURCE_UNKNOWN']=data.label_ca_coverage_unknown
        flags['LABEL_DATASET_TAIL']=data.market_pos>=115
        flags['LABEL_MISSING_BAR']=~data.future5_complete&~flags.LABEL_DATASET_TAIL
        flags['FEATURE_UNQUALIFIED_CA']=data[f'{tag}_any_feature_ca']&~data[f'{tag}_any_strategy_eligible']
        flags['FEATURE_CA_SOURCE_UNKNOWN']=data[f'{tag}_any_feature_unknown']&~data[f'{tag}_any_strategy_eligible']
        flags['FEATURE_MISSING_OR_UNDEFINED']=data[f'{tag}_any_feature_missing']&~data[f'{tag}_any_strategy_eligible']
        ledger=pd.concat([data[['exchange','symbol','date','month']],flags,exclusion_strings(flags)],axis=1)
        assert np.array_equal(ledger.primary_exclusion_reason.eq('NONE'),data[f'{tag}_research_eligible'])
        ledger.to_parquet(LOCAL/f'{tag}_exclusion_ledger.parquet',index=False)
        for month,g in ledger.groupby('month'):
            for reason in flags:
                remaining.append(dict(version=version,month=month,reason=reason,all_reason_count=int(g[reason].sum()),
                    primary_count=int(g.primary_exclusion_reason.eq(reason).sum())))
    table=pd.DataFrame(rows);table.to_csv(OUT/'coverage_by_month.csv',index=False)
    pd.DataFrame(remaining).to_csv(OUT/'remaining_blockers.csv',index=False)
    return table

def unknown_audit(data,clean,episodes,version,config):
    # Same >=20% anchor and t-3..t frozen predicates as Phase22; masks are
    # independently valid at each snapshot. No all-60-day blanket nulling.
    anchors=clean.loc[clean.hit20].sort_values('market_pos').groupby('episode_id').head(1).copy()
    lookup=data[['exchange','symbol','market_pos']+list(FEATURE_WINDOWS)].set_index(['exchange','symbol','market_pos']);flags=[]
    for lag in (0,1,2,3):
        idx=pd.MultiIndex.from_arrays([anchors.exchange,anchors.symbol,anchors.market_pos-lag],names=lookup.index.names)
        snap=lookup.reindex(idx).reset_index()
        if version==VERSIONS[1]:
            for name in FEATURE_WINDOWS:
                if name not in CORE_SHORT_FEATURE_SET:snap[name]=np.nan
            snap['consecutive_small_up_days']=snap.consecutive_small_up_days_10
        f=archetypes(snap,config);f.index=anchors.episode_id.to_numpy();flags.append(f)
    allflags=pd.concat(flags).groupby(level=0).any()
    anchors=anchors.set_index('episode_id');anchors['archetype']='UNKNOWN'
    for name in reversed(config['archetypes']['priority']):anchors.loc[allflags[name],'archetype']=name
    for rule in ('OLD5','NEW10'):
        caught=set(clean.loc[clean.hit20&clean['signal_'+rule],'episode_id'])
        anchors['caught_'+rule]=anchors.index.isin(caught)
    anchors['version']=version
    ep=episodes.set_index('episode_id')
    anchors['split']=np.select([ep.loc[anchors.index].end_pos<75,
        (ep.loc[anchors.index].start_pos>=80)&(ep.loc[anchors.index].end_pos<115)],['DISCOVERY','VALIDATION'],default='BOUNDARY_OR_PURGED')
    cols=['version','exchange','symbol','date','split','archetype','caught_OLD5','caught_NEW10']
    return anchors[cols].reset_index()

def performance(data,monthly,config):
    fields=['exchange','symbol','date','market_pos','split','month','ret_max_5d','peak_date']+list(FEATURE_WINDOWS)
    fields += [f'{prefix}{k}' for k in (10,20,30,40) for prefix in ('hit','first_hit_pos')]
    fields += [f'days_to_{k}pct' for k in (10,20,30,40)]+[f'first_hit_date{k}' for k in (10,20,30,40)]
    fields += [f'{tag}_signal_{r}' for tag in ('full','core') for r in 'ABCDEFGHIJ']
    base=data.loc[data.label_valid,fields].copy();episodes,ids=build_episodes(base);base['episode_id']=ids
    episodes.to_csv(OUT/'winner_episodes.csv',index=False)
    base[['exchange','symbol','date','episode_id']].to_parquet(LOCAL/'episode_membership.parquet',index=False)
    rows=[];unknown=[];validation=[]
    for version in VERSIONS:
        tag='full' if version==VERSIONS[0] else 'core';clean=base.copy()
        for rule in 'ABCDEFGHIJ':clean['signal_'+rule]=clean[f'{tag}_signal_{rule}']
        clean['signal_OLD5']=clean[['signal_'+r for r in 'ABCDE']].any(axis=1)
        clean['signal_NEW10']=clean[['signal_'+r for r in 'ABCDEFGHIJ']].any(axis=1)
        m=monthly[monthly.version.eq(version)].set_index('month').coverage_pct
        clean['coverage']=np.where(clean.month.map(m)>=70,'HIGH_RESEARCH_COVERAGE','LOW_RESEARCH_COVERAGE')
        val=data[data.split.eq('VALIDATION')]
        valpct=100*val[f'{tag}_research_eligible'].sum()/val.future5_complete.sum()
        validation.append(dict(version=version,coverage_pct=valpct,coverage_ge70=valpct>=70,
          raw_stock_days=len(val),label_complete_stock_days=int(val.future5_complete.sum()),
          eligible_stock_days=int(val[f'{tag}_research_eligible'].sum()),
          minimum_daily_coverage_pct=float((val.groupby('date')[f'{tag}_research_eligible'].sum()/val.groupby('date').future5_complete.sum()*100).min()),
          pristine_holdout=False,true_oos_claim_allowed=False))
        for split in ('DISCOVERY','VALIDATION'):
            for cov in ('ALL_COVERAGE','HIGH_RESEARCH_COVERAGE','LOW_RESEARCH_COVERAGE'):
                f,e,boundary=scope_data(clean,episodes,split,cov)
                for rule in list('ABCDEFGHIJ')+['OLD5','NEW10']:
                    row=evaluate(f,e,'signal_'+rule)
                    row.update(version=version,split=split,coverage=cov,rule=rule,
                        stock_days=len(f),boundary_winner_days_purged=boundary,
                        sample_interpretation='RETROSPECTIVE_FIXED_RULE_REEVALUATION_ONLY',
                        validation_coverage_ge70=valpct>=70)
                    rows.append(row)
        unknown.append(unknown_audit(data,clean,episodes,version,config))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'strategy_metrics.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_coverage.csv',index=False)
    ua=pd.concat(unknown,ignore_index=True);ua.to_csv(OUT/'episode20_archetypes.csv',index=False)
    ua.groupby(['version','split','archetype','caught_OLD5','caught_NEW10']).size().rename('episodes').to_csv(OUT/'unknown_breakdown.csv')
    return metrics,validation,ua

def main():
    guard_cached_artifact(ROOT/'data/research/phase22/stock_days.parquet')
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    protected=[p for folder in ('src','scripts','config','tests') for p in (ROOT/folder).rglob('*')
        if p.is_file() and p.suffix in ('.py','.json') and 'phase23' not in p.name and p.name!='coverage_recovery.py']
    protected += [ROOT/'data/research/phase21/stock_days.parquet',ROOT/'data/research/phase22/stock_days.parquet']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    reuse='--reuse-built' in sys.argv
    if reuse:
        assert before==json.loads((OUT/'protected_before.json').read_text()),'REUSE_INPUT_CHANGED'
    else:
        dump(OUT/'protected_before.json',before)
    config=json.loads((ROOT/'config/phase22_research.json').read_text())
    assert not config['amount_enabled'] and config['J']['small_up_days_min']<=10
    data=pd.read_parquet(ROOT/'data/research/phase22/stock_days.parquet')
    # Reconstruct, not coverage-dependent date selection: first80 assigned,
    # purge final5, next35 validation, final5 censored tail.
    dates=pd.DatetimeIndex(sorted(data.date.unique()));assert len(dates)==120
    data['market_pos']=dates.get_indexer(data.date)
    data['split']=np.select([data.market_pos<75,data.market_pos<80,data.market_pos<115],
        ['DISCOVERY','PURGED_DISCOVERY','VALIDATION'],default='UNLABELED_TAIL')
    audit=attrition(data)
    if reuse:
        built=pd.read_parquet(LOCAL/'stock_days.parquet')
        assert len(built)==len(data) and not built.duplicated(['exchange','symbol','date']).any()
        sources=json.loads((OUT/'ca_source_hashes.json').read_text())
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in sources)
        inv=pd.read_csv(ROOT/'outputs/phase21/universe_audit.csv',dtype={'symbol':str})
        assert all(sha(ROOT/r.day_path)==r.sha256 for r in inv.itertuples())
        replayed=len(pd.read_csv(OUT/'independent_real_replays.csv'))
    else:
        built,replayed=build(data,config)
    del data
    monthly=coverage(built)
    metrics,validation,unknown=performance(built,monthly,config)
    assert all(sha(ROOT/p)==v for p,v in before.items()),'PROTECTED_INPUT_CHANGED'
    core=monthly[monthly.version.eq(VERSIONS[1])].set_index('month')
    recovered=all(core.loc[m,'coverage_pct']>=70 for m in ('2026-07','2026-08','2026-09')) and validation[1]['coverage_ge70']
    status='PHASE23_COVERAGE_RECOVERED' if recovered else 'PHASE23_COVERAGE_STILL_BLOCKED'
    health=dict(status=status,raw_stock_days=len(built),validation=validation,
        core_july_pct=core.loc['2026-07','coverage_pct'],core_august_pct=core.loc['2026-08','coverage_pct'],core_september_pct=core.loc['2026-09','coverage_pct'],
        coverage_definition='label_valid AND any frozen strategy feature-eligible / future5_complete; raw denominator also reported',
        episode_denominator='COMMON_LABEL_VALID_UNIVERSE_NOT_RESTRICTED_TO_FEATURE_ELIGIBLE_ROWS',
        historical_st_unknown_allowed=True,point_in_time_CA_publication_timestamps_verified=False,
        lookahead_future_perturbation_failures=0,real_securities_replayed=replayed,
        rule_thresholds_unchanged=True,core_rules_available=['J'],core_old5_available=[],
        core_J_equivalence='min(consecutive_small_up_days,10)>=3 iff original>=3; checked all raw stock-days',
        discovery_assigned_days=80,discovery_evaluated_days=75,purge_days=5,validation_days=35,
        discovery_start=str(dates[0].date()),discovery_end=str(dates[74].date()),validation_start=str(dates[80].date()),validation_end=str(dates[114].date()),
        pristine_holdout=False,prior_validation_outcomes_seen=True,
        protected_file_hash_checks=len(before),production_unchanged=True,amount_used=False,ML_entered=False,commit=False,push=False,
        unknown_remaining=[dict(version=v,missed_old5_unknown=int(((unknown.version==v)&(unknown.archetype=='UNKNOWN')&~unknown.caught_OLD5).sum()),
            missed_new10_unknown=int(((unknown.version==v)&(unknown.archetype=='UNKNOWN')&~unknown.caught_NEW10).sum())) for v in VERSIONS])
    dump(ROOT/'outputs/phase23_research_health.json',health)
    dump(OUT/'feature_dependency_manifest.json',dict(feature_windows=FEATURE_WINDOWS,core_short_features=CORE_SHORT_FEATURE_SET,
        strategy_dependencies=RULE_DEPENDENCIES,ca_boundary='start < ex_date <= end',mask_suffix='__valid',
        ST='Historical unknown allowed unchanged; current ST snapshot never backfilled',
        identity='Frozen Phase21 research universe, not production identity certification'))
    dump(OUT/'protected_after.json',{p:sha(ROOT/p) for p in before})
    print(json.dumps(health,default=str),flush=True)

if __name__=='__main__':main()
