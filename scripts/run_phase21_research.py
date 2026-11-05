"""Frozen120-session observational research. No network, trading, or production gate writes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data.tdx_discovery import RECORD_SIZE
from src.universe_policy import exclusion_reason, report_audit
from src.data.adjustment_engine import action_transform
from src.data.corporate_actions import resolve_dividend_basis
from src.research.target_discovery import FEATURES, RULE_NAMES, features, labels, signals, event_mask

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase21';LOCAL=ROOT/'data/research/phase21'


def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def board_for(symbol,exchange,official_board=None):
    if exchange=='SSE':
        if symbol.startswith('688'):return 'STAR'
        if symbol.startswith(('600','601','603','605')):return 'SSE_MAIN'
    if exchange=='SZSE':
        if symbol.startswith(('300','301','302')):return 'CHINEXT'
        if symbol.startswith(('000','001','002','003')):return 'SZSE_MAIN'
    if exchange=='BSE' and official_board=='BSE' and symbol.startswith(('4','8','920')):return 'BSE'
    return None


def read_prices(path):
    """Same verified32-byte layout; Amount word is never interpreted or used."""
    raw=path.read_bytes()
    if not raw or len(raw)%RECORD_SIZE:raise ValueError('DAY_RECORD_SIZE')
    words=np.frombuffer(raw,dtype='<u4').reshape(-1,RECORD_SIZE//4)
    dates=pd.to_datetime(words[:,0].astype(str),format='%Y%m%d',errors='coerce')
    frame=pd.DataFrame(dict(date=dates,open=words[:,1]/100,high=words[:,2]/100,
        low=words[:,3]/100,close=words[:,4]/100,volume=words[:,6].astype(float)))
    if frame.date.duplicated().any():raise ValueError('DUPLICATE_DAY_DATE')
    good=(frame.date.notna()&(frame[['open','high','low','close']]>0).all(axis=1)&
        (frame.high>=frame[['open','low','close']].max(axis=1))&
        (frame.low<=frame[['open','high','close']].min(axis=1)))
    return frame[good].set_index('date').sort_index(),int((~good).sum()),hashlib.sha256(raw).hexdigest()


def universe(asof):
    physical=pd.read_csv(ROOT/'outputs/tdx_official_universe.csv',dtype={'symbol':str})
    master=pd.read_parquet(ROOT/'data/normalized/security_master.parquet')
    m={(r.exchange,str(r.symbol).zfill(6)):r for r in master.itertuples()}
    records=[];bars={};exclusions=[];calendar=set()
    for ordinal,r in enumerate(physical.itertuples()):
        exchange={'sh':'SSE','sz':'SZSE','bj':'BSE'}.get(r.market)
        symbol=str(r.symbol).zfill(6);key=(exchange,symbol);official=m.get(key)
        reason=exclusion_reason(exchange,symbol)
        if reason:
            exclusions.append(dict(symbol=symbol,exchange=exchange,eligible=False,reason=reason,exclusion_reason=reason))
            continue
        official_board=getattr(official,'board',None) if official is not None else None
        # BSE numerical codes alone can also designate NEEQ instruments.
        if exchange=='BSE' and (official is None or str(official.source).startswith('TDX_')):
            continue
        board=board_for(symbol,exchange,official_board)
        if board is None:continue
        if key in bars:raise ValueError('DUPLICATE_SECURITY_FILE')
        path=Path(r.file_path).resolve()
        assert path.is_relative_to((ROOT/'data/tdx_official_extract').resolve())
        assert path.parent.parent.name==r.market and path.stem==r.market+symbol
        try:frame,invalid,digest=read_prices(path)
        except ValueError as exc:
            exclusions.append(dict(symbol=symbol,exchange=exchange,reason=str(exc)));continue
        if frame.empty:continue
        first=frame.index.min()
        listing=pd.to_datetime(getattr(official,'listing_date',None),errors='coerce')
        delisting=pd.to_datetime(getattr(official,'delisting_date',None),errors='coerce')
        lower=max(first,listing) if not pd.isna(listing) else first
        frame=frame.loc[(frame.index>=lower)&(frame.index<=asof)]
        if not pd.isna(delisting):frame=frame.loc[frame.index<=delisting]
        # Enough history to discover180 actual sessions; no imported holidays.
        frame=frame.loc[frame.index>=pd.Timestamp('2025-01-01')]
        if frame.empty:continue
        status=str(getattr(official,'listing_status','HISTORICAL_UNKNOWN'))
        st=getattr(official,'st_flag',None)
        st=False if st is None or pd.isna(st) else bool(st)
        row=dict(symbol=symbol,exchange=exchange,board=board,name=str(getattr(official,'security_name','UNKNOWN')),
            current_listing_status=status,current_st=st,listing_date=None if pd.isna(listing) else str(listing.date()),
            delisting_date=None if pd.isna(delisting) else str(delisting.date()),
            first_raw_bar_date=str(first.date()),eligible_first_date=str(lower.date()),
            identity_source=str(getattr(official,'source','TDX_CODE_EXCHANGE_RESEARCH_ONLY')),
            day_path=str(path.relative_to(ROOT)),sha256=digest,invalid_rows=invalid)
        records.append(row);bars[key]=frame;calendar.update(frame.index)
        if ordinal%1000==0:print(json.dumps({'inventory_row':ordinal,'loaded_research_security_files':len(bars)}),flush=True)
    return records,bars,pd.DatetimeIndex(sorted(calendar)),exclusions


def action_inventory():
    eventdates=defaultdict(set);source_members=set();native_full=set();gap_members=set();qualified={}
    g=pd.read_csv(ROOT/'data/cache/tdx_gbbq_parsed.csv',dtype={'code':str})
    market={0:'SZSE',1:'SSE',2:'BSE'}
    # Category1 is normalized ex-rights. Other selected categories are treated
    # conservatively as potential price-basis changes, never assigned guessed A/B.
    cautious={1,4,11,12,13,14,15}
    for r in g.itertuples():
        if r.market not in market:continue
        key=(market[r.market],str(r.code).zfill(6));source_members.add(key)
        if r.category in cautious:eventdates[key].add(pd.Timestamp(str(int(r.datetime))))
    for path in sorted((ROOT/'data/tdxaidata_raw/gap_corporate_actions').glob('*.json')):
        o=json.loads(path.read_text(encoding='utf-8'))
        if str(o.get('ErrorId'))!='0' or o.get('schema_valid') is not True:continue
        if o.get('requested_start')!='2026-08-25' or o.get('requested_end')!='2026-09-30':continue
        key=(o['exchange'],o['symbol']);gap_members.add(key)
        for d in o['Date']:eventdates[key].add(pd.Timestamp(d))
    sample=pd.read_csv(ROOT/'outputs/tdxaidata_100_sample.csv',dtype={'symbol':str})
    for r in sample.itertuples():
        path=ROOT/f'data/tdxaidata_raw/probe100/{r.symbol}.divid.native.json'
        o=json.loads(path.read_text(encoding='utf-8'))
        assert str(o['ErrorId'])=='0' and len(o['Date'])==len(o['Type'])==len(o['Value'])
        key=(r.exchange,r.symbol);native_full.add(key)
        for d in o['Date']:eventdates[key].add(pd.Timestamp(d))
    fixturepaths=['differential_dividend/events.json','differential_dividend/events_phase19e.json',
                  'phase19f/reviewed_differential.json','phase19i/reviewed_differential.json']
    verified_rows=[]
    for name in fixturepaths:
        for e in load('data/test_fixtures/'+name):
            resolved=resolve_dividend_basis(e);transform=action_transform(pd.Series(resolved))
            raw=Path(e['basis_raw_reference'])
            if not raw.is_absolute():raw=ROOT/raw
            assert raw.resolve().is_relative_to(ROOT.resolve()) and raw.is_file()
            if e.get('source_sha256'):assert sha(raw).lower()==e['source_sha256'].lower()
            key=(e['exchange'],e['symbol']);d=pd.Timestamp(e['ex_date'])
            qualified[(key,d)]=(str(d.date()),transform.A,transform.B)
            eventdates[key].add(d)
            verified_rows.append(dict(symbol=e['symbol'],exchange=e['exchange'],ex_date=e['ex_date'],A=transform.A,B=transform.B))
    assert len(qualified)==68
    # Two rights fixtures certify isolated components, not completeness of every
    # simultaneous action. Keep their dates unqualified for this new study.
    for p in (ROOT/'data/test_fixtures/rights_validation').glob('*.json'):
        e=json.loads(p.read_text());eventdates[(e['exchange'],e['symbol'])].add(pd.Timestamp(e['ex_date']))
    pd.DataFrame(verified_rows).to_csv(OUT/'qualified_event_parameters.csv',index=False)
    return eventdates,qualified,source_members,native_full,gap_members


def window_intersects(calendar,first,start,end):
    starts=calendar[np.maximum(np.arange(len(calendar))-60,0)]
    ends=calendar[np.minimum(np.arange(len(calendar))+5,len(calendar)-1)]
    return (starts<=pd.Timestamp(end))&(ends>=pd.Timestamp(start))&(pd.Timestamp(first)<=pd.Timestamp(end))


def compute(config):
    asof=pd.Timestamp(config['as_of'])
    records,raw,observed,excluded=universe(asof)
    assert len(observed)>=180 and observed[-1]==asof
    calendar=observed[-180:];research=calendar[-120:]
    pd.DataFrame({'date':observed,'is_research_window':observed.isin(research)}).to_csv(OUT/'observed_market_calendar.csv',index=False)
    events,qualified,members,full,gap=action_inventory()
    outputs=[];inventory=[];counts=Counter();before_hashes={}
    for num,r in enumerate(records):
        key=(r['exchange'],r['symbol']);b=raw[key].reindex(calendar)
        if not b.loc[research].close.notna().any():continue
        before_hashes[r['day_path']]=r['sha256']
        lower=pd.Timestamp(r['eligible_first_date'])
        dates=[d for d in events[key] if lower<=d<=asof]
        good=[qualified[(key,d)] for d in dates if (key,d) in qualified]
        bad=[d for d in dates if (key,d) not in qualified]
        ratio={'SSE_MAIN':.10,'SZSE_MAIN':.10,'STAR':.20,'CHINEXT':.20,'BSE':.30}[r['board']]
        f=features(b,good,ratio);y=labels(b,good);s=signals(f,config['rule_thresholds'])
        d=pd.concat([f,y,s.add_prefix('signal_')],axis=1)
        d['close_t']=b.close
        d['future5_complete']=y.TARGET_5D_10.notna()
        d['CA_UNVERIFIED_WINDOW']=event_mask(calendar,bad)
        coverage=np.zeros(len(calendar),dtype=bool)
        if key not in full:
            if key not in members:
                coverage|=window_intersects(calendar,lower,'1900-01-01','2026-08-18')
            coverage|=window_intersects(calendar,lower,'2026-08-19','2026-08-24')
            if key not in gap:
                coverage|=window_intersects(calendar,lower,'2026-08-25','2026-09-30')
        d['CA_COVERAGE_UNKNOWN_WINDOW']=coverage
        d['clean_research']=d.future5_complete&~d.CA_UNVERIFIED_WINDOW&~coverage
        d['historical_st_status']='UNKNOWN'
        # Snapshot status is known only for its own date, never backward-filled.
        if r['current_listing_status']=='ACTIVE':
            d.loc[asof,'historical_st_status']='KNOWN_ST' if r['current_st'] else 'KNOWN_NON_ST'
        d['current_or_known_st']=r['current_st']|(d.historical_st_status=='KNOWN_ST')
        for lag in (1,2,3):
            for feat in ('ret_1d','ret_5d','close_location','breakout_20d','volume_ratio_5','position_60d'):
                # Audit-only precursor columns need their OWN clean window.
                # A t-clean sample does not prove the t-3 sixty-day window clean.
                safe=~d.CA_UNVERIFIED_WINDOW&~d.CA_COVERAGE_UNKNOWN_WINDOW
                d[f'pre{lag}_{feat}']=f[feat].where(safe).shift(lag)
        d=d.loc[research].loc[b.loc[research].close.notna()].copy()
        for k in ('symbol','exchange','board','name','current_listing_status','listing_date','delisting_date','first_raw_bar_date'):d[k]=r[k]
        d.index.name='date';outputs.append(d.reset_index())
        inventory.append(r|dict(research_rows=len(d),clean_rows=int(d.clean_research.sum()),
            observed_ca_events=len(dates),unqualified_ca_events=len(bad),qualified_ca_events=len(good),
            ca_event_excluded=int(d.CA_UNVERIFIED_WINDOW.sum()),coverage_unknown_excluded=int(d.CA_COVERAGE_UNKNOWN_WINDOW.sum())))
        counts.update({'invalid_bar_rows':r['invalid_rows']})
        if num%1000==0:print(json.dumps({'processed_securities':num,'research_rows_so_far':sum(len(x) for x in outputs)}),flush=True)
    data=pd.concat(outputs,ignore_index=True)
    assert not data.duplicated(['exchange','symbol','date']).any()
    assert 'amount' not in data and len(research)==120
    data.to_parquet(LOCAL/'stock_days.parquet',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'universe_audit.csv',index=False)
    pd.DataFrame(excluded).to_csv(OUT/'file_exclusions.csv',index=False)
    changed=[p for p,digest in before_hashes.items() if sha(ROOT/p)!=digest]
    assert not changed,'RAW_CHANGED'
    return data,research,inventory,counts,len(before_hashes)


def winner(frame,pct):
    raw_peak=np.rint(frame.close_t*(1+frame.raw_ret_max_5d)*100)
    raw_hit=raw_peak*100 >= np.rint(frame.close_t*100)*(100+pct)
    comparable=frame.future_max_5d*100 >= frame.close_t*(100+pct)
    return pd.Series(np.where(frame.label_crosses_qualified_ca,comparable,raw_hit),index=frame.index)


def strategy_table(frame,sample_name):
    report_audit(frame)
    rows=[];targets={n:winner(frame,n) for n in (10,20,30,40)}
    for rule in ['A','B','C','D','E','UNION']:
        mask=frame['signal_'+rule];chosen=frame.loc[mask]
        hit=int((mask&targets[10]).sum());total=int(targets[10].sum())
        row=dict(bse_candidate_count=0,sample=sample_name,rule=rule,name=RULE_NAMES.get(rule,'Union of fixed rules'),
            candidate_count=len(chosen),winner_count=hit,all_winner_denominator=total,
            precision_pct=100*hit/len(chosen) if len(chosen) else None,
            capture_rate_pct=100*hit/total if total else None)
        for n in (20,30,40):
            denominator=int(targets[n].sum());caught=int((mask&targets[n]).sum())
            row[f'winner{n}_count']=denominator;row[f'caught{n}']=caught
            row[f'capture{n}_pct']=100*caught/denominator if denominator else None
        for h in (2,3,5):row[f'target{h}_count']=int(chosen[f'TARGET_{h}D_10'].sum())
        for h in (1,3,5):
            row[f'mean_max_return_{h}d']=chosen[f'ret_max_{h}d'].mean()
            row[f'median_max_return_{h}d']=chosen[f'ret_max_{h}d'].median()
            row[f'worst_adverse_excursion_{h}d']=chosen[f'max_adverse_{h}d'].min()
        rows.append(row)
    return pd.DataFrame(rows)


def feature_statistics(frame,sample):
    win=winner(frame,10);normal=frame.ret_max_5d<.05
    failed=(frame.signal_B|frame.signal_C)&~win
    groups={'STRONG_WINNER':win,'NORMAL':normal,'FALSE_BREAKOUT':failed,
            'LOSER':(frame.ret_max_5d<.03)&(frame.max_adverse_5d<=-.05)}
    rows=[];ranking=[];buckets=[]
    for name in FEATURES:
        for group,mask in groups.items():
            x=frame.loc[mask,name].dropna()
            rows.append(dict(sample=sample,feature=name,group=group,n=len(x),mean=x.mean(),
                median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75)))
        valid=frame[name].notna();x=frame.loc[valid,name];target=win.loc[valid]
        n1=int(target.sum());n0=int((~target).sum())
        auc=(x.rank(method='average').loc[target].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
        ranking.append(dict(sample=sample,feature=name,n=len(x),coverage_pct=100*len(x)/len(frame),
            descriptive_auc=auc,abs_rank_biserial=abs(2*auc-1),direction='higher' if auc>=.5 else 'lower'))
        if x.nunique()>1:
            if x.nunique()<=5:
                cut=x  # Preserve distinct binary/discrete values; quantiles collapse them.
            else:
                cut=pd.cut(x,[-np.inf,.8,1.2,1.5,2,3,np.inf],right=False) if name=='volume_ratio_5' else pd.qcut(x,5,duplicates='drop')
            for bucket,idx in cut.groupby(cut,observed=True).groups.items():
                y=target.loc[idx];buckets.append(dict(sample=sample,feature=name,bucket=str(bucket),n=len(y),winners=int(y.sum()),hit_rate_pct=100*y.mean()))
    return pd.DataFrame(rows),pd.DataFrame(ranking).sort_values('abs_rank_biserial',ascending=False),pd.DataFrame(buckets)


def miss_audit(frame):
    audit=frame.loc[winner(frame,20)].copy()
    audit['at_least30']=winner(audit,30);audit['captured']=audit.signal_UNION
    # These are observable structural predicates, not causal explanations.
    predicates={'NO_BREAKOUT':audit.breakout_20d==0,'NO_VOLUME_EXPANSION':audit.volume_ratio_5<1.5,
        'ALREADY_HIGH_POSITION':(audit.position_60d>.85)&(audit.close_vs_ma5>.05),
        'EARLY_REVERSAL':(audit.ret_20d<0)&(audit.ret_3d>0),
        'LIMIT_UP_START':audit.limit_up_today==1}
    for k,v in predicates.items():audit['reason_'+k]=v
    audit['SECTOR_DRIVEN']='UNKNOWN_NO_SECTOR_DATA'
    order=['LIMIT_UP_START','EARLY_REVERSAL','ALREADY_HIGH_POSITION','NO_BREAKOUT','NO_VOLUME_EXPANSION']
    audit['miss_type']='UNKNOWN'
    for reason in reversed(order):audit.loc[predicates[reason],'miss_type']=reason
    audit.loc[audit.captured,'miss_type']='CAPTURED'
    # Overlapping observations remain distinct; event clustering is supplemental.
    audit=audit.sort_values(['exchange','symbol','date'])
    audit.to_csv(OUT/'winner20_30_miss_audit.csv',index=False)
    audit.loc[audit.at_least30].to_csv(OUT/'winner30_miss_audit.csv',index=False)
    return audit


def main():
    config=load('config/phase21_research.json')
    assert not config['amount_enabled'] and not config['production_tradable_universe_ready']
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    production_before=sha(ROOT/'scripts/phase2_readiness.py')
    cfg_hash=sha(ROOT/'config/phase21_research.json')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=DeprecationWarning)
        data,research,inventory,counts,hashed=compute(config)
    clean=data.loc[data.clean_research].copy()
    assert len(clean)>0,'NO_CLEAN_RESEARCH_SAMPLES'
    samples={'ALL_RESEARCH_SAMPLE':clean,
        'CURRENTLY_OR_KNOWN_ST_EXCLUDED_SAMPLE':clean.loc[~clean.current_or_known_st]}
    summaries={};strategy=[];ranks=[];groups=[];bins=[];audits={}
    for name,frame in samples.items():
        strategies=strategy_table(frame,name);strategy.append(strategies)
        g,r,b=feature_statistics(frame,name);groups.append(g);ranks.append(r);bins.append(b)
        summaries[name]=dict(stock_days=len(frame),stocks=int(frame.groupby(['exchange','symbol']).ngroups),
            historical_st_unknown_pct=100*frame.historical_st_status.eq('UNKNOWN').mean(),
            **{f'winners{n}':int(winner(frame,n).sum()) for n in (10,20,30,40)},
            target_2d_10=int(frame.TARGET_2D_10.sum()),target_3d_10=int(frame.TARGET_3D_10.sum()),
            hit_rate_pct=100*winner(frame,10).mean(),top10=r.head(10).feature.tolist(),
            raw_formula_winners10=int((np.rint(frame.close_t*(1+frame.raw_ret_max_5d)*100)*100>=np.rint(frame.close_t*100)*110).sum()),
            qualified_ca_target_windows=int(frame.label_crosses_qualified_ca.sum()))
    strategies=pd.concat(strategy);rank=pd.concat(ranks)
    strategies.to_csv(OUT/'baseline_metrics.csv',index=False);rank.to_csv(OUT/'feature_ranking.csv',index=False)
    pd.concat(groups).to_csv(OUT/'feature_group_statistics.csv',index=False)
    pd.concat(bins).to_csv(OUT/'feature_buckets.csv',index=False)
    audit=miss_audit(clean)
    # Define best as largest captured winner count, tie-break precision. No tuning.
    best=strategies[(strategies['sample']=='ALL_RESEARCH_SAMPLE')&(strategies.rule!='UNION')].sort_values(['winner_count','precision_pct'],ascending=False).iloc[0]
    missed=audit.loc[~audit.captured,'miss_type'].value_counts().to_dict()
    health=dict(bse_candidate_count=report_audit(data)['bse_candidate_count'],run_time=datetime.now(timezone.utc).isoformat(),status='PHASE2_RESEARCH_BASELINE_COMPLETE',
        scope='RESEARCH_ONLY_NOT_TRADABLE_BACKTEST',RESEARCH_UNIVERSE_READY=True,
        PRODUCTION_TRADABLE_UNIVERSE_READY=False,calendar_status='OFFICIAL_CALENDAR_NOT_YET_PRODUCTION_VERIFIED',
        config_sha256=cfg_hash,observed_trading_days=len(research),start=str(research[0].date()),end=str(research[-1].date()),
        raw_research_stock_days=len(data),raw_research_stocks=int(data.groupby(['exchange','symbol']).ngroups),
        terminal_status_stock_days=int(data.current_listing_status.eq('TERMINATED').sum()),
        terminal_status_clean_stock_days=int(clean.current_listing_status.eq('TERMINATED').sum()),
        future5_unavailable=int((~data.future5_complete).sum()),
        ca_unverified_window_excluded=int(data.CA_UNVERIFIED_WINDOW.sum()),
        ca_coverage_unknown_window_excluded=int(data.CA_COVERAGE_UNKNOWN_WINDOW.sum()),
        ca_union_excluded=int((data.CA_UNVERIFIED_WINDOW|data.CA_COVERAGE_UNKNOWN_WINDOW).sum()),
        mature_ca_union_excluded=int((data.future5_complete&(data.CA_UNVERIFIED_WINDOW|data.CA_COVERAGE_UNKNOWN_WINDOW)).sum()),
        counts=dict(counts),samples=summaries,best_baseline=best.to_dict(),
        union_baseline=strategies[(strategies['sample']=='ALL_RESEARCH_SAMPLE')&(strategies.rule=='UNION')].iloc[0].to_dict(),
        missed20_types={k:int(v) for k,v in missed.items()},winner20_unique_stocks=int(audit.groupby(['exchange','symbol']).ngroups),
        raw_hashes_rechecked=hashed,RAW_modified=False,production_gate_unchanged=True,
        lookahead_feature_test='PENDING_TEST_XML',future_event_filter_used_only_for_cohort_hygiene=True,
        model_trained=False,amount_used=False,commit_created=False,pushed=False)
    assert sha(ROOT/'scripts/phase2_readiness.py')==production_before
    (ROOT/'outputs/phase21_research_health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(health,ensure_ascii=False,indent=2,default=str),flush=True)


if __name__=='__main__':main()
