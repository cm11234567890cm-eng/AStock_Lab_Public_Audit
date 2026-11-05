"""Frozen Discovery-only transparent ranking; retrospective research only."""
import json
import numpy as np
import pandas as pd
from scripts.run_phase21_research import ROOT,sha
from scripts.run_phase23_research import dump
from src.research.daily_ranking import METHODS,KS,KEYS,FEATURES,GROUPS,decision_view,fit_model,score_rows,rank_rows,choose_method
from src.research.episodes import build_episodes,evaluate,consecutive_groups
from src.universe_policy import guard_cached_artifact, report_audit

OUT=ROOT/'outputs/phase24a';LOCAL=ROOT/'data/research/phase24a'

def contract():
    spec=dict(scope='RESEARCH_ONLY_NOT_TRADABLE_BACKTEST',methods=list(METHODS),groups=GROUPS,
        primary_target='max(high[t+2],...,high[t+5])/close[t]-1; strict integer-cent threshold; CA censor (t,t+5]',
        compatibility_target='original max(high[t+1],...,high[t+5])/close[t]-1; episode weights remain on this frozen definition',
        fit='Discovery dates only, last5 of assigned80 purged; no global future episode identities used to select fit rows',
        selection='Discovery mean(TOP3 hit10_2_5,TOP5 hit10_2_5); tie daily_success_TOP3 then R1,R2,R3,R4; selected-label coverage >=70%',
        ranking_universe='All contemporaneous feature-eligible stocks, original A-J candidates first; score then outcome-blind SHA256 tie key',
        label_censoring='Rank before any future-label filtering; never replace contaminated TOP slots with lower ranks',
        feature_normalization='101 Discovery quantiles; feature direction is sign of Discovery mean percentile hit10 minus non-hit10; fixed six groups, equal weights',
        R4='0.5 normalized_R2 + 0.5 equal_mean_six_feature_scores; invalid raw features remain NaN, missing centered contribution 0',
        opportunity='Daily max literal 1..5 return in common CA-clean label universe, including feature-ineligible/noncandidate; unranked counts as infinity; ties retained',
        effectiveness='Discovery-selected method only: Validation TOP3 AND TOP5 pooled hit10_2_5 lift >=5pp over same-unit unranked candidate baseline, 95% paired 5-day moving-block bootstrap mean daily lift lower bound >0, and label coverage>=70%',
        bootstrap=dict(seed=2401,replicates=2000,block_sessions=5),
        episode_reference_precision=15.95685,episode_reference_not_comparable_to_daily_stock_hit_rate=True,
        pristine_holdout=False,prior_validation_seen=True,amount_used=False,ML_entered=False,
        frozen_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'config/phase21_research.json',ROOT/'config/phase22_research.json',
            ROOT/'src/research/target_discovery.py',ROOT/'src/research/episodes.py',ROOT/'src/research/coverage_recovery.py']})
    p=OUT/'analysis_contract.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==spec,'CONTRACT_CHANGED'
    else:dump(p,spec)
    return spec

def fit_strategy_weights(discovery):
    assert discovery.split.eq('DISCOVERY').all()
    report_audit(discovery)
    d=discovery[discovery.label_valid].copy()
    eps,ids=build_episodes(d);d['episode_id']=ids
    rows=[];weights={}
    for r in 'ABCDEFGHIJ':
        col='full_signal_'+r
        row=evaluate(d,eps,col)
        clusters=consecutive_groups(d,d[col])
        for k in (20,30):
            successes=int(d.loc[clusters.index,f'hit{k}'].groupby(clusters).any().sum()) if len(clusters) else 0
            row[f'successful_candidate_episodes{k}']=successes
            row[f'episode_precision{k}']=100*successes/clusters.nunique() if len(clusters) else 0.
        row.update(strategy=r,fit_split='DISCOVERY',episode_boundary_policy='UNCHANGED_GROUPING_WITHIN_DISCOVERY_OBSERVATIONS_ONLY',fit_stock_days=len(d))
        p10=row['episode_precision'] if np.isfinite(row['episode_precision']) else 0.
        weights[r]=dict(precision10=p10/100,big_winner=(row['episode_precision20']+row['episode_precision30'])/200)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'discovery_strategy_weights.csv',index=False)
    return weights

def evaluate_top(ranked,truth,split_dates):
    report_audit(ranked)
    report_audit(truth)
    top=ranked[ranked['rank']<=20].merge(truth,on=KEYS,validate='one_to_one')
    rows=[];daily=[]
    for split,dates in split_dates.items():
        for k in KS:
            p=top[top.split.eq(split)&top['rank'].le(k)].copy()
            for horizon in ('2_5d','1_5d'):
                for threshold in (10,20,30):
                    col=f'outcome_hit{threshold}_{horizon}'
                    known=p.label_valid&p[col].notna()
                    hits=known&p[col].fillna(False).astype(bool)
                    day=pd.DataFrame(dict(date=p.date,known=known,hits=hits,unknown=~known,selected=True)).groupby('date').sum().reindex(dates,fill_value=0)
                    successes=day.hits.gt(0);unresolved=~successes&day.unknown.gt(0)
                    row=dict(bse_candidate_count=0,method=ranked.method.iloc[0],split=split,k=k,horizon=horizon,threshold=threshold,
                        selected_slots=len(p),expected_slots=k*len(dates),evaluable_slots=int(known.sum()),hit_slots=int(hits.sum()),
                        hit_rate_pct=100*hits.sum()/known.sum() if known.any() else np.nan,
                        hit_lower_pct=100*hits.sum()/(k*len(dates)),label_coverage_pct=100*known.sum()/len(p) if len(p) else 0,
                        days=len(dates),successful_days=int(successes.sum()),unresolved_days=int(unresolved.sum()),
                        daily_success_lower_pct=100*successes.mean(),daily_success_upper_pct=100*(successes|unresolved).mean())
                    rows.append(row)
                    for date,x in day.iterrows():daily.append(dict(method=row['method'],split=split,k=k,horizon=horizon,threshold=threshold,
                        date=date,selected=int(x.selected),evaluable=int(x.known),hits=int(x.hits),unknown=int(x.unknown),
                        daily_hit_rate_pct=100*x.hits/x.known if x.known else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(daily),top

def candidate_baseline(decisions,truth,split_dates):
    report_audit(decisions)
    report_audit(truth)
    joined=decisions[decisions.candidate].merge(truth,on=KEYS,validate='one_to_one')
    rows=[];daily=[];counts=[]
    for split,dates in split_dates.items():
        d=joined[joined.split.eq(split)]
        cnt=d.groupby('date').size().reindex(dates,fill_value=0)
        counts.append(dict(split=split,days=len(dates),mean=float(cnt.mean()),median=float(cnt.median()),
            p25=float(cnt.quantile(.25)),p75=float(cnt.quantile(.75)),max=int(cnt.max()),min=int(cnt.min())))
        for horizon in ('2_5d','1_5d'):
            for n in (10,20,30):
                col=f'outcome_hit{n}_{horizon}';known=d.label_valid&d[col].notna();hit=known&d[col].fillna(False).astype(bool)
                per=pd.DataFrame(dict(date=d.date,known=known,hit=hit)).groupby('date').sum().reindex(dates,fill_value=0)
                rows.append(dict(split=split,horizon=horizon,threshold=n,candidates=len(d),evaluable=int(known.sum()),hits=int(hit.sum()),
                    hit_rate_pct=100*hit.sum()/known.sum(),label_coverage_pct=100*known.sum()/len(d)))
                for date,x in per.iterrows():daily.append(dict(split=split,horizon=horizon,threshold=n,date=date,
                    evaluable=int(x.known),hits=int(x.hit),daily_hit_rate_pct=100*x.hit/x.known if x.known else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(daily),pd.DataFrame(counts)

def opportunity(ranked,truth,decisions,dates):
    report_audit(ranked)
    report_audit(truth)
    report_audit(decisions)
    d=truth[truth.label_valid&truth.outcome_max_return_1_5d.notna()&truth.date.isin(dates)].merge(
        decisions[KEYS+['tie_key','candidate','full_any_strategy_eligible']],on=KEYS,validate='one_to_one')
    peak=d.groupby('date').outcome_max_return_1_5d.transform('max')
    winners=d[d.outcome_max_return_1_5d.eq(peak)].copy()
    winners=winners.merge(ranked[KEYS+['rank','tie_best_rank','tie_worst_rank']],on=KEYS,how='left',validate='one_to_one')
    winners=winners.sort_values(['date','tie_key']);primary=winners.drop_duplicates('date').copy()
    assert len(primary)==len(dates),'MISSING_MARKET_WINNER_DAY'
    ranks=primary['rank'].fillna(np.inf).to_numpy()
    def q(x):
        v=float(np.quantile(ranks,x,method='inverted_cdf'));return 'UNRANKED' if not np.isfinite(v) else v
    summary=dict(method=ranked.method.iloc[0],split='VALIDATION',days=len(primary),winner_rank_p25=q(.25),
        winner_rank_median=q(.5),winner_rank_p75=q(.75),unranked_days=int(primary['rank'].isna().sum()),
        noncandidate_winner_days=int((~primary.candidate).sum()),multiple_max_winner_days=int((winners.groupby('date').size()>1).sum()))
    for k in (3,5,10,20):
        summary[f'winner_in_top{k}_pct']=100*float((ranks<=k).mean())
        summary[f'any_co_winner_in_top{k}_pct']=100*float(winners.assign(in_top=winners['rank'].le(k)).groupby('date').in_top.any().mean())
    primary['method']=summary['method'];winners['method']=summary['method']
    return summary,primary,winners

def paired_lift(top_daily,base_daily,method,k,seed=2401):
    a=top_daily[(top_daily.method==method)&(top_daily.split=='VALIDATION')&(top_daily.k==k)&
        (top_daily.horizon=='2_5d')&(top_daily.threshold==10)]
    b=base_daily[(base_daily.split=='VALIDATION')&(base_daily.horizon=='2_5d')&(base_daily.threshold==10)]
    p=a.merge(b,on='date',suffixes=('_rank','_base'),validate='one_to_one').sort_values('date')
    diff=(p.daily_hit_rate_pct_rank-p.daily_hit_rate_pct_base).to_numpy();n=len(diff)
    rng=np.random.default_rng(seed);means=[]
    for _ in range(2000):
        start=rng.integers(0,n-5+1,size=int(np.ceil(n/5)))
        idx=np.concatenate([np.arange(s,s+5) for s in start])[:n]
        means.append(float(np.nanmean(diff[idx])))
    return dict(method=method,k=k,paired_days=int(np.isfinite(diff).sum()),mean_daily_lift_pp=float(np.nanmean(diff)),
        lower95_pp=float(np.quantile(means,.025)),upper95_pp=float(np.quantile(means,.975)),block_sessions=5,replicates=2000,seed=seed)

def main():
    guard_cached_artifact(ROOT/'data/research/phase23c/stock_days.parquet')
    guard_cached_artifact(LOCAL/'outcomes.parquet')
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    spec=contract()
    before=json.loads((OUT/'protected_before.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in before.items()),'PROTECTED_INPUT_CHANGED'
    data=pd.read_parquet(ROOT/'data/research/phase23c/stock_days.parquet')
    outcomes=pd.read_parquet(LOCAL/'outcomes.parquet')
    pd.testing.assert_frame_equal(data[KEYS],outcomes[KEYS])
    truth=outcomes.merge(data[KEYS+['label_valid']],on=KEYS,validate='one_to_one')
    decisions=decision_view(data)
    # Causal observed first-hit dates, not full-history group minima backfilled.
    history=decisions[KEYS+['strategy_'+r for r in 'ABCDEFGHIJ']+['strategy_count','candidate','split']].copy()
    symbols=pd.MultiIndex.from_frame(decisions[['exchange','symbol']])
    calendar=pd.DatetimeIndex(sorted(data.date.unique()))
    for rule in list('ABCDEFGHIJ')+['ANY']:
        yes=decisions.candidate if rule=='ANY' else decisions['strategy_'+rule]
        pos=pd.Series(np.where(yes,decisions.market_pos,np.inf),index=decisions.index).groupby(symbols,sort=False).cummin()
        first=pd.Series(pd.NaT,index=decisions.index,dtype='datetime64[ns]');finite=np.isfinite(pos)
        first.loc[finite]=calendar[pos[finite].astype(int)].to_numpy()
        history[f'strategy_{rule}_first_observed_hit_date']=first
        assert (first[finite]<=decisions.loc[finite,'date']).all()
    history['hit_time_precision']='DAILY_CLOSE_OBSERVATION_ONLY_INTRADAY_TIME_UNKNOWN'
    history.to_parquet(LOCAL/'strategy_history.parquet',index=False)
    discovery=decisions[decisions.split.eq('DISCOVERY')]
    weights=fit_strategy_weights(data[data.split.eq('DISCOVERY')])
    fit_outcomes=truth.loc[discovery.index].copy();fit_outcomes['split']='DISCOVERY'
    model=fit_model(discovery,fit_outcomes,weights)
    model['fit_label_end_date']=str(calendar[79].date());model['validation_start_date']=str(calendar[80].date())
    assert model['fit_label_end_date']<model['validation_start_date']
    dump(OUT/'frozen_ranking_model.json',model);model_hash=sha(OUT/'frozen_ranking_model.json')
    pd.DataFrame([dict(feature=n,**{k:v for k,v in d.items() if k!='knots'}) for n,d in model['features'].items()]).to_csv(OUT/'feature_calibration.csv',index=False)
    scored=score_rows(decisions,model);scored.to_parquet(LOCAL/'daily_scores.parquet',index=False)
    split_dates={s:pd.DatetimeIndex(sorted(data.loc[data.split.eq(s),'date'].unique())) for s in ('DISCOVERY','VALIDATION')}
    assert len(split_dates['DISCOVERY'])==75 and len(split_dates['VALIDATION'])==35
    baseline,base_daily,counts=candidate_baseline(decisions,truth,split_dates)
    baseline.to_csv(OUT/'unranked_candidate_baseline.csv',index=False);base_daily.to_csv(OUT/'unranked_candidate_daily.csv',index=False)
    counts.to_csv(OUT/'candidate_count_summary.csv',index=False)
    decisions.groupby(['split','date']).candidate.sum().rename('candidate_count').to_csv(OUT/'daily_candidate_counts.csv')
    ranked_all={};disc_metrics=[]
    # Select using Discovery before evaluating ANY Validation ranking outcomes.
    for method in METHODS:
        ranked=rank_rows(scored,method);ranked_all[method]=ranked
        ranked.to_parquet(LOCAL/f'daily_ranking_{method}.parquet',index=False)
        m,_,_=evaluate_top(ranked,truth,{'DISCOVERY':split_dates['DISCOVERY']});disc_metrics.append(m)
    selected,selection=choose_method(pd.concat(disc_metrics,ignore_index=True))
    selection.to_csv(OUT/'discovery_method_selection.csv',index=False)
    dump(OUT/'frozen_method_selection.json',dict(selected_method=selected,selection_split='DISCOVERY',model_sha256=model_hash,
        selection_criterion=spec['selection']))
    frozen_selection_hash=sha(OUT/'frozen_method_selection.json')
    metrics=[];dailies=[];tops=[];opps=[];winners=[];co=[]
    for method,ranked in ranked_all.items():
        m,d,t=evaluate_top(ranked,truth,split_dates);metrics.append(m);dailies.append(d);tops.append(t)
        o,w,c=opportunity(ranked,truth,decisions,split_dates['VALIDATION']);opps.append(o);winners.append(w);co.append(c)
    metrics=pd.concat(metrics,ignore_index=True);daily=pd.concat(dailies,ignore_index=True);top=pd.concat(tops,ignore_index=True)
    metrics.to_csv(OUT/'ranking_metrics.csv',index=False);daily.to_csv(OUT/'daily_ranking_metrics.csv',index=False)
    top.to_parquet(LOCAL/'top20_with_expost_audit.parquet',index=False)
    for k in KS:top[top['rank']<=k].to_csv(OUT/f'daily_top{k}.csv',index=False)
    pd.DataFrame(opps).to_csv(OUT/'opportunity_cost_summary.csv',index=False)
    pd.concat(winners).to_csv(OUT/'daily_market_winner_rank.csv',index=False)
    pd.concat(co).to_csv(OUT/'daily_market_co_winners.csv',index=False)
    execution=[]
    for (method,split),g in top[top.split.isin(split_dates)].groupby(['method','split']):
        for k in KS:
            p=g[g['rank']<=k]
            execution.append(dict(method=method,split=split,k=k,selected=len(p),
                next_gap_ge5_count=int(p.outcome_gap_ge5pct.fillna(False).sum()),
                next_one_price_up_count=int(p.outcome_next_day_one_price_up.fillna(False).sum()),
                next_bar_missing_count=int(p.outcome_next_day_gap.isna().sum()),
                fill_proven=False,expost_flags_used_in_ranking=False))
    pd.DataFrame(execution).to_csv(OUT/'expost_execution_flags.csv',index=False)
    lifts=[paired_lift(daily,base_daily,selected,k) for k in (3,5)];pd.DataFrame(lifts).to_csv(OUT/'selected_method_paired_lift.csv',index=False)
    sel=metrics[(metrics.method==selected)&(metrics.horizon=='2_5d')&(metrics.threshold==10)]
    val=sel[sel.split=='VALIDATION'].set_index('k');disc=sel[sel.split=='DISCOVERY'].set_index('k')
    base=float(baseline[(baseline.split=='VALIDATION')&(baseline.horizon=='2_5d')&(baseline.threshold==10)].hit_rate_pct.iloc[0])
    effective=all(val.loc[k,'hit_rate_pct']>=base+5 and val.loc[k,'label_coverage_pct']>=70 and lifts[i]['lower95_pp']>0 for i,k in enumerate((3,5)))
    decay=[dict(k=k,discovery_hit10_pct=float(disc.loc[k,'hit_rate_pct']),validation_hit10_pct=float(val.loc[k,'hit_rate_pct']),
        hit10_delta_pp=float(val.loc[k,'hit_rate_pct']-disc.loc[k,'hit_rate_pct']),
        daily_success_delta_pp=float(val.loc[k,'daily_success_lower_pct']-disc.loc[k,'daily_success_lower_pct'])) for k in KS]
    pd.DataFrame(decay).to_csv(OUT/'selected_ranking_decay.csv',index=False)
    cov=[]
    for split,g in data.groupby('split'):
        mature=int(g.future5_complete.sum());eligible=int(g.full_research_eligible.sum())
        cov.append(dict(split=split,raw_stock_days=len(g),complete_label_stock_days=mature,research_eligible_stock_days=eligible,
            research_coverage_pct=100*eligible/mature if mature else None))
    pd.DataFrame(cov).to_csv(OUT/'coverage.csv',index=False)
    assert sha(OUT/'frozen_ranking_model.json')==model_hash and sha(OUT/'frozen_method_selection.json')==frozen_selection_hash
    assert all(sha(ROOT/p)==h for p,h in before.items()),'PROTECTED_INPUT_CHANGED'
    health=dict(status='PHASE24_RANKING_BASELINE_COMPLETE' if effective else 'PHASE24_RANKING_NOT_EFFECTIVE',
        scope=spec['scope'],selected_method=selected,selected_on='DISCOVERY',model_sha256=model_hash,
        selection_sha256=frozen_selection_hash,candidate_counts=counts.to_dict('records'),
        bse_candidate_count=0,
        validation_same_unit_candidate_hit10_pct=base,decay=decay,paired_lift=lifts,coverage=cov,
        effectiveness=spec['effectiveness'],primary_target=spec['primary_target'],
        feature_lookahead_count='PENDING_INDEPENDENT_VERIFICATION',production_unchanged=True,thresholds_unchanged=True,
        prior_validation_seen=True,pristine_holdout=False,amount_used=False,ML_entered=False,commit=False,push=False,
        protected_file_checks=len(before),fit_date_end=model['fit_date_max'],fit_label_end_date=model['fit_label_end_date'])
    dump(OUT/'research_health.json',health)
    print(json.dumps(dict(status=health['status'],selected_method=selected)),flush=True)

if __name__=='__main__':main()
