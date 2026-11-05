"""Transparent daily cross-sectional scores. No outcome input at scoring time."""
import hashlib
import numpy as np
import pandas as pd
from src.universe_policy import predictive_universe, report_audit

METHODS=('R1','R2','R3','R4')
KS=(1,3,5,10,20)
KEYS=['exchange','symbol','date']
GROUPS={
    'trend':['close_vs_ma10','ma10_slope','ret_10d'],
    'acceleration':['price_acceleration_3','price_acceleration_5'],
    'volume':['volume_ratio_5','volume_acceleration_3'],
    'compression':['range_compression_5','volatility_5'],
    'recovery':['recovery_from_drawdown_3','distance_from_5d_low','lower_shadow_ratio'],
    'momentum':['ret_3d','ret_5d','close_location'],
}
FEATURES=tuple(dict.fromkeys(n for group in GROUPS.values() for n in group))

def decision_view(data):
    """Explicit allowlist: labels, publication, next-day conditions cannot pass."""
    data=predictive_universe(data)
    cols=KEYS+['split','market_pos','full_any_strategy_eligible']
    cols += [n for f in FEATURES for n in (f,f+'__valid')]
    out=data[cols].copy()
    for rule in 'ABCDEFGHIJ':out['strategy_'+rule]=data['full_signal_'+rule].astype(bool)
    out['strategy_count']=out[['strategy_'+r for r in 'ABCDEFGHIJ']].sum(axis=1)
    out['candidate']=out.strategy_count.gt(0)
    out['tie_key']=[hashlib.sha256(('2401|'+str(d)[:10]+'|'+e+'|'+s).encode()).hexdigest()
        for e,s,d in out[KEYS].itertuples(index=False,name=None)]
    return out

def percentile(values,knots):
    """Monotone Discovery-only quantile mapping, midpoint for repeated knots."""
    q=np.asarray(knots,float)
    unique,inverse=np.unique(q,return_inverse=True)
    probabilities=np.linspace(0,1,len(q))
    mid=np.bincount(inverse,weights=probabilities)/np.bincount(inverse)
    return np.interp(np.asarray(values,float),unique,mid,left=0.,right=1.)

def fit_model(discovery,outcomes,weights):
    report_audit(discovery)
    if discovery.empty or not discovery.split.eq('DISCOVERY').all():raise ValueError('DISCOVERY_ONLY_FIT')
    if len(discovery)!=len(outcomes) or not discovery.index.equals(outcomes.index):raise ValueError('OUTCOME_ALIGNMENT')
    if not outcomes.split.eq('DISCOVERY').all():raise ValueError('DISCOVERY_ONLY_OUTCOMES')
    model=dict(bse_candidate_count=0,methods=list(METHODS),groups=GROUPS,weights=weights,features={},
        fit_split='DISCOVERY',fit_rows=len(discovery),fit_date_min=str(discovery.date.min()),fit_date_max=str(discovery.date.max()),
        quantiles=101,missing_feature_policy='Keep raw NaN. Missing centered-score contribution is zero; never fabricate feature values.',
        R1='sum(strategy_A..strategy_J)',R2='sum(strategy_r * discovery_episode_precision10_r)',
        R3='sum(strategy_r * mean(discovery_episode_precision20_r,discovery_episode_precision30_r))',
        R4='0.5 * R2/sum(all10_precision_weights) + 0.5 * (1 + mean(six_centered_group_scores))/2',
        ranking='feature-eligible rows; candidate-first, score descending, fixed outcome-blind SHA256 tie key',
        direction='sign(mean(Discovery percentile | strict2..5 hit10) - mean(percentile | no hit10)); zero if unidentifiable')
    for name in FEATURES:
        valid=discovery[name+'__valid']&np.isfinite(discovery[name])&discovery.full_any_strategy_eligible
        values=discovery.loc[valid,name].to_numpy(float)
        if not len(values):
            model['features'][name]=dict(knots=[0.]*101,direction=0,quantile_rows=0,direction_rows=0,mean_hit=None,mean_miss=None)
            continue
        knots=np.quantile(values,np.linspace(0,1,101)).tolist()
        known=valid&outcomes.label_valid&outcomes.outcome_hit10_2_5d.notna()
        p=percentile(discovery.loc[known,name],knots)
        hits=outcomes.loc[known,'outcome_hit10_2_5d'].astype(bool).to_numpy()
        yes=float(p[hits].mean()) if hits.any() else None
        no=float(p[~hits].mean()) if (~hits).any() else None
        direction=int(np.sign(yes-no)) if yes is not None and no is not None else 0
        model['features'][name]=dict(knots=knots,direction=direction,quantile_rows=int(valid.sum()),direction_rows=int(known.sum()),mean_hit=yes,mean_miss=no)
    return model

def score_rows(decisions,model):
    """No labels or ex-post execution columns are read or returned."""
    decisions=predictive_universe(decisions)
    out=decisions[KEYS+['split','market_pos','full_any_strategy_eligible','candidate','strategy_count','tie_key']+
        ['strategy_'+r for r in 'ABCDEFGHIJ']].copy()
    out['R1']=out.strategy_count.astype(float)
    for method,wcol in [('R2','precision10'),('R3','big_winner')]:
        out[method]=sum(out['strategy_'+r].astype(float)*float(model['weights'][r][wcol]) for r in 'ABCDEFGHIJ')
    components={}
    for name in FEATURES:
        spec=model['features'][name]
        known=decisions[name+'__valid']&np.isfinite(decisions[name])
        x=np.full(len(decisions),np.nan)
        x[known]=spec['direction']*(2*percentile(decisions.loc[known,name],spec['knots'])-1)
        components[name]=x
        out[name+'__score_valid']=known
    groups=[]
    for group,names in GROUPS.items():
        arr=np.column_stack([components[n] for n in names])
        # Fixed denominators: unavailable features neither earn a high score nor
        # delete the stock-day. NaN raw features remain NaN in the source table.
        centered=np.nan_to_num(arr,nan=0.).sum(axis=1)/len(names)
        out[group+'_score']=(centered+1)/2
        out[group+'_valid_feature_count']=np.isfinite(arr).sum(axis=1)
        groups.append(centered)
    out['feature_score']=(np.column_stack(groups).mean(axis=1)+1)/2
    total=sum(float(model['weights'][r]['precision10']) for r in 'ABCDEFGHIJ')
    out['strategy_score']=out.R2/total if total else 0.
    out['R4']=.5*out.strategy_score+.5*out.feature_score
    return out

def rank_rows(scored,method):
    if method not in METHODS:raise ValueError('UNKNOWN_RANKING_METHOD')
    scored=predictive_universe(scored)
    out=scored[scored.full_any_strategy_eligible].copy()
    out=out.sort_values(['date','candidate',method,'tie_key'],ascending=[True,False,False,True],kind='stable')
    out['rank']=out.groupby('date').cumcount()+1
    grouped=out.groupby(['date','candidate',method],dropna=False)['rank']
    out['tie_best_rank']=grouped.transform('min');out['tie_worst_rank']=grouped.transform('max')
    out['method']=method
    out['bse_candidate_count']=report_audit(out)['bse_candidate_count']
    return out

def choose_method(discovery_metrics):
    if not discovery_metrics.split.eq('DISCOVERY').all():raise ValueError('DISCOVERY_ONLY_SELECTION')
    rows=[]
    for method in METHODS:
        d=discovery_metrics[(discovery_metrics.method==method)&(discovery_metrics.horizon=='2_5d')&
            (discovery_metrics.threshold==10)].set_index('k')
        qualified=all(d.loc[k,'label_coverage_pct']>=70 for k in (3,5))
        rows.append(dict(method=method,qualified=qualified,selection_score=(d.loc[3,'hit_rate_pct']+d.loc[5,'hit_rate_pct'])/2,
            daily_success_top3=d.loc[3,'daily_success_lower_pct'],tie_priority=METHODS.index(method)))
    table=pd.DataFrame(rows).sort_values(['qualified','selection_score','daily_success_top3','tie_priority'],ascending=[False,False,False,True])
    if not table.iloc[0].qualified:raise ValueError('DISCOVERY_TOP_K_COVERAGE_BLOCKED')
    return str(table.iloc[0].method),table
