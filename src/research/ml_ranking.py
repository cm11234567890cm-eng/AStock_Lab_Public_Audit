"""Fixed full-universe research models; no execution or label-based ranking gate."""
import hashlib
import numpy as np
import pandas as pd
from src.research.coverage_recovery import FEATURE_WINDOWS
from src.research.daily_ranking import decision_view, score_rows
from src.universe_policy import predictive_universe, predictive_mask, report_audit, require_training_identity

QUALITY = ['feature_valid_count', 'long_feature_valid'] + [f'ca_boundary_{n}' for n in (5,10,20,60)]
FEATURE_NAMES = (list(FEATURE_WINDOWS) + [n+'__valid' for n in FEATURE_WINDOWS] +
                 ['strategy_'+r for r in 'ABCDEFGHIJ'] + ['strategy_count','R1','R2','R3','R4'] + QUALITY)
CONFIGS = {
 'M1': dict(num_leaves=15,max_depth=4,min_data_in_leaf=200,learning_rate=.04,num_iterations=160,lambda_l2=10.),
 'M2': dict(num_leaves=31,max_depth=5,min_data_in_leaf=100,learning_rate=.035,num_iterations=240,lambda_l2=10.),
}
COMMON = dict(objective='binary',metric='binary_logloss',seed=2501,num_threads=4,
              deterministic=True,force_col_wise=True,feature_fraction=.8,bagging_fraction=1.,
              bagging_freq=0,verbosity=-1,feature_pre_filter=False,use_missing=True,zero_as_missing=False)

def make_features(data,frozen):
    """Frozen Discovery transform; callable without any labels or outcome fields.

    Its learned parameters are valid only for fits/tests AFTER their calibration
    period. Historical training rows are transformed with this training-derived
    map, not claimed to be contemporaneous historical predictions.
    """
    data=predictive_universe(data)
    view=decision_view(data)
    scores=score_rows(view,frozen)
    out={}
    for name in FEATURE_WINDOWS:
        valid=data[name+'__valid'].astype(bool)
        out[name]=data[name].astype(float).where(valid)
        out[name+'__valid']=valid.astype(float)
    for name in ['strategy_'+r for r in 'ABCDEFGHIJ']+['strategy_count','R1','R2','R3','R4']:
        out[name]=scores[name].astype(float)
    for name in QUALITY:out[name]=data[name].astype(float)
    matrix=pd.DataFrame(out,index=data.index)[FEATURE_NAMES].replace([np.inf,-np.inf],np.nan)
    assert not any(x in FEATURE_NAMES for x in ('symbol','date','market_pos','listing_date','label_valid','TARGET_2_5D_10'))
    return matrix,scores

def walk_forward(calendar,first_test,block_size=20,purge=5,horizon=5):
    dates=pd.DatetimeIndex(calendar).sort_values().unique()
    if purge<horizon:raise ValueError('PURGE_TOO_SHORT')
    first=int(dates.searchsorted(pd.Timestamp(first_test)))
    end=len(dates)-horizon
    if first-purge<1 or first>=end:raise ValueError('INSUFFICIENT_WALK_FORWARD_DATES')
    blocks=[]
    for number,start in enumerate(range(first,end,block_size),1):
        stop=min(start+block_size,end)
        blocks.append(dict(block=f'B{number}',train_start=0,train_stop=start-purge,
            test_start=start,test_stop=stop,train_start_date=str(dates[0].date()),
            train_end_date=str(dates[start-purge-1].date()),
            train_label_end_date=str(dates[start-purge-1+horizon].date()),
            test_start_date=str(dates[start].date()),test_end_date=str(dates[stop-1].date()),
            purge_sessions=purge,test_days=stop-start))
    return blocks

def training_mask(data,block):
    return (predictive_mask(data) & data.market_pos.ge(block['train_start']) & data.market_pos.lt(block['train_stop']) &
            data.full_any_strategy_eligible & data.label_valid & data.outcome_hit10_2_5d.notna())

def fit_binary(x,y,name,*,metadata=None):
    require_training_identity(x,metadata)
    import lightgbm as lgb
    if name not in CONFIGS:raise ValueError('UNFROZEN_MODEL')
    if list(x.columns)!=FEATURE_NAMES:raise ValueError('FEATURE_ALLOWLIST_MISMATCH')
    target=np.asarray(y,dtype=np.int32)
    if len(x)!=len(target) or set(np.unique(target))!={0,1}:raise ValueError('BINARY_TRAINING_REQUIRED')
    # Every eligible, labeled training negative is retained; no resampling.
    params=COMMON|CONFIGS[name]
    rounds=params.pop('num_iterations')
    return lgb.train(params,lgb.Dataset(x,label=target,feature_name=FEATURE_NAMES),num_boost_round=rounds)

def rank_predictions(metadata,probabilities,r3,gate):
    """Rows must be the SAME eligible universe, including unknown future labels."""
    out=metadata.copy()
    gate=pd.Series(np.asarray(gate,bool),index=metadata.index)
    out['score_R3_ALL']=np.asarray(r3,float)
    for name,p in probabilities.items():
        p=np.asarray(p,float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('INVALID_MODEL_PROBABILITY')
        out['score_'+name]=p
    out=predictive_universe(out)
    for name in list(probabilities)+['R3_ALL','R3_GATE']:
        mask=gate.loc[out.index].to_numpy() if name=='R3_GATE' else np.ones(len(out),bool)
        score='score_R3_ALL' if name=='R3_GATE' else 'score_'+name
        selected=out.loc[mask].sort_values(['date',score,'tie_key'],ascending=[True,False,True],kind='stable')
        ranks=selected.groupby('date').cumcount()+1
        out['rank_'+name]=ranks.reindex(out.index)
    out['bse_candidate_count']=report_audit(out)['bse_candidate_count']
    return out

def training_digest(x,y,metadata):
    h=hashlib.sha256()
    for frame in (x,pd.DataFrame({'target':np.asarray(y)}),metadata.reset_index(drop=True)):
        h.update(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes())
    h.update('|'.join(x.columns).encode())
    return h.hexdigest()

def permutation_training(model,x,y,dates,max_rows=20000,seed=2501):
    """Training-only diagnostic, never used to select features or configurations.

    Deterministic random subset is for importance computation only. Permutation
    within each date retains market-regime distributions. Values and their
    validity flags are permuted together to avoid artificial missingness pairs.
    """
    rng=np.random.default_rng(seed)
    indices=np.sort(rng.choice(len(x),min(len(x),max_rows),replace=False))
    a=x.iloc[indices].reset_index(drop=True);y=np.asarray(y)[indices]
    d=pd.Series(np.asarray(dates)[indices]);groups=list(d.groupby(d).groups.values())
    def loss(p):
        p=np.clip(p,1e-9,1-1e-9)
        return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
    base=loss(model.predict(a,num_threads=4))
    rows=[]
    for name in FEATURE_NAMES:
        b=a.copy();order=np.arange(len(a))
        for group in groups:order[group]=rng.permutation(group)
        cols=[name]
        if name in FEATURE_WINDOWS:cols.append(name+'__valid')
        for col in cols:b[col]=a[col].to_numpy()[order]
        rows.append(dict(feature=name,permutation_logloss_increase=loss(model.predict(b,num_threads=4))-base,
                         diagnostic_rows=len(a),basis='TRAIN_ONLY_WITHIN_DATE_NOT_SELECTION',paired_columns=';'.join(cols)))
    return pd.DataFrame(rows)
