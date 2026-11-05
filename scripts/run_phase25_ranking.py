"""Frozen two-configuration expanding walk-forward, full research universe."""
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_phase21_research import ROOT,sha
from scripts.phase24a_prepare_outcomes import safe_dump
from src.research.ml_ranking import (CONFIGS,COMMON,FEATURE_NAMES,QUALITY,make_features,
    walk_forward,training_mask,fit_binary,rank_predictions,training_digest,permutation_training)
from src.research.daily_ranking import KEYS
from src.universe_policy import predictive_universe, report_audit, guard_cached_artifact

OUT=ROOT/'outputs/phase25';LOCAL=ROOT/'data/research/phase25'

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    snapshot={p:sha(ROOT/p) for p in tracked if (ROOT/p).is_file()}
    contract=dict(scope='RESEARCH_ONLY_NOT_TRADABLE_BACKTEST',model_library='lightgbm==4.6.0',
        common=COMMON,models=CONFIGS,primary_model='M1',model_selection='NONE_M2_DIAGNOSTIC_ONLY',
        features=FEATURE_NAMES,first_test='2026-08-05',block_size=20,purge=5,requested_sessions=250,
        target='max(high[t+2:t+5])/close[t]-1 >=10%; integer cents; censor ANY CA in (t,t+5]',
        universe='UNCHANGED_PHASE23C_feature_eligibility; NO_SIGNAL_GATE; label masks applied AFTER ranking',
        R3='Frozen Phase24 weights; ALL score descending/hash ties; GATE only A-J hits; no refit',
        R4='Frozen Phase24 formula, weights, quantiles and directions; train-derived transform on training rows',
        first_test_rationale='Frozen R2-R4 calibration anchors end July28, labels end Aug4. No earlier OOF allowed.',
        negative_sampling='NONE; all clean eligible training rows; unit sample weights',
        importance='Training gain plus within-date training-only permutation; not used for selection',
        success='M1 TOP3 AND TOP5 lift >=5pp vs R3_ALL pooled; both positive in EVERY block; '
                'market-winner TOP20 AND TOP50 recall higher pooled; TOP5 episode20 AND episode30 capture higher pooled; '
                'both TOP3/5 paired daily lift 5-session bootstrap lower95>0; >=2 blocks, all block universe and selected coverage>=70%',
        pristine_holdout=False,old_validation_seen=True,OOF_not_prospective_holdout=True,
        feature_fit_used_test=False,amount_used=False,production_changed=False,
        phase23_commit='773e5fc',phase24_commit='1bae237',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_ranking_sha256=sha(ROOT/'outputs/phase24a/frozen_ranking_model.json'))
    p=OUT/'analysis_contract.json'
    if p.exists():assert json.loads(p.read_text())==contract,'CONTRACT_CHANGED'
    else:
        safe_dump(p,contract);safe_dump(OUT/'protected_before.json',snapshot)
    return contract

def bootstrap(daily,method='M1'):
    rows=[]
    for k in (3,5):
        a=daily[(daily.method==method)&(daily.k==k)&(daily.threshold==10)]
        b=daily[(daily.method=='R3_ALL')&(daily.k==k)&(daily.threshold==10)]
        # Worker daily table has one record per date/method/k/threshold.
        p=a.merge(b,on='date',suffixes=('_ml','_r3')).sort_values('date')
        diff=(100*p.hits_ml/p.evaluable_ml-100*p.hits_r3/p.evaluable_r3).dropna().to_numpy()
        rng=np.random.default_rng(2501);means=[];n=len(diff)
        for _ in range(2000):
            starts=rng.integers(0,n-5+1,size=int(np.ceil(n/5)))
            idx=np.concatenate([np.arange(s,s+5) for s in starts])[:n]
            means.append(diff[idx].mean())
        rows.append(dict(method=method,k=k,paired_days=n,mean_daily_lift_pp=float(diff.mean()),
            lower95_pp=float(np.quantile(means,.025)),upper95_pp=float(np.quantile(means,.975)),block_sessions=5,replicates=2000))
    return pd.DataFrame(rows)

def main():
    guard_cached_artifact(LOCAL/'stock_days.parquet')
    contract=freeze()
    from src.research.ml_ranking_metrics import evaluate_oof
    import lightgbm as lgb
    assert lgb.__version__=='4.6.0'
    frozen=json.loads((ROOT/'outputs/phase24a/frozen_ranking_model.json').read_text())
    data=pd.read_parquet(LOCAL/'stock_days.parquet')
    data=predictive_universe(data)
    dates=pd.DatetimeIndex(sorted(data.date.unique()));assert len(dates)==250
    blocks=walk_forward(dates,contract['first_test'])
    assert all(pd.Timestamp(b['test_start_date'])>pd.Timestamp('2026-08-04') for b in blocks)
    safe_dump(OUT/'walk_forward_blocks.json',blocks)
    matrices=[];scores=[]
    for start in range(0,len(data),75000):
        x,s=make_features(data.iloc[start:start+75000],frozen)
        matrices.append(x.astype('float32'));scores.append(s[KEYS+['tie_key','R3','candidate']])
    x=pd.concat(matrices);s=pd.concat(scores);del matrices,scores;gc.collect()
    assert x.index.equals(data.index)
    print(json.dumps(dict(prepared_ml_rows=len(x),features=len(x.columns),blocks=len(blocks))),flush=True)
    oof=[];fitrecords=[];gains=[];permutations=[]
    for b in blocks:
        tr=training_mask(data,b)
        test=(data.market_pos.ge(b['test_start'])&data.market_pos.lt(b['test_stop'])&data.full_any_strategy_eligible)
        y=data.loc[tr,'outcome_hit10_2_5d'].astype(int)
        assert len(y)>0 and data.loc[tr,'market_pos'].max()+5<b['test_start']
        training_x=x.loc[tr];test_x=x.loc[test]
        digest=training_digest(training_x,y,data.loc[tr,KEYS])
        metadata=data.loc[test,KEYS+['market_pos']].copy();metadata['tie_key']=s.loc[test,'tie_key'];metadata['block']=b['block']
        predicted={}
        for modelname in CONFIGS:
            print(json.dumps(dict(training=b['block'],model=modelname,train=len(y),test=int(test.sum()),positives=int(y.sum()))),flush=True)
            model=fit_binary(training_x,y,modelname,metadata=data.loc[tr,KEYS])
            model_path=LOCAL/f'{b["block"]}_{modelname}.txt';model.save_model(str(model_path))
            predicted[modelname]=model.predict(test_x,num_threads=4)
            gain=model.feature_importance(importance_type='gain')
            gains.extend(dict(block=b['block'],model=modelname,feature=f,gain=float(g),gain_share=float(g/gain.sum()) if gain.sum() else 0.) for f,g in zip(FEATURE_NAMES,gain))
            perm=permutation_training(model,training_x,y,data.loc[tr,'date'],seed=2501)
            perm['block']=b['block'];perm['model']=modelname;permutations.append(perm)
            fitrecords.append(b|dict(model=modelname,train_rows=len(y),train_positives=int(y.sum()),train_negatives=int((y==0).sum()),
                negative_rows_removed=0,OOF_rows=int(test.sum()),OOF_evaluable=int((test&data.label_valid).sum()),
                training_digest=digest,model_sha256=sha(model_path),model_path=str(model_path.relative_to(ROOT)),
                eligible_noncandidates=int((test&~s.candidate).sum()),max_training_label_pos=int(data.loc[tr,'market_pos'].max()+5)))
            print(json.dumps(dict(finished=b['block'],model=modelname)),flush=True)
        ranked=rank_predictions(metadata,predicted,s.loc[test,'R3'],s.loc[test,'candidate'])
        truthcols=['label_valid','future5_complete','LABEL_CROSSES_CA']+['full_signal_'+r for r in 'ABCDEFGHIJ']+[
            f'outcome_hit{n}_{h}' for n in (10,20,30) for h in ('2_5d','1_5d')]+['outcome_max_return_1_5d']
        ranked=ranked.join(data.loc[test,truthcols]);oof.append(ranked)
        pd.DataFrame(fitrecords).to_csv(OUT/'fit_audit.csv',index=False)
        del model,training_x,test_x;gc.collect()
    joined=pd.concat(oof).sort_values(['date','tie_key']).reset_index(drop=True)
    joined.to_parquet(LOCAL/'oof_predictions.parquet',index=False)
    assert not joined.duplicated(KEYS).any()
    pd.DataFrame(gains).to_csv(OUT/'gain_importance_by_fold.csv',index=False)
    pd.concat(permutations).to_csv(OUT/'permutation_importance_by_fold.csv',index=False)
    g=pd.DataFrame(gains).groupby(['model','feature']).agg(gain=('gain','sum'),mean_gain_share=('gain_share','mean')).reset_index()
    p=pd.concat(permutations).groupby(['model','feature']).permutation_logloss_increase.mean().reset_index()
    importance=g.merge(p,on=['model','feature']).sort_values(['model','mean_gain_share'],ascending=[True,False])
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    result=evaluate_oof(joined)
    for name,frame in result.items():frame.to_csv(OUT/(name+'.csv'),index=False)
    boot=bootstrap(result['daily']);boot.to_csv(OUT/'paired_lift.csv',index=False)
    (joined[joined[[f'rank_{m}' for m in ('M1','M2','R3_ALL','R3_GATE')]].le(50).any(axis=1)]
        .to_csv(OUT/'daily_top50_union.csv',index=False))
    safe_dump(OUT/'run_health.json',dict(bse_candidate_count=report_audit(joined)['bse_candidate_count'],scope=contract['scope'],status='PENDING_INDEPENDENT_VERIFICATION',
        primary_model='M1',model_library=lgb.__version__,research_days=len(dates),OOF_days=joined.date.nunique(),blocks=len(blocks),
        OOF_stock_days=len(joined),OOF_evaluable_stock_days=int(joined.label_valid.sum()),
        OOF_noncandidate_stock_days=int((~joined[['full_signal_'+r for r in 'ABCDEFGHIJ']].any(axis=1)).sum()),
        OOF_coverage_pct=100*joined.label_valid.mean(),features=len(FEATURE_NAMES),
        data_sha256=sha(LOCAL/'stock_days.parquet'),predictions_sha256=sha(LOCAL/'oof_predictions.parquet'),
        amount_used=False,production_changed=False,phase25_committed=False,pristine_holdout=False))
    print('PHASE25_MODELS_AND_OOF_WRITTEN',flush=True)

if __name__=='__main__':main()
