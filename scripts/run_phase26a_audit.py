"""Frozen OOF reevaluation only. No training imports, model scoring or trading."""
import json
import subprocess
import numpy as np
import pandas as pd
from scripts.run_phase21_research import ROOT,sha
from scripts.phase24a_prepare_outcomes import safe_dump
from src.research.objective_tradability import (KEYS,METHODS,classify_entries,evaluate_selection,
    fallback_open_only,decile_table,opportunity_table)
from src.universe_policy import guard_cached_artifact

OUT=ROOT/'outputs/phase26a';LOCAL=ROOT/'data/research/phase26a'
RULE_SOURCE='https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml'

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    spec=dict(scope='RESEARCH_ONLY_NOT_TRADABLE_BACKTEST',predictions='Phase25 frozen OOF only; no model loading/retraining/reselection',
        targets={'A':'P(max high[t+2:t+5]/close[t]-1 >=10%)','B':'>=20% same horizon','C':'continuous max high[t+2:t+5]/close[t]-1'},
        deciles='Per-date all eligible M1 ranking: D10=top tenth. Labels excluded AFTER binning.',
        entry_time='Earliest permitted execution t+1 normal continuous auction; daily open is only an auction/first-trade price benchmark, NOT a demonstrated continuous-auction fill',
        exit_horizon='Original signal t+2..t+5; do not shift to t+3..t+6',
        scenarios={'E1':'next open benchmark, every observed open including inaccessible/unknown flags',
        'E2':'E1 but skip gap>=5%, not counted as failed trade; one-price fill remains unknown',
        'E3':'EX_POST_DAILY_PROXY_ONLY: exclude any one-price day, nominal-board limit-up open, zero/missing day volume or missing OHLC; actual fill never confirmed'},
        gap_buckets='negative=NORMAL_ENTRY bucket; [0,3),[3,5),[5,8],>8 percent; integer-cent comparisons',
        limit_status='nominal board .10/.20/.30 price proxy only; historical ST/special limits unqualified; never call an official limit fact',
        CA='Keep frozen Phase25 label_valid, no relaxation for entry-at-ex-date and no future-label selection',
        fallback='PRIMARY OPEN_ONLY: original ranking order, observed next_open, gap<5%, not nominal upper-limit open; first3, no labels/high/low/close/volume in selection; all eligible universe scanned, no cap',
        actionable='Only conditional open-price daily proxies can be computed. Actual continuous-auction success NOT_IDENTIFIABLE without intraday/queue evidence',
        opportunity='Strict 2..5, original topK then label cleaning; ratio daily bestK/market best, only positive denominator; aggregate arithmetic mean and median',
        decision='Preserve prediction-edge and extreme-recall conclusions separately. If M1 close TOP3/5 improves in both blocks but market-max TOP20/50 is worse than R3, DUAL_OBJECTIVE_REQUIRED; no live-actionability confirmation from daily bars',
        production_changed=False,amount_used=False,commit=False,push=False,official_rule_reference=RULE_SOURCE)
    path=OUT/'analysis_contract.json'
    if path.exists():assert json.loads(path.read_text())==spec,'PHASE26_CONTRACT_CHANGED'
    else:
        safe_dump(path,spec)
        paths=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
        paths.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
        for directory in ('outputs/phase25','data/research/phase25','outputs/phase24a'):
            paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/directory).rglob('*') if p.is_file())
        paths={p for p in paths if 'phase26' not in p and not p.endswith('objective_tradability.py')}
        safe_dump(OUT/'protected_before.json',{p:sha(ROOT/p) for p in sorted(paths) if (ROOT/p).is_file()})
    return spec

def main():
    guard_cached_artifact(ROOT/'data/research/phase25/oof_predictions.parquet')
    guard_cached_artifact(LOCAL/'entry_bars.parquet')
    freeze()
    frozen=pd.read_parquet(ROOT/'data/research/phase25/oof_predictions.parquet')
    bars=pd.read_parquet(LOCAL/'entry_bars.parquet')
    pd.testing.assert_frame_equal(frozen[KEYS],bars[KEYS])
    data=classify_entries(frozen.merge(bars,on=KEYS,validate='one_to_one'))
    assert len(data)==len(frozen)
    data.to_parquet(LOCAL/'audit_stock_days.parquet',index=False)
    deciles,bins=decile_table(data);deciles.to_csv(OUT/'score_deciles.csv',index=False)
    bins.to_parquet(LOCAL/'decile_membership.parquet',index=False)
    metrics=[];daily=[]
    for method in METHODS:
        for k in (1,3,5):
            m,d=evaluate_selection(data,data['rank_'+method].le(k),method,k)
            metrics.append(m);daily.append(d)
        selected=fallback_open_only(data,method,3)
        data['fallback_'+method]=selected
        m,d=evaluate_selection(data,selected,method,3,'FALLBACK_OPEN_ONLY')
        metrics.append(m);daily.append(d)
    metrics=pd.concat(metrics,ignore_index=True);daily=pd.concat(daily,ignore_index=True)
    metrics.to_csv(OUT/'entry_metrics.csv',index=False);daily.to_csv(OUT/'daily_entry_metrics.csv',index=False)
    choice_cols=KEYS+['block','rank_M1','rank_R3_ALL','next_day_date','next_open','next_day_gap','pattern_class','intraday_entry_status',
        'label_valid','entry_hit10','entry_hit20','entry_hit30','fallback_M1','fallback_R3_ALL']
    data[data.fallback_M1|data.fallback_R3_ALL][choice_cols].to_csv(OUT/'fallback_selections.csv',index=False)
    opportunity_daily,opportunity=opportunity_table(data)
    opportunity_daily.to_csv(OUT/'opportunity_daily.csv',index=False);opportunity.to_csv(OUT/'opportunity_summary.csv',index=False)
    top=data[data.rank_M1.le(3)].copy()
    # Exact rows, not outcome-selected examples. Each price pattern is retained.
    top[top.one_price_up].to_csv(OUT/'one_price24_case_audit.csv',index=False)
    top[top.gap_ge5].to_csv(OUT/'high_gap65_case_audit.csv',index=False)
    categories=[];fallback=[]
    for method in METHODS:
        for k in (1,3,5):
            part=data[data['rank_'+method].le(k)]
            for kind in ('gap_bucket','pattern_class'):
                for value,n in part[kind].value_counts().items():categories.append(dict(method=method,k=k,classification=kind,value=value,count=int(n)))
    pd.DataFrame(categories).to_csv(OUT/'candidate_classifications.csv',index=False)
    for date,d in data.groupby('date'):
        old=d[d.rank_M1.le(3)];picked=d[d.fallback_M1]
        top_remaining=old.E2_entry_eligible.sum()
        fallback.append(dict(date=date,block=d.block.iloc[0],top3_high_gap_count=int(old.gap_ge5.sum()),
            rank1_high_gap=bool(old.loc[old.rank_M1.eq(1),'gap_ge5'].iloc[0]),
            original_top3_no_chase_remaining=int(top_remaining),has_original_second_or_third=bool(old[old.rank_M1.gt(1)].E2_entry_eligible.any()),
            replacement_count=int(picked.rank_M1.gt(3).sum()),fallback_count=len(picked),max_original_rank=int(picked.rank_M1.max()) if len(picked) else None,
            fallback_one_price_unknown=int(picked.one_price_day.sum()),fallback_label_censored=int((~picked.label_valid).sum())))
    pd.DataFrame(fallback).to_csv(OUT/'top3_fallback_daily.csv',index=False)
    for name in ('gap_bucket','pattern_class'):
        rows=[]
        for value,g in data[data.rank_M1.le(3)].groupby(name):
            # The full selected-row file already contains labels; summary is
            # kept concise without conditioning selection on its outcome.
            rows.append(dict(category=value,count=len(g),close_hit10=int((g.label_valid&g.outcome_hit10_2_5d.fillna(False)).sum()),
                open_hit10=int((g.label_valid&g.entry_hit10.fillna(False)).sum()),label_evaluable=int(g.label_valid.sum())))
        pd.DataFrame(rows).to_csv(OUT/(name+'_TOP3_outcomes.csv'),index=False)
    safe_dump(OUT/'run_health.json',dict(status='PENDING_INDEPENDENT_VERIFICATION',rows=len(data),days=data.date.nunique(),
        top3_high_gap_count=int(top.gap_ge5.sum()),top3_one_price_up_count=int(top.one_price_up.sum()),
        actual_entry_confirmation='NOT_IDENTIFIABLE_WITH_DAILY_BARS',model_retrained=False,
        actual_next_day_data_used_as_t_feature=False,predictions_sha256=sha(ROOT/'data/research/phase25/oof_predictions.parquet')))
    print('PHASE26A_REEVALUATION_WRITTEN',flush=True)

if __name__=='__main__':main()
