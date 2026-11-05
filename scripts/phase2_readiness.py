"""Read-only Phase2 data gate: a current snapshot must never masquerade as PIT."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
AS_OF = '2026-09-30'  # Frozen source endpoint, not the machine's current date.
REQUIRED_GATES = (
    'historical_identity_intervals_verified',
    'historical_st_intervals_verified',
    'historical_suspension_and_tradability_verified',
    'historical_security_code_mapping_verified',
    'all_universe_corporate_action_coverage_verified',
    'official_calendar_2025_2026_verified',
)


def readiness(gates):
    """Unknown, missing, numeric truthiness and current-only evidence fail closed."""
    return [name for name in REQUIRED_GATES if gates.get(name) is not True]


def complete_label_dates(sessions, source_end, horizon=5):
    """Exchange-session horizon, never the next five available stock bars."""
    values = pd.DatetimeIndex(pd.to_datetime(sessions))
    if values.hasnans or values.has_duplicates or not values.is_monotonic_increasing:
        raise ValueError('Calendar must be unique and chronological')
    if horizon < 1:
        raise ValueError('Positive horizon required')
    values = values[values <= pd.Timestamp(source_end)]
    return values[:-horizon] if len(values) > horizon else values[:0]


def summarize(path):
    frame = pd.read_parquet(path)
    result = dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=len(frame),columns=list(frame.columns))
    if 'symbol' in frame:
        result['symbols'] = int(frame.symbol.nunique())
    if 'ex_date' in frame:
        dates = pd.to_datetime(frame.ex_date)
        result.update(first_ex_date=str(dates.min().date()),last_ex_date=str(dates.max().date()))
    for col in ['listing_status','bar_status','source','cash_basis_status','basis_verified']:
        if col in frame:
            result[col+'_counts'] = {str(k):int(v) for k,v in frame[col].fillna('NULL').astype(str).value_counts().items()}
    for col in ['listing_date','delisting_date','ex_reference_cash_dividend_per_share','basis_source']:
        if col in frame:result[col+'_nonnull'] = int(frame[col].notna().sum())
    return result


def main():
    def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
    frozen = load('outputs/phase19k_health.json')
    canonical = load('outputs/canonical_universe_health.json')
    gap = load('outputs/corporate_actions_gap_health.json')
    tables = [summarize(p) for p in sorted((ROOT/'data/normalized').glob('*.parquet'))]
    master = pd.read_parquet(ROOT/'data/normalized/security_master.parquet')
    # Inspect every local normalized Parquet schema, not just expected filenames.
    all_schemas = {str(p.relative_to(ROOT)):pq.read_schema(p).names
        for p in (ROOT/'data/normalized').rglob('*.parquet')}
    interval_candidates = [name for name,columns in all_schemas.items()
        if {'valid_from','valid_to','symbol','source'}.issubset(columns)]
    daily_status_candidates = [name for name,columns in all_schemas.items()
        if {'symbol','date','st_flag','suspend_flag','source'}.issubset(columns)]
    # Discovery never auto-certifies a newly placed file. Re-audit its temporal
    # provenance, transitions, coverage and economic parameters before use.
    gates = {name:False for name in REQUIRED_GATES}
    blockers = readiness(gates)
    evidence = {
        'historical_identity_intervals_verified':'security_master.py uses fixed AS_OF and current rows supersede historical code reuse; no qualified historical membership intervals',
        'historical_st_intervals_verified':'st_flag derives from current security_name; no historical effective/release dates',
        'historical_suspension_and_tradability_verified':'DAY format has no suspend/status flags; absent bar is not proof of suspension',
        'historical_security_code_mapping_verified':'current master deduplicates exchange+symbol; complete historical reuse/migration continuity not certified',
        'all_universe_corporate_action_coverage_verified':frozen['official_CA_scope'],
        'official_calendar_2025_2026_verified':'trading_calendar.py contains only explicit2026 holidays;2025 weekdays are not an authoritative exchange calendar',
    }
    state = dict(run_time=datetime.now(timezone.utc).isoformat(),status='PHASE2_DATA_BLOCKED',
        foundation_commit='040ea7932054ad684f30fd6bdc7bf7a36b82c415',as_of_date=AS_OF,
        requested_windows=[120,250],completed_research_trading_days=0,
        valid_stock_day_sample_count=None,baseline_executed=False,
        target_2d_10=None,target_3d_10=None,target_5d_10=None,target_2_5d_10=None,
        target_2_5d_10_pct=None,winners_5d_20=None,winners_5d_30=None,winners_5d_40=None,
        top_features=None,weak_features=None,baseline_best=None,hit_rate=None,capture_rate=None,
        missed_winner_types=None,raw_signal_result=None,tradable_result=None,
        amount_enabled=False,turnover_enabled=False,market_regime_enabled=False,sector_strength_enabled=False,
        current_active_snapshot_count=int(master.listing_status.eq('ACTIVE').sum()),
        current_st_snapshot_count=int(master.loc[master.listing_status.eq('ACTIVE'),'st_flag'].sum()),
        current_bar_status_counts=master.loc[master.listing_status.eq('ACTIVE'),'bar_status'].value_counts().to_dict(),
        canonical_health=canonical,normalized_tables=tables,
        discovered_interval_candidates=interval_candidates,discovered_daily_status_candidates=daily_status_candidates,
        corporate_action_gap=dict(start=gap['start'],end=gap['end'],symbols=gap['verified_symbols'],events=gap['observed_events'],
            qualification='Event presence/type observation only; not all economic price-basis parameters verified'),
        gates=gates,blockers=blockers,blocker_evidence=evidence,
        lookahead_detection='NOT_RUN_FOR_PHASE2',prior_foundation_lookahead=frozen['LOOKAHEAD'],
        current_snapshot_backfill_prevented=True,unknown_status_filled=False,
        quarantines_preserved=['11 ACTIVE_NO_BAR_NEEDS_STATUS','302132 ACTIVE_MISSING_DAY_FILE'],
        source_files_modified=False,next_stage_started=False)
    (ROOT/'outputs/phase2_readiness_health.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    stats='\n'.join(f"- `{t['path']}`: {t['rows']} rows, {t.get('symbols','N/A')} symbols; columns and SHA256 in readiness JSON." for t in tables)
    failures='\n'.join(f'- **{k}**: {evidence[k]}' for k in blockers)
    report=f'''# PHASE2 — 2–5 day +10% target discovery baseline

结论：**PHASE2_DATA_BLOCKED**。数据层已单独封板，但历史可交易股票池所需的 ST/停牌/身份状态链以及全市场复权参数覆盖不足。本轮只完成前置核验，没有实施特征统计或 baseline，所有收益与覆盖率指标保持 DATA_NOT_AVAILABLE，不能解释为零。

数据源/算法的有限集合验收通过，不等于全市场历史研究数据已建库完成。没有继续调查 Amount，也没有用当前5572只股票回填历史。此报告保留后续复跑所需口径及阻断证据。

## Decision: PHASE2_DATA_BLOCKED

Generated {state['run_time']}. Frozen data endpoint: {AS_OF}, not today's market. Foundation commit `{state['foundation_commit']}` was pushed separately. Amount/tick arbitration and private front/back investigation were not resumed.

Source/algorithm qualification is not proof of complete historical research coverage. The current5572-security snapshot must not be backfilled into120/250 historical sessions. No winner sample was silently removed, no missing status became false/zero, and no100-symbol sample substituted for the market denominator.

## Data evidence

Current snapshot ACTIVE={state['current_active_snapshot_count']}, current ST={state['current_st_snapshot_count']}. These are NOT historical research denominators. Current bar-state counts: `{json.dumps(state['current_bar_status_counts'])}`. The11 no-bar securities and302132 remain isolated pending their independent status investigation; they were not filled or permanently removed from a hypothetical historical universe.

{stats}

Existing gap inventory observes{gap['observed_events']} events across a5572-symbol current universe for2026-08-25–2026-09-30. This establishes observation coverage, not full historical price-basis correctness. Historical PIT tests include rejection of incomplete basis; passing tests does not mean those missing event parameters now exist.

## Blocking inputs

{failures}

Normalized interval-table candidates: `{interval_candidates}`. Daily historical status-table candidates: `{daily_status_candidates}`. A newly supplied table still requires source/timestamp/coverage verification; mere filename presence cannot pass this gate. The old generic data_health report is not used as the accepted official DAY endpoint.

## A–O requested research results

| Item | Result |
|---|---|
| A. Research sessions | Planned120 and250; completed0. Exact interval pending authoritative calendar and mature-label alignment. |
| B. Stock-session samples | DATA_NOT_AVAILABLE, not0 market samples |
| C. TARGET_2D_10 count/rate | DATA_NOT_AVAILABLE |
| D. TARGET_3D_10 count/rate | DATA_NOT_AVAILABLE |
| E. TARGET_5D_10 count/rate | DATA_NOT_AVAILABLE |
| F. TARGET_2_5D_10 count/rate | DATA_NOT_AVAILABLE |
| G/H/I. Five-day >=20/30/40% | DATA_NOT_AVAILABLE |
| J. Top20 discriminating features | NOT_COMPUTED |
| K. Weak/ineffective features | NOT_COMPUTED; no claims of uselessness |
| L. Winner/normal/false-breakout differences | NOT_COMPUTED |
| M. Baselines, raw/tradable | NOT_RUN |
| N. Full-market winner capture | DATA_NOT_AVAILABLE; denominator unqualified |
| O. Largest missed-winner type | DATA_NOT_AVAILABLE |

## Frozen interpretation for subsequent implementation (not measured results)

- Decision time is after session t close. close[t] is a diagnostic reference, NOT an assumed attainable fill after computing the close signal. A tradable study needs a separate next-session entry and T+1 sellability path, including unfilled orders.
- Follow the requested literal formulas: horizon h uses exchange sessions t+1..t+h, for h=2,3,5. TARGET_2_5D_10 is their union and therefore equals TARGET_5D_10 under these nested definitions, including a day1 touch. A day2..day5-only target would be a different label, not silently substituted.
- Missing a future market-session bar must not cause indexing to skip suspension and use a later stock bar. Incomplete future observations are censored/unknown, never losers. The final5 exchange sessions lack a mature5-day label at the source endpoint.
- Price features use only data available through t and actions effective by t. Do not reuse a static current-front history. Keep requested RAW target diagnostics clearly identified; corporate actions inside a label horizon require documented comparable-basis handling, not a concealed redefinition of the label.
- Future labels remain in a separate table. Winner, NORMAL (<5%), FALSE_BREAKOUT (causal signal but <10%), and LOSER (<3% plus a prespecified adverse-excursion threshold) can overlap; report overlap/membership rather than force flattering disjoint groups.
- Warm-up needs at least60 prior valid market sessions for60-day features; do not trim the market denominator solely to beautify indicators. Missing-feature coverage must be explicit. New-listing eligibility and next-day high-gap limits must be fixed before outcomes are inspected.
- Feature contract: ret_1/3/5/10/20d, close_vs_ma5/10/20, ma5/10_slope, distance_to_20/60d_high, breakout_20/60d, close_location, gap_pct, range_pct, volume_ratio_5/10, volume_change_1d, up_days_5/10, limit_up_today/count_5/count_10, max_drawdown_5/10. Breakout reference excludes current-day high; zero denominators remain missing. All Amount-derived inputs are banned. Turnover disabled until its source qualifies.
- Candidate fixed-rule families remain Strong Trend Pullback, Breakout Confirmation, Volume Expansion Breakout, Limit-up Follow-through and Low-position Reversal. No thresholds were fit, no best rule selected, and no candidate-count reduction used to raise hit rate.
- Primary capture denominator is unique eligible (security_id, exchange_session) winner observations. Also report unique securities and clustered burst episodes separately; overlapping5-day labels must not masquerade as independent events. Lag1/2/3 pre-burst features require causal timestamps.
- Cross-sectional group differences and buckets are exploratory associations, not proven predictive utility. Time-separated holdout/embargo and clustered uncertainty are necessary before claiming generalization. Raw forward-high reachability is not realized strategy return.

## Leakage and execution

Phase2 feature/label/strategy calculation did not run; therefore no Phase2 LOOKAHEAD=0 claim is made. Existing foundation LOOKAHEAD=0 is retained as prior evidence only. The gate prevented current-name historical-ST leakage and survivorship-biased snapshot backfill. Missing historical limits and state evidence also prevents reliable one-word-limit, halt and next-open feasibility classification.

Two paths were considered: run current survivors/RAW without missing-state controls, or fail closed before publishing performance. The first violates the requested denominator and PIT requirements; this report takes the second. It does not request relaxation of any exclusion or tolerance.

Next required input is a versioned, official-source historical membership/code mapping and ST/trading-state dataset spanning the requested period plus warm-up, an authoritative2025/2026 exchange calendar, and complete verified corporate-action coverage for that historical universe (including delisted securities). No new paid API, account or credential request was made. No automatic trading. Reports/audit tests are the only Phase2 changes, in a separate commit from the data freeze.
'''
    (ROOT/'outputs/phase2_target_discovery_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:state[k] for k in ['status','as_of_date','current_active_snapshot_count','current_st_snapshot_count','blockers']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
