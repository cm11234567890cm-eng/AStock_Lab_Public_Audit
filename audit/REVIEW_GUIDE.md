# Scope and architecture

This snapshot covers the selected `main` commit, not every experimental branch. See FILE_MANIFEST.json for the exact boundary and source/public hashes.

| Concern | Original code to inspect | What an auditor can check |
|---|---|---|
| Input normalization | `src/data/adapter.py`, `schema.py`, `security_master.py`, `inventory.py` | Schema, missing fields, identity, provenance, input immutability |
| Corporate actions and adjustment | `src/data/corporate_actions.py`, `adjustment_engine.py`, `adjustment_qualification.py` | Official-vs-reported basis, affine transforms, ex-date causality, missing-parameter rejection |
| Selection, factors, research labels | `src/research/target_discovery.py`, `coverage_recovery.py`, `label_censoring.py`, `daily_ranking.py` | Feature/label separation, lookahead, censored labels, universe scope and ranking |
| ML ranking and evaluation | `src/research/ml_ranking.py`, `ml_ranking_metrics.py`, `episodes.py` | Training isolation, walk-forward boundaries, metric denominators, episode vs stock-day distinctions |
| Trade feasibility | `src/research/objective_tradability.py`, `src/market_rules/china_a_share.py` | Gap thresholds, daily-bar proxies, unknown execution, no ex-post replacement |
| Risk and eligibility gates | `src/universe_policy.py`, adjustment/label validity gates | BSE exclusion, fail-closed behavior, unsupported data and invalid samples |
| Research orchestration | Included `scripts/run_phase*_research.py`, ranking/prepare scripts | How modules connect; input/output contracts; conditions that must precede interpretation |
| Dashboard | `src/dashboard/app.py` | Presentation and data-read behavior; live operational data is excluded |
| Backtest | `run_backtest.py` | Explicit unimplemented placeholder; no invented simulation engine |

The broad flow is external data → canonical fields/identity → corporate-action evidence and point-in-time state → feature/label separation → ranking/evaluation → entry-feasibility diagnostics. Public synthetic tests enter at these interfaces, without vendor acquisition.

Prioritize incorrect adjustment bases, future-event influence, missing-data promotion to success, train/validation leakage, hidden sample exclusions, denominator mismatches, and claims of execution that exceed daily-bar evidence. Do not treat a passing synthetic test as proof of historical coverage or profitable trading.

The original repository and Production remain unchanged. Original secret files, data responses, cached evidence, trained artifacts, historical result tables, and author email metadata are not included. The snapshot does not use submodules or copy the original `.git` directory.
