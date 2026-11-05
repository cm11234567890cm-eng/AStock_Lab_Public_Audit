# AStock Lab — Public Code Audit Snapshot

This independent repository is a code-review snapshot, not the operational research workspace. It has fresh Git history and contains no original author metadata, credentials, vendor datasets, trained model artifacts, or historical performance reports.

Source repository: `cm11234567890cm-eng/AStock_Lab` (remains private).

Source revision: `fc5153dc4581fd4094d1e9c263dd0ed420ef170a` (`main`). Other branches and uncommitted local work are outside this snapshot. The source revision is a provenance identifier, not an imported commit.

## Start the audit here

1. Read [review scope and architecture](audit/REVIEW_GUIDE.md).
2. Read [data interfaces and synthetic examples](audit/DATA_INTERFACES.md).
3. Inspect `src/data/adjustment_engine.py`, `src/data/corporate_actions.py`, `src/research/`, `src/market_rules/`, and `src/universe_policy.py`.
4. Consult [included and excluded files](audit/FILE_MANIFEST.json) and [validation results](audit/VALIDATION.json).
5. Follow [rights and exclusions](audit/RIGHTS_AND_EXCLUSIONS.md).

Included original computational code is byte-identical to the selected source revision unless an individual manifest entry explicitly says otherwise. Algorithms have not been rewritten to make publication or tests pass. Generated audit documentation and synthetic examples are separate additions.

## Run the offline synthetic suite

Use Python 3.12 in a fresh virtual environment:

```sh
python -m venv .venv
# Activate this new environment using the command appropriate to your shell.
python -m pip install -r audit/requirements-test.txt
python audit/run_offline_tests.py
python examples/synthetic_adjustment.py
```

No TdxAiData Key, broker account, terminal, original project directory, or real market data is required. Test inputs are constructed synthetic fixtures. The audit runner uses Python audit hooks to deny networking, subprocesses, reads outside the snapshot/Python runtime, and writes outside its scratch directory. These guards support the reviewed tests; they are not an operating-system sandbox for arbitrary untrusted code. It does not run live-fetch scripts, research pipeline entry points, or historical data qualification suites.

`requirements.txt` and `requirements-phase25.txt` document original project dependencies; some belong only to optional data acquisition or ML research. Third-party packages and native libraries are not bundled here.

## Important boundaries

- The original `run_backtest.py` deliberately reports that a complete backtest engine is not implemented. This snapshot does not add one or claim that one exists.
- `objective_tradability.py` analyzes daily-bar entry proxies and opening-information-only selection. It does not prove actual fills, queue priority, intraday liquidity, or live execution.
- Risk-related controls are universe exclusions, trading rules, data validity, temporal isolation, and fail-closed handling. Do not infer a complete broker-connected portfolio risk service.
- Historical OHLC/PIT qualification and past test counts are not republished as newly reproduced results. Public tests establish only the stated synthetic-code checks.
- Missing vendor data is intentional. Full-market empirical validation, original performance reproduction, and vendor API entitlement testing cannot be performed with this repository alone.
- The included 12 original synthetic test files passed 90 tests with zero skips. Mixed real/synthetic test files were excluded as whole files, not silently skipped during execution. See the manifest for their names. This is not the old full acceptance suite.
- NumPy emitted 57 timedelta deprecation warnings in `target_discovery.py`; the original code is retained for review. This is a future compatibility issue, not a failed test in the recorded environment.
- The dashboard is available for source review but requires omitted operational configuration/data to run. `trading_calendar.py` contains a limited, fixed calendar; it is not a maintained universal exchange calendar.
- Some original code contains installation-path defaults or reads local input files when explicitly invoked. Read those implementations; use only the offline audit runner for the provided reproducibility check.

Review findings should identify file/function, the triggering scenario, expected and actual behavior, and a minimal synthetic reproducer. Distinguish confirmed defects from missing evidence. Do not turn research outputs into automated orders.
