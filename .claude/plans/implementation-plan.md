# Fantasy F1 2026 — Data Pipeline & Feature Engineering Plan

## Context
Building a data science pipeline for F1 Fantasy 2026 that predicts fantasy points and optimizes team selection. The project skeleton exists (config.yaml, schema.py, config.py, directory structure) but all implementation modules are empty. Dependencies are not yet installed. User will manually download Kaggle data.

## Scope
Data pipeline (all 4 sources) + feature engineering module. NOT models/optimizer/backtest yet.

## Data Flow
```
Kaggle CSVs (2020-2025 base) ──> kaggle_loader.py ──┐
TracingInsights GitHub (DOTD, pitstops) ──> tracing_loader.py ──┤
OpenF1 API (weather, overtakes, 2023+) ──> openf1_loader.py ──┤──> pipeline.py ──> scoring.py ──> features/builder.py
FastF1 library (fallback + cache) ──> fastf1_loader.py ──┘       │                    │
                                                          sessions.parquet    features.parquet
```

## Implementation Order

### Phase 1: Foundation
1. **Install dependencies** — `pip install -r requirements.txt` in .venv
2. **`src/utils/logging.py`** — Consistent logger setup for all modules
3. **`src/data/scoring.py`** (~200 lines) — Fantasy points calculator from config.yaml rules. Separate functions for qualifying/sprint/race scoring (driver + constructor). Handles DNF, DSQ, pitstop tiers, DOTD, fastest lap.
4. **Update `src/data/schema.py`** — Add STATUS_CLASSIFICATION dict (Kaggle status codes -> "Finished"/"DNF"/"DSQ")

### Phase 2: Data Loaders (can be built in parallel)
5. **`src/data/kaggle_loader.py`** (~250 lines) — Load CSVs from data/raw/kaggle/, join via numeric foreign keys (raceId, driverId, etc.), normalize names via schema.py, output UNIFIED_COLUMNS DataFrame. Handle `\N` nulls, convert qualifying time strings to ms.
6. **`src/data/tracing_loader.py`** (~200 lines) — Download DOTD JSON + PitStops-Archive JSONs from GitHub raw URLs. Cache locally in data/raw/tracing_insights/.
7. **`src/data/openf1_loader.py`** (~300 lines) — Rate-limited client (3 req/sec). Fetch weather, overtakes, pitstops. Session key resolution. Coverage: 2023+.
8. **`src/data/fastf1_loader.py`** (~200 lines) — FastF1 wrapper with caching. Fallback source for weather/timing.

### Phase 3: Pipeline Assembly
9. **`src/data/pipeline.py`** (~250 lines) — Orchestrator: load Kaggle base -> left-join TracingInsights -> left-join OpenF1 -> compute fantasy points -> validate joins (report mismatch counts, never drop rows) -> save parquet. Modes: full/incremental/2026_only.

### Phase 4: Feature Engineering
10. **`src/features/track_features.py`** — Static circuit features from config.yaml (type, overtake_difficulty, DRS zones, downforce, sprint flag)
11. **`src/features/contextual_features.py`** — Weather condition, season phase, era_weight
12. **`src/features/driver_features.py`** — Rolling fantasy pts (3/5 race), historical avg finish per circuit, overtake rate, DNF rate. All with shift(1) to prevent leakage.
13. **`src/features/team_features.py`** — Car-track compatibility score, pitstop performance stats, dev trajectory/score/reg adaptation from config
14. **`src/features/builder.py`** — Orchestrate all feature modules, validate output

### Phase 5: Entry Points & Tests
15. **`scripts/run_pipeline.py`** — CLI entry point for pipeline
16. **`scripts/run_features.py`** — CLI entry point for feature engineering
17. **Tests** — test_scoring.py, test_schema.py, test_kaggle_loader.py, test_features.py

## Key Design Decisions
- **Left joins only** — never silently drop rows; report mismatches
- **Kaggle as base table** — covers 2020-2025; OpenF1 only has 2023+
- **Scoring separate from loaders** — reusable by backtesting, testable in isolation
- **Expanding windows + shift(1)** for features — prevents data leakage
- **TracingInsights pitstops preferred** over Kaggle when both available (DHL official data)
- **Constructor qualifying bonuses stack** (both_q2 + both_q3 = 13pts, per official rules)

## Verification
1. Run pipeline with `python scripts/run_pipeline.py` — check sessions.parquet has ~8,000+ rows
2. Validate fantasy point totals against known 2024/2025 results
3. Run `pytest tests/` — all scoring edge cases pass
4. Check join validation report for acceptable match rates
5. Run features — verify no NaN in feature columns for 2023+ data
