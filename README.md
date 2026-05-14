# Fantasy F1 2026 — Data Science Model

A weekly decision-support tool for the official 2026 Fantasy F1 game. It ingests
historical and live race data, predicts driver and constructor fantasy points
for the upcoming round, and recommends a 5-driver / 2-constructor lineup
(plus DRS Boost) under the £100M budget and transfer rules.

The whole thing is config-driven: race week updates happen in `config.yaml`,
and the pipeline + optimizer + Streamlit console all read from it.

## Why this exists — the 2026 season experiment

This project is one of three Fantasy F1 teams I'm running in parallel across
the full 2026 season to answer a question I actually care about:

> **Does AI-assisted human judgement beat pure human judgement — and does it
> beat pure AI decision-making?**

The three teams:

1. **Vibe-coded data science model** *(this repo)* — AI-assisted human. I
   decide which factors matter (track characteristics, weather sensitivity,
   how much fantasy points actually correlate with podium finishes vs.
   overtakes vs. price gains), and the model surfaces signal I can't track
   on my own as a layperson hobbyist.
2. **Just me** — pure human judgement. How I normally play: gut feel as a
   fan, plus the usual F1 budget-builder community sources for likely
   weekly price changes.
3. **Claude chat project** — pure AI. A Claude project given only the
   minimal inputs (race results + the official fantasy scoring rules) and
   asked to pick a team each week, with no engineered features or
   domain-specific tooling.

The interesting bit about team #1 isn't that it's "AI-powered" — it's that
it lets me act on factors I believe are important but have no way to
quantify by hand: high-speed-corner counts, downforce demands, rainfall
impact on finishing order, the relative weight of qualifying vs. race vs.
sprint scoring, etc. The model is where my hypotheses about *what matters*
get tested against historical data.

## What it does

- **Pulls race data** from Kaggle (historical), FastF1, OpenF1, and Tracing Insights
  (DOTD, pitstops, archive) into a unified `sessions.parquet`.
- **Computes fantasy points** for every past session using the official 2026
  scoring rules (qualifying, sprint, race; driver + constructor).
- **Engineers features** — track characteristics, driver rolling form, team
  development score, regulation-era weight, weather, car-track interactions.
- **Trains a points model** (LightGBM by default, XGBoost swappable) with
  walk-forward backtesting on a held-out season.
- **Models price changes** so the optimizer can value short-term price gains
  alongside expected points.
- **Optimizes the team** with PuLP integer programming under budget and
  transfer constraints; supports full-team selection or transfer-only mode.
- **Streamlit console** (`frontend/`) for race-week control: edit working
  config, view predictions and recommendations, open a PR back to the repo
  with the chosen lineup.

## Project layout

```
config.yaml              # Master config — scoring rules, calendar, prices, current team, weather
src/
  data/                  # Loaders (Kaggle, FastF1, OpenF1, Tracing) + pipeline + scoring
  features/              # Feature builders (driver, team, track, contextual, car-track)
  model/                 # LightGBM / XGBoost training
  price/                 # Price-change model
  optimizer/             # PuLP team selector + transfer optimizer
  backtest/              # Walk-forward season evaluation
  ui_services/           # Services backing the Streamlit app (recommendations, GitHub write-back)
scripts/                 # CLI entry points (pipeline, train, optimize, tune, weekly run)
frontend/                # Streamlit multipage app
tests/                   # pytest suite (scoring, features, optimizer, walk-forward, UI services)
docs/                    # Weekly workflow + frontend quickstart
data/                    # raw / processed / fantasy / cache (most contents gitignored)
notebooks/               # Exploratory analysis
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12.8 (see `.python-version`).

## Typical workflow

### One-time / occasional

```bash
# Build the unified session table from all sources
PYTHONPATH=. python3 scripts/run_pipeline.py --mode full

# Build features from sessions
PYTHONPATH=. python3 scripts/run_features.py

# Train the points model
PYTHONPATH=. python3 scripts/train_model.py
```

### Each race week

Full step-by-step lives in [docs/weekly_workflow_2026.md](docs/weekly_workflow_2026.md).
Short version:

1. After the previous race, update `current_team` and `prices.*` in `config.yaml`.
2. Set `weather_override` for the upcoming round.
3. Generate predictions + lineup:
   ```bash
   PYTHONPATH=. python3 scripts/run_2026_team_selection.py \
     --round <NEXT_ROUND> \
     --output data/processed/predictions/2026_round<NEXT_ROUND>_predictions.parquet
   ```
4. Get transfer recommendations from your actual team:
   ```bash
   PYTHONPATH=. python3 scripts/run_optimizer.py \
     --predictions data/processed/predictions/2026_round<NEXT_ROUND>_predictions.parquet \
     --season-year 2026 --round <NEXT_ROUND> --mode transfers
   ```

### Streamlit console

```bash
PYTHONPATH=. streamlit run frontend/app.py
```

Setup notes (passwords, GitHub write-back) in
[docs/frontend_quickstart.md](docs/frontend_quickstart.md).

## Tests

```bash
PYTHONPATH=. pytest
```

## Status

Built for the 2026 season. Currently config-driven for race-week updates;
live in-season retraining from 2026 results is a future addition.
