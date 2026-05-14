# Fantasy F1 2026 Weekly Workflow

Use this every race week.

## 1. Right after race weekend (results are final)

Update `config.yaml`:
- `current_team.drivers`
- `current_team.constructors`
- `current_team.drs_boost`
- `current_team.budget`
- `current_team.free_transfers`
- `current_team.banked_transfers`
- `current_team.chips_used` / `chips_remaining`

If a driver swap happened (new driver in, old driver out), update:
- `prices.drivers` entries (team + price + name as needed)

## 2. When official price changes are published (usually 1-2 days later)

Update `config.yaml`:
- `prices.drivers.*.price`
- `prices.constructors.*.price`

## 3. Before the next race (forecast window)

Update `config.yaml`:
- `weather_override.next_race_round`
- `weather_override.rain_probability` (0.0 to 1.0)
- `weather_override.notes`

Rule of thumb:
- `0.00-0.20`: mostly dry weekend
- `0.20-0.50`: mixed/uncertain
- `0.50-1.00`: likely wet

## 4. Run prediction + lineup selection

From project root:

```bash
PYTHONPATH=. python3 scripts/run_2026_team_selection.py \
  --round <NEXT_ROUND> \
  --output data/processed/predictions/2026_round<NEXT_ROUND>_predictions.parquet
```

This gives:
- predictions parquet for that round
- recommended 5 drivers + 2 constructors
- recommended DRS driver

## 5. Run transfer recommendation from your actual current team

```bash
PYTHONPATH=. python3 scripts/run_optimizer.py \
  --predictions data/processed/predictions/2026_round<NEXT_ROUND>_predictions.parquet \
  --season-year 2026 \
  --round <NEXT_ROUND> \
  --mode transfers
```

## 6. Save your final decision before lock

In `config.yaml`, set `current_team` to the team you actually submit.

## Notes

- Spending the full 100M is not required; optimizer can intentionally leave budget.
- Current 2026 flow is config-driven (team/prices/weather updates) plus model inference.
- If you want true weekly retraining from live 2026 results, add a 2026 live data ingest step later.
