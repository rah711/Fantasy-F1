# Kaggle F1 data

Place CSV files from Kaggle here. The pipeline will load them automatically.

## 2024 data (existing)

- Race results: `f1_2024_race_results.csv`
- Qualifying: `f1_qualifying_results_2024.csv`

## 2025 data

1. Download the dataset: [F1 Race Result 2025](https://www.kaggle.com/datasets/makslypko/f1-race-result-2025)
2. Extract the CSV(s) into this folder (`data/raw/kaggle/`).
3. The loader will pick up files whose names contain `2025` and `race` (e.g. `f1_2025_race_results.csv`, `f1-race-result-2025.csv`, or similar). For qualifying, use a name containing `qualifying` and `2025`.

If the 2025 CSV uses different column names (e.g. "Pos" instead of "position", "Driver" instead of "driver_name"), the loader will map common variants automatically. If your file still fails, rename columns to match: `position`, `driver_name`, `team`, `circuit`, `race_time`, `points`, `fastest_lap`, and either `race_id` or `round`.

After adding 2025 data, run:

```bash
PYTHONPATH=. python scripts/run_pipeline.py --mode full
PYTHONPATH=. python scripts/run_features.py
```

Then you can train with a 2025 holdout: `PYTHONPATH=. python scripts/train_model.py` (default test year is 2025).
