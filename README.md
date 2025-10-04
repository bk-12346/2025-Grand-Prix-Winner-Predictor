## 2025 Grand Prix Winner Predictor

Predict the winner of Formula 1 Grands Prix using data from 2023–2025.

### Features

- Winner-only classification target (Top-1)
- Inputs: Grid position, qualifying delta to pole, driver/constructor form, track history, DNFs, weather, practice pace
- Data sources: FastF1 (timing), Ergast (results/standings)
- Baselines: Logistic Regression and XGBoost
- Rolling backtest across 2023–2025
- Windows-only setup (tested on Windows 11)

### Quickstart (Windows / PowerShell)

1) Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Prime FastF1 cache (first run may take a while)

```powershell
python -m src.scripts.prime_cache --years 2023 2024 2025
```

4) Build dataset and train baselines

```powershell
python -m src.cli.train_eval --years 2023 2024 2025 --models logreg xgb
```

5) View evaluation report

Outputs are written to `artifacts/` with per-season and overall metrics (AUC, log-loss, Brier, Top-1 hit rate).

### Repo Layout

```
src/
  data/
    ingest_fastf1.py
    build_dataset.py
  features/
    feature_pipeline.py
  models/
    baselines.py
    evaluate.py
  cli/
    train_eval.py
  scripts/
    prime_cache.py
artifacts/
requirements.txt
```

### Configuration

- FastF1 cache directory: defaults to `~/.cache/fastf1`. Override via env var `FASTF1_CACHE_DIR`.
- All scripts accept `--years` to limit the dataset (default: 2023 2024 2025).

### Notes

- No paid APIs are used. Network access is required to download session data on first use.
- If FastF1 fails to download timing for a session, the pipeline will still build rows using Ergast.

### License

MIT


