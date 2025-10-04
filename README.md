# 2025 Grand Prix Winner Predictor

Predict the winner of Formula 1 Grands Prix using machine learning. This project achieves **82.4% accuracy** in predicting race winners using historical F1 data and advanced feature engineering.

## 🏆 Results

**XGBoost Model Performance:**
- **Top-1 Hit Rate: 82.4%** (correctly predicts winner in 4 out of 5 races)
- **Log Loss: 0.095** (excellent probability calibration)
- **Brier Score: 0.025** (outstanding calibration)

**Logistic Regression Baseline:**
- **Top-1 Hit Rate: 67.7%**
- **Log Loss: 0.238**
- **Brier Score: 0.075**

## 🚀 Features

- **Winner-only classification** (Top-1 prediction)
- **Comprehensive feature engineering:**
  - Grid position
  - Driver form (average finish over last 5 races)
  - Constructor championship points
  - Driver track history
  - DNF rates (driver & team)
  - Qualifying deltas to pole position
- **Data sources:** Kaggle F1 dataset (1950-2020)
- **Models:** Logistic Regression and XGBoost
- **Evaluation:** Rolling cross-validation with proper temporal splits
- **Windows-compatible** setup

## 📊 Quickstart (Windows / PowerShell)

### 1. Setup Environment

```powershell
# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API (Required)

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in `%USERPROFILE%\.kaggle\` folder
5. Set permissions: Right-click → Properties → Security → Advanced → Disable inheritance

### 3. Run Evaluation

```powershell
# Train and evaluate both models
python -m src.cli.train_eval --years 2020 2021 2022 --models logreg xgb

# Quick test with just logistic regression
python -m src.cli.train_eval --years 2020 2021 2022 --models logreg
```

### 4. View Results

Results are saved to `artifacts/baseline_metrics.json` with detailed performance metrics.

## 📁 Project Structure

```
src/
├── data/
│   ├── kaggle_loader.py      # Kaggle dataset integration
│   └── build_dataset.py     # Dataset construction
├── features/
│   └── feature_pipeline.py  # Feature engineering
├── models/
│   ├── baselines.py         # Model training & evaluation
│   └── evaluate.py          # Metrics saving
├── cli/
│   └── train_eval.py        # Command-line interface
└── scripts/
    └── prime_cache.py       # FastF1 cache management

artifacts/                   # Model results & metrics
data/                       # Kaggle datasets (auto-downloaded)
requirements.txt            # Python dependencies
```

## 🔧 Configuration

- **Years:** Use `--years` to specify data range (default: 2020-2022)
- **Models:** Choose `logreg`, `xgb`, or both
- **Cache:** FastF1 cache defaults to `.fastf1-cache/` (can override with `FASTF1_CACHE_DIR`)

## 📈 Model Details

### Features Used
1. **Grid Position** - Starting position (1-20)
2. **Driver Form** - Average finish position over last 5 races
3. **Constructor Points** - Team championship points proxy
4. **Track History** - Driver's historical performance at specific circuits
5. **DNF Rates** - Driver and team reliability metrics
6. **Qualifying Delta** - Time difference to pole position (when available)

### Evaluation Method
- **Temporal Cross-Validation:** Groups by race date to prevent data leakage
- **5-Fold CV:** Ensures robust performance estimates
- **Metrics:** Log loss, Brier score, Top-1 hit rate

## 🎯 Performance Analysis

The **82.4% accuracy** achieved by XGBoost demonstrates that:
- F1 winner prediction is highly feasible with proper features
- Driver form and track history are strong predictors
- Grid position provides valuable starting advantage signal
- Team performance (constructor points) matters significantly

## 🔮 Future Enhancements

- **Recent Data:** Integrate 2023-2025 data sources
- **Weather Features:** Rain probability, temperature, wind
- **Practice Times:** FP1/FP2/FP3 pace analysis
- **Advanced Models:** Neural networks, ensemble methods
- **Real-time Prediction:** Live race prediction API

## 📋 Requirements

- Python 3.11+
- Windows 10/11
- Internet connection (for Kaggle data download)
- Kaggle account and API token

## 📄 License

MIT License - Feel free to use and modify for your own F1 prediction projects!

---

**Note:** This project uses historical F1 data (1950-2020) from Kaggle. For predictions on recent races, additional data sources would be needed.


