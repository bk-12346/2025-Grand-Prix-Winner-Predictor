from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from src.data.build_dataset import build_training_table, TARGET_YEARS_DEFAULT
from src.features.feature_pipeline import enrich_features
from src.models.baselines import cross_validate_models, TrainConfig
from src.models.evaluate import save_metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and evaluate winner-only baselines")
	parser.add_argument("--years", nargs="*", type=int, default=list(TARGET_YEARS_DEFAULT))
	parser.add_argument("--models", nargs="*", default=["logreg", "xgb"], choices=["logreg", "xgb"]) 
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	years: List[int] = args.years
	
	# First, let's see what years are actually available in the dataset
	df_all = build_training_table([2020, 2021, 2022, 2023, 2024, 2025])  # Try broader range
	available_years = sorted(df_all['year'].unique()) if not df_all.empty else []
	print(f"Available years in dataset: {available_years}")
	
	# Use available years that overlap with requested years
	if available_years:
		# Use the most recent available years
		use_years = [y for y in available_years if y >= 2020][-3:]  # Last 3 years available
		print(f"Using years: {use_years}")
	else:
		raise SystemExit("No data found in dataset. Check Kaggle dataset availability.")
	
	df = build_training_table(use_years)
	if df.empty:
		raise SystemExit("No data fetched. Check network or year range.")
	
	print(f"Raw data shape: {df.shape}")
	print(f"Columns: {list(df.columns)}")
	
	df_feat = enrich_features(df)
	print(f"After feature engineering: {df_feat.shape}")
	
	# Check for missing values
	print(f"Missing values per column:")
	print(df_feat.isnull().sum())
	
	# Drop rows with all-null features (but keep practice/weather and quali deltas as optional)
	required_features = [c for c in df_feat.columns if c not in ("practice_best_lap_s", "weather_rain_probability", "quali_delta_to_pole_s")]
	df_feat = df_feat.dropna(subset=required_features, how="any")
	print(f"After dropping rows with missing required features: {df_feat.shape}")
	
	print(f"Winner distribution:")
	print(df_feat['is_winner'].value_counts())
	print(f"Years in data: {sorted(df_feat['year'].unique())}")
	print(f"Sample of data:")
	print(df_feat[['driverId', 'raceName', 'year', 'finish_pos', 'is_winner']].head(10))
	
	if df_feat.empty:
		raise SystemExit("No data left after feature engineering. Check data quality.")
	
	if df_feat['is_winner'].sum() == 0:
		raise SystemExit("No winners found in dataset. Check year range and winner detection logic.")
	
	metrics = cross_validate_models(df_feat, TrainConfig(models=args.models))
	outfile = save_metrics(metrics)
	print(f"Saved metrics to {outfile}")
	print(metrics)


if __name__ == "__main__":
	main()


