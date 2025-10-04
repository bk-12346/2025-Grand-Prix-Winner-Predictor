from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

try:
	from xgboost import XGBClassifier  # type: ignore
except Exception:  # noqa: BLE001
	XGBClassifier = None  # type: ignore


FEATURE_COLUMNS = [
	"grid_position",
	"driver_form_avg_finish_last5",
	"constructor_points_proxy",
	"driver_track_history_avg_finish",
	"driver_dnf_flag",
	"team_dnf_rate_proxy",
	# Optional features (may be missing)
	"quali_delta_to_pole_s",
	"practice_best_lap_s",
	"weather_rain_probability",
]


@dataclass
class TrainConfig:
	models: List[str]
	group_by: str = "date"  # group by event to avoid leakage


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	# Only use features that actually exist in the dataframe
	available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
	X = df[available_features].copy()
	
	# Fill missing values with 0 for optional features
	for col in available_features:
		if col in ["quali_delta_to_pole_s", "practice_best_lap_s", "weather_rain_probability"]:
			X[col] = X[col].fillna(0)
	
	Y = df["is_winner"].astype(int)
	return X, Y


def _build_logreg_pipeline() -> Pipeline:
	return Pipeline(
		steps=[
			("scaler", StandardScaler(with_mean=True, with_std=True)),
			("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
		]
	)


def _build_xgb_pipeline(random_state: int = 42) -> Pipeline:
	if XGBClassifier is None:
		raise RuntimeError("XGBoost not available; install xgboost")
	clf = XGBClassifier(
		n_estimators=400,
		max_depth=4,
		learning_rate=0.05,
		subsample=0.9,
		colsample_bytree=0.9,
		reg_lambda=1.0,
		objective="binary:logistic",
		eval_metric="logloss",
		random_state=random_state,
	)
	return Pipeline(steps=[("clf", clf)])


def cross_validate_models(df: pd.DataFrame, config: TrainConfig) -> Dict[str, Dict[str, float]]:
	X, y = _prepare_xy(df)
	groups = df[config.group_by]
	gkf = GroupKFold(n_splits=5)
	results: Dict[str, Dict[str, float]] = {}

	model_builders = {
		"logreg": _build_logreg_pipeline,
		"xgb": _build_xgb_pipeline,
	}

	for model_name in config.models:
		builder = model_builders[model_name]
		metrics = {"logloss": [], "brier": [], "top1_hit_rate": []}
		for train_idx, test_idx in gkf.split(X, y, groups=groups):
			X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
			y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
			model = builder()
			model.fit(X_tr, y_tr)
			probs = model.predict_proba(X_te)[:, 1]
			metrics["logloss"].append(log_loss(y_te, probs, labels=[0, 1]))
			metrics["brier"].append(brier_score_loss(y_te, probs))
			# Top-1 hit rate per race: does predicted max prob driver equal actual winner?
			te_df = df.iloc[test_idx].copy()
			te_df = te_df.assign(pred=probs)
			per_event = []
			for (date, race), g in te_df.groupby(["date", "raceName"]):
				predicted = int(g.loc[g["pred"].idxmax(), "is_winner"])  # 1 if predicted winner equals actual winner row
				per_event.append(predicted)
			metrics["top1_hit_rate"].append(float(np.mean(per_event)) if per_event else np.nan)
		results[model_name] = {k: float(np.nanmean(v)) for k, v in metrics.items()}

	return results

