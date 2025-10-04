from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import numpy as np


@dataclass
class FeatureConfig:
	rolling_window_races: int = 5


def _compute_driver_form(df: pd.DataFrame, window: int) -> pd.Series:
	# average finish position over last N races per driver (smaller is better)
	df_sorted = df.sort_values(["driverId", "date"]).copy()
	# Convert finish_pos to numeric, handling non-numeric values
	df_sorted["finish_pos_numeric"] = pd.to_numeric(df_sorted["finish_pos"], errors='coerce')
	
	# Compute rolling mean per driver
	df_sorted["driver_form"] = df_sorted.groupby("driverId")["finish_pos_numeric"].shift(1).rolling(window=window, min_periods=1).mean()
	
	return df_sorted["driver_form"]


def _compute_constructor_points(df: pd.DataFrame) -> pd.Series:
	# Use cumulative season constructor points at race date if available; fallback to sum of finish-based points.
	# Placeholder simple mapping: 1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1
	points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
	# Convert finish_pos to numeric first
	finish_numeric = pd.to_numeric(df["finish_pos"], errors='coerce')
	pts = finish_numeric.map(lambda p: points_map.get(int(p), 0) if not pd.isna(p) else 0)
	return pts.groupby([df["constructorId"], df["date"], df["raceName"]]).transform("sum")


def _compute_track_history(df: pd.DataFrame) -> pd.Series:
	# Driver historical average finish at this track up to prior races
	df_sorted = df.sort_values(["driverId", "raceName", "date"]).copy()
	# Convert finish_pos to numeric, handling non-numeric values
	df_sorted["finish_pos_numeric"] = pd.to_numeric(df_sorted["finish_pos"], errors='coerce')
	
	# Compute expanding mean per driver-track combination
	df_sorted["track_history"] = df_sorted.groupby(["driverId", "raceName"])["finish_pos_numeric"].shift(1).expanding(min_periods=1).mean()
	
	return df_sorted["track_history"]


def _compute_dnf_flags(df: pd.DataFrame) -> pd.DataFrame:
	status = df["status"].astype(str).str.lower()
	is_dnf = status.str.contains("dnf") | status.str.contains("accident") | status.str.contains("mechanical") | status.str.contains("collision") | status.str.contains("engine")
	return pd.DataFrame({"is_dnf": is_dnf.astype(int)})


def enrich_features(df: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
	out = df.copy()
	# Core provided inputs mapping
	out["grid_position"] = pd.to_numeric(out["grid"], errors="coerce")
	out["quali_delta_to_pole_s"] = out.get("q_delta_to_pole_s")

	# Driver form: average finish over last N races
	out["driver_form_avg_finish_last5"] = _compute_driver_form(out, config.rolling_window_races)

	# Constructor points (proxy). Real constructor standings can be integrated later.
	out["constructor_points_proxy"] = _compute_constructor_points(out)

	# Track history per driver
	out["driver_track_history_avg_finish"] = _compute_track_history(out)

	# DNF Signals (driver/team). Approximate from status history.
	dnf_df = _compute_dnf_flags(out)
	out["driver_dnf_flag"] = dnf_df["is_dnf"]
	# Team DNF rate proxy: mean of is_dnf per constructor YTD
	out["team_dnf_rate_proxy"] = (
		out.groupby(["constructorId", "date"])  # per event per team
		["driver_dnf_flag"].transform("mean")
	)

	# Weather and practice pace placeholders (to be joined later if available)
	if "practice_best_lap_s" not in out.columns:
		out["practice_best_lap_s"] = np.nan
	if "weather_rain_probability" not in out.columns:
		out["weather_rain_probability"] = np.nan

	return out

