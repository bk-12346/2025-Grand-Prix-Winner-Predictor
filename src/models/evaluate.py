from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def save_metrics(metrics: Dict[str, Dict[str, float]], out_dir: str = "artifacts") -> str:
	path = Path(out_dir)
	path.mkdir(parents=True, exist_ok=True)
	outfile = path / "baseline_metrics.json"
	with outfile.open("w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	return str(outfile)

