from __future__ import annotations

import argparse
from typing import List

from src.data.ingest_fastf1 import prime_years


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prime FastF1 cache for given years")
	parser.add_argument("--years", nargs="*", type=int, default=[2023, 2024, 2025])
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	years: List[int] = args.years
	prime_years(years)


if __name__ == "__main__":
	main()


