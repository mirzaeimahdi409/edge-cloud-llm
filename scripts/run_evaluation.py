#!/usr/bin/env python
"""CLI entry point for step 9: run the always-edge vs. threshold-policy
comparison and save the results to results/comparison.csv and .json."""

import csv
import json
from pathlib import Path

from edge_cloud_llm.evaluation import rows_to_dicts, run_comparison

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = rows_to_dicts(run_comparison())

    csv_path = RESULTS_DIR / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "comparison.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
