#!/usr/bin/env python
"""CLI entry point for section 11's τ/k sweep: runs every combination of
threshold and window size over the sample prompts and saves the results to
results/sweep.csv and .json."""

import csv
import json
from pathlib import Path

from edge_cloud_llm.evaluation import rows_to_dicts, run_sweep

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = rows_to_dicts(run_sweep())

    csv_path = RESULTS_DIR / "sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "sweep.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
