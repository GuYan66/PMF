"""Raw per-seed metrics and separate summaries for polarity/magnitude runs.

Summary standard deviations use ddof=0. Undefined correlations are blank;
corr_n reports how many seeds contribute to the correlation summary.
"""

import csv
import math
from pathlib import Path
from statistics import fmean, pstdev

METRICS = {"acc3": "Acc3", "macro_f1": "MacroF1", "mae": "MAE", "corr": "Corr"}
DETAIL_COLUMNS = ["run_id", "model", "seed", "split", "scenario", "best_epoch", *METRICS]
SUMMARY_COLUMNS = [
    "run_id", "model", "seeds", "split", "scenario", "n_seeds",
    *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")],
    "corr_n",
]


def _check_header(path, columns):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), None)
        if header != columns:
            raise ValueError(f"Incompatible result CSV header: {path}; expected {columns}, got {header}")


def prepare_result_files(directory, dataset):
    """Validate both destinations before training; leave legacy CSVs untouched."""
    directory = Path(directory)
    detail = directory / f"{dataset}_polmag_v1.csv"
    summary = directory / f"{dataset}_polmag_v1_summary.csv"
    _check_header(detail, DETAIL_COLUMNS)
    _check_header(summary, SUMMARY_COLUMNS)
    return detail, summary


def _append(path, columns, rows):
    path = Path(path)
    _check_header(path, columns)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_seed_results(path, run_id, model, seed, scenario, result):
    rows = []
    for split in ("valid", "test", *(["test2"] if "test2" in result else [])):
        row = dict(run_id=run_id, model=model, seed=seed, split=split,
                   scenario="attachment3_missing_public_labels" if split == "test2" else scenario,
                   best_epoch=result["best_epoch"])
        for column, key in METRICS.items():
            value = float(result[split][key])
            row[column] = value if math.isfinite(value) else None
        rows.append(row)
    _append(path, DETAIL_COLUMNS, rows)


def append_summary(path, run_id, model, seeds, scenario, results):
    """Summarize this invocation only, never mixing prior runs/configurations."""
    rows = []
    if len({"test2" in result for result in results}) > 1:
        raise ValueError("All seeds must evaluate the same splits.")
    for split in ("valid", "test", *(["test2"] if "test2" in results[0] else [])):
        row = dict(run_id=run_id, model=model, seeds=" ".join(map(str, seeds)),
                   split=split, scenario="attachment3_missing_public_labels" if split == "test2" else scenario,
                   n_seeds=len(results))
        for column, key in METRICS.items():
            values = [float(result[split][key]) for result in results]
            values = [value for value in values if math.isfinite(value)]
            row[f"{column}_mean"] = fmean(values) if values else None
            row[f"{column}_std"] = pstdev(values) if values else None
            if column == "corr":
                row["corr_n"] = len(values)
        rows.append(row)
    _append(path, SUMMARY_COLUMNS, rows)
