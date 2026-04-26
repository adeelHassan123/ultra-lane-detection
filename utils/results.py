import csv
from pathlib import Path


def save_epoch_results(rows: list, filepath: str) -> None:
    if not rows:
        return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_results(filepath: str) -> list:
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: _try_cast(v) for k, v in row.items()})
    return rows


def _try_cast(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
