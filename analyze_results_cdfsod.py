#!/usr/bin/env python3
"""Summarize CD-FSOD mAP from the dataset/shot experiment tree."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


DATASETS = ("ArTaxOr", "Clipart1k", "DIOR", "FISH", "NEU-DET", "UODD")
SHOTS = (1, 5, 10)
METRIC = "coco/bbox_mAP"


def iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Read either a JSON object or MMEngine's line-delimited JSON logs."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value
    else:
        if isinstance(value, dict):
            yield value


def latest_metric(work_dir: Path) -> float | None:
    candidates: list[tuple[int, int, float]] = []
    for json_path in work_dir.rglob("*.json"):
        try:
            modified = json_path.stat().st_mtime_ns
            for order, record in enumerate(iter_json_objects(json_path)):
                value = record.get(METRIC)
                if isinstance(value, (int, float)):
                    candidates.append((modified, order, float(value)))
        except (OSError, UnicodeError):
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize exp_cdfsod_results/<dataset>/<shot>shot metrics")
    parser.add_argument(
        "experiment_directory",
        nargs="?",
        default="exp_cdfsod_results",
        type=Path,
        help="experiment root (default: exp_cdfsod_results)")
    args = parser.parse_args()

    root = args.experiment_directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"experiment directory does not exist: {root}")

    results: dict[int, list[tuple[str, float]]] = {shot: [] for shot in SHOTS}
    print(f"Results root: {root}")

    for shot in SHOTS:
        print(f"\n{shot}-shot")
        print("-" * 44)
        for dataset in DATASETS:
            work_dir = root / dataset / f"{shot}shot"
            metric = latest_metric(work_dir) if work_dir.is_dir() else None
            if metric is None:
                print(f"{dataset:<12} {'N/A':>8}")
                continue
            map_percent = metric * 100
            results[shot].append((dataset, map_percent))
            print(f"{dataset:<12} {map_percent:>7.3f}")

        values = [value for _, value in results[shot]]
        if values:
            print("-" * 44)
            print(f"Average      {sum(values) / len(values):>7.3f} ({len(values)}/6 datasets)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
