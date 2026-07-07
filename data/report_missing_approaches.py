#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import ast
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FILENAME_DEFAULTS = {
    "task-1-wikipedia-20260614-test.csv": ("task1", "wikipedia-eval"),
    "task-1-wikipedia-small.csv": ("task1", "wikipedia-small"),
    "task-2-llama.csv": ("task2", "llama-eval"),
    "task-2-llama-pg174.csv": ("task2", "llama-pg174"),
    "task-2-llama-dev.csv": ("task2", "llama-dev"),
    "task-3-nq-20260610-test.csv": ("task3", "nq-eval"),
    "task-3-fiqa.csv": ("task3", "fiqa-dev"),
}
DATASET_ALIASES = {
    "benchmark-dev-wikipedia-bge-m3-small": "wikipedia-small",
    "benchmark-eval-wikipedia-bge-m3+goldstandard": "wikipedia-eval",
    "benchmark-dev-wikipedia-bge-m3": "wikipedia-dev",
}


DS_TO_TIRA = {
    "wikipedia-eval": "task-1-wikipedia-20260614-test",
    "wikipedia-small": "task-1-wikipedia-small-20260629-training",
    "llama-eval": "task-2-llama-20260614-test",
    "llama-pg174": "task-2-llama-pg174-20260629-training",
    "llama-dev": "task-2-llama-dev-20260629-training",
    "nq-eval": "task-3-nq-20260610-test",
    "fiqa-dev": "task-3-fiqa-20260630-training",
    "wikipedia-dev": "task-1-wikipedia-dev-20260701-training",
}

DS_TO_PATH = {
    "wikipedia-eval": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-1-wikipedia-20260614-test/",
    "wikipedia-small": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-1-wikipedia-small/",
    "llama-eval": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-2-llama/",
    "llama-pg174": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-2-llama-pg174/",
    "llama-dev": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-2-llama-dev/",
    "nq-eval": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-3-nq-20260610-test/",
    "fiqa-dev": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-3-fiqa/",
    "wikipedia-dev": "/mnt/ceph/storage/web/files/data-in-progress/data-research/sisap-2025/task-1-wikipedia-dev-20260701-training/"
}

def clean_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    text = str(value).strip()
    if len(text) >= 3 and text[0] == "b" and text[1] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8", errors="replace").strip()
        return str(parsed).strip()
    return text


def normalize_task(value: str) -> str:
    text = clean_scalar(value).lower().replace("-", "")
    if text == "task":
        return ""
    if text.startswith("task") and text[4:].isdigit():
        return text
    return ""


def normalize_dataset(value: str) -> str:
    text = clean_scalar(value)
    if text.lower() == "dataset":
        return ""
    return DATASET_ALIASES.get(text, text)


def parse_software_blob(blob: str) -> dict:
    text = clean_scalar(blob)
    if not text:
        return {}
    return json.loads(text)


def most_common(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else ""


def infer_defaults(csv_path: Path, rows: list[dict[str, str]]) -> tuple[str, str]:
    default_task, default_dataset = FILENAME_DEFAULTS.get(csv_path.name, ("", ""))
    row_tasks = [normalize_task(row.get("task", "")) for row in rows]
    row_datasets = [normalize_dataset(row.get("dataset", "")) for row in rows]

    dominant_task = most_common([task for task in row_tasks if task])
    dominant_dataset = most_common([dataset for dataset in row_datasets if dataset])

    return dominant_task or default_task, dominant_dataset or default_dataset


def load_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    default_task, default_dataset = infer_defaults(csv_path, rows)
    records: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=2):
        software = parse_software_blob(row.get("software", ""))
        vm_id = clean_scalar(software.get("vm_id"))
        display_name = clean_scalar(software.get("display_name"))
        if vm_id or display_name:
            approach = f"{vm_id}/{display_name}"
        else:
            approach = "unknown/unknown"
        task = normalize_task(row.get("task", "")) or default_task
        dataset = normalize_dataset(row.get("dataset", "")) or default_dataset

        if not task:
            raise ValueError(f"Could not determine task for {csv_path.name} line {index}")
        if not dataset:
            raise ValueError(f"Could not determine dataset for {csv_path.name} line {index}")

        records.append(
            {
                "task": task,
                "dataset": dataset,
                "approach": approach,
                "vm_id": vm_id,
                "display_name": display_name,
            }
        )

    return records


def merge_metadata(current: dict[str, str] | None, vm_id: str, display_name: str) -> dict[str, str]:
    if current is None:
        return {"vm_id": vm_id, "display_name": display_name}
    return {
        "vm_id": current["vm_id"] or vm_id,
        "display_name": current["display_name"] or display_name,
    }


def build_missing_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    datasets_by_task: dict[str, set[str]] = defaultdict(set)
    approach_datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    approach_metadata: dict[tuple[str, str], dict[str, str]] = {}

    for record in records:
        task = record["task"]
        dataset = record["dataset"]
        approach = record["approach"]
        key = (task, approach)

        datasets_by_task[task].add(dataset)
        approach_datasets[key].add(dataset)
        approach_metadata[key] = merge_metadata(
            approach_metadata.get(key),
            record["vm_id"],
            record["display_name"],
        )

    missing_rows: list[dict[str, str]] = []
    for task in sorted(datasets_by_task):
        all_datasets = sorted(datasets_by_task[task])
        for approach in sorted(
            candidate_approach for candidate_task, candidate_approach in approach_metadata if candidate_task == task
        ):
            key = (task, approach)
            present_datasets = sorted(approach_datasets[key])
            metadata = approach_metadata[key]

            for missing_dataset in sorted(set(all_datasets) - set(present_datasets)):
                missing_rows.append(
                    {
                        "task": task,
                        "approach": approach,
                        "vm_id": metadata["vm_id"],
                        "display_name": metadata["display_name"],
                        "missing_dataset": missing_dataset,
                        "present_datasets": ";".join(present_datasets),
                    }
                )

    return missing_rows


def main() -> int:
    csv_paths = sorted(path for path in SCRIPT_DIR.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {SCRIPT_DIR}")

    records: list[dict[str, str]] = []
    for csv_path in csv_paths:
        records.extend(load_records(csv_path))
    
    for record in build_missing_rows(records):
        print(record["task"] + "\t" + "tira-cli run local --cpus 8 --memory 24g --input " + DS_TO_TIRA[record["missing_dataset"]] + " --out " + DS_TO_PATH[record["missing_dataset"]] + " --approach sisap-2026/" + record["approach"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
