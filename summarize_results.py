#!/usr/bin/env python3
"""Summarize per-run result.json files without training dependencies."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

COMPLETE_STATUSES = {"complete", "completed", "success", "succeeded", "ok"}
SEEDS = {0, 7, 13, 21, 42, 55, 77, 88, 99, 123, 256, 512}


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def metrics_from(result: Mapping[str, Any]) -> dict[str, float]:
    """Flatten scalar test metrics and per-class F1 from the result schema."""
    test = result.get("test") if isinstance(result.get("test"), Mapping) else {}
    metrics: dict[str, float] = {}
    for key, value in test.items():
        converted = number(value)
        if converted is not None:
            metrics[safe_key(str(key))] = converted

    aliases = {
        "macro_f1": ("test_macro_f1", "macro_f1"),
        "accuracy": ("test_accuracy", "accuracy"),
    }
    for canonical, keys in aliases.items():
        if canonical not in metrics:
            for key in keys:
                converted = number(result.get(key))
                if converted is not None:
                    metrics[canonical] = converted
                    break

    per_class = test.get("per_class_f1", result.get("per_class_f1"))
    if isinstance(per_class, Mapping):
        for label, value in per_class.items():
            converted = number(value)
            if converted is not None:
                metrics[f"f1_{safe_key(str(label))}"] = converted
    if "macro_f1" not in metrics:
        raise ValueError("result has no numeric test.macro_f1 (or supported alias)")
    for name, value in metrics.items():
        if (name in {"macro_f1", "accuracy"} or name.startswith("f1_")) and not 0 <= value <= 1:
            raise ValueError(f"result has out-of-range {name}: {value}")
        if name == "loss" and value < 0:
            raise ValueError(f"result has negative loss: {value}")
    return metrics


def read_results(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    identities: set[tuple[str, str, int]] = set()
    for path in sorted(root.rglob("result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            model = str(result["model"])
            protocol = str(result["protocol"])
            seed = int(result["seed"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid result {path}: {error}") from error
        status = str(result.get("status", "")).lower()
        if not status:
            raise ValueError(f"result has no explicit status: {path}")
        if status not in COMPLETE_STATUSES:
            skipped.append(path.relative_to(root).as_posix())
            continue
        if protocol not in {"A", "B"}:
            raise ValueError(f"invalid protocol in {path}: {protocol!r}")
        if seed not in SEEDS:
            raise ValueError(f"invalid final-run seed in {path}: {seed}")
        identity = (model, protocol, seed)
        if identity in identities:
            raise ValueError(f"duplicate result identity: {identity}")
        identities.add(identity)
        records.append({
            "model": model,
            "family": str(result.get("family", "")),
            "protocol": protocol,
            "seed": seed,
            "metrics": metrics_from(result),
            "source": path.relative_to(root).as_posix(),
        })
    return records, skipped


def metric_order(names: Iterable[str]) -> list[str]:
    names = set(names)
    preferred = [name for name in ("macro_f1", "accuracy", "loss") if name in names]
    return preferred + sorted(names - set(preferred))


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_identity = {}
    for record in records:
        cells[(record["model"], record["protocol"])].append(record)
        by_identity[(record["model"], record["protocol"], record["seed"])] = record

    all_metrics = metric_order(
        name for record in records for name in record["metrics"]
    )
    cell_rows = []
    for (model, protocol), group in sorted(cells.items(), key=lambda item: (item[0][0].casefold(), item[0][1])):
        families = {record["family"] for record in group if record["family"]}
        if len(families) > 1:
            raise ValueError(f"inconsistent family values for {model} protocol {protocol}")
        cell_rows.append({
            "model": model,
            "family": next(iter(families), ""),
            "protocol": protocol,
            "n": len(group),
            "metrics": {
                metric: stats([r["metrics"][metric] for r in group if metric in r["metrics"]])
                for metric in all_metrics
                if any(metric in r["metrics"] for r in group)
            },
        })

    paired = []
    models = sorted({record["model"] for record in records}, key=str.casefold)
    for model in models:
        a_seeds = {seed for m, protocol, seed in by_identity if m == model and protocol == "A"}
        b_seeds = {seed for m, protocol, seed in by_identity if m == model and protocol == "B"}
        for seed in sorted(a_seeds & b_seeds):
            a = by_identity[(model, "A", seed)]
            b = by_identity[(model, "B", seed)]
            shared = metric_order(set(a["metrics"]) & set(b["metrics"]))
            paired.append({
                "model": model,
                "family": a["family"] or b["family"],
                "seed": seed,
                "gaps_a_minus_b": {
                    metric: a["metrics"][metric] - b["metrics"][metric]
                    for metric in shared
                },
            })
    paired_summary = []
    for model in models:
        group = [pair for pair in paired if pair["model"] == model]
        if not group:
            continue
        gap_metrics = metric_order(
            metric for pair in group for metric in pair["gaps_a_minus_b"]
        )
        paired_summary.append({
            "model": model,
            "family": group[0]["family"],
            "n_pairs": len(group),
            "metrics": {
                metric: stats([
                    pair["gaps_a_minus_b"][metric]
                    for pair in group if metric in pair["gaps_a_minus_b"]
                ])
                for metric in gap_metrics
            },
        })
    return {
        "metric_order": all_metrics, "cells": cell_rows,
        "paired_seed_gaps": paired, "paired_gap_summary": paired_summary,
    }


def csv_value(value: Any) -> Any:
    return "" if value is None else (f"{value:.12g}" if isinstance(value, float) else value)


def write_outputs(root: Path, output: Path) -> dict[str, Any]:
    records, skipped = read_results(root)
    if not records:
        raise ValueError(f"no completed result.json files found under {root}")
    summary = summarize(records)
    output.mkdir(parents=True, exist_ok=True)
    metrics = summary.pop("metric_order")
    payload = {
        "schema_version": 1,
        "completed_results": len(records),
        "skipped_noncomplete": skipped,
        **summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cell_fields = ["model", "family", "protocol", "n"]
    for metric in metrics:
        cell_fields.extend((f"{metric}_n", f"{metric}_mean", f"{metric}_sample_std"))
    with (output / "cell_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields)
        writer.writeheader()
        for cell in payload["cells"]:
            row = {key: cell[key] for key in ("model", "family", "protocol", "n")}
            for metric, values in cell["metrics"].items():
                for key, value in values.items():
                    row[f"{metric}_{key}"] = csv_value(value)
            writer.writerow(row)

    gap_metrics = metric_order(
        metric for pair in payload["paired_seed_gaps"] for metric in pair["gaps_a_minus_b"]
    )
    gap_fields = ["model", "family", "seed"] + [f"{m}_a_minus_b" for m in gap_metrics]
    with (output / "paired_seed_gaps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=gap_fields)
        writer.writeheader()
        for pair in payload["paired_seed_gaps"]:
            row = {key: pair[key] for key in ("model", "family", "seed")}
            row.update({f"{key}_a_minus_b": csv_value(value)
                        for key, value in pair["gaps_a_minus_b"].items()})
            writer.writerow(row)

    summary_fields = ["model", "family", "n_pairs"]
    for metric in gap_metrics:
        summary_fields.extend(
            (f"{metric}_n", f"{metric}_mean", f"{metric}_sample_std")
        )
    with (output / "paired_gap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for item in payload["paired_gap_summary"]:
            row = {key: item[key] for key in ("model", "family", "n_pairs")}
            for metric, values in item["metrics"].items():
                row.update({f"{metric}_{key}": csv_value(value)
                            for key, value in values.items()})
            writer.writerow(row)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, help="benchmark root (positional alias)")
    parser.add_argument("--input", dest="input_dir", type=Path, help="benchmark root")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path)
    args = parser.parse_args(argv)
    if bool(args.results) == bool(args.input_dir):
        parser.error("provide exactly one of positional RESULTS or --input RESULTS")
    root = args.input_dir or args.results
    output = args.output_dir or root / "summary"
    try:
        payload = write_outputs(root, output)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Summarized {payload['completed_results']} completed run(s) into {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
