#!/usr/bin/env python3
"""Run the 26 selected model/protocol recipes over the 12 benchmark seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (0, 7, 13, 21, 42, 55, 77, 88, 99, 123, 256, 512)
COMPLETE_STATUSES = {"complete", "completed", "success", "succeeded", "ok"}
REQUIRED_ARTIFACTS = {"result.json", "checkpoint.pt", "history.csv", "predictions.csv"}

def csv_has_data_row(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, None) is not None and next(reader, None) is not None

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class Run:
    model: str
    protocol: str
    seed: int

    @property
    def relative_dir(self) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model).strip("_").lower()
        return Path(slug) / f"protocol_{self.protocol}" / f"seed_{self.seed}"

def read_cells(path: Path) -> list[tuple[str, str]]:
    """Read and validate the fixed 26-cell design from the S6 recipes."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = [(row.get("model", "").strip(), row.get("protocol", "").strip()) for row in rows]
    if len(cells) != 26 or len(set(cells)) != 26:
        raise ValueError(f"{path} must contain 26 unique model/protocol rows")
    by_model: dict[str, set[str]] = {}
    for model, protocol in cells:
        if not model or protocol not in {"A", "B"}:
            raise ValueError(f"{path} has a blank model or a protocol other than A/B")
        by_model.setdefault(model, set()).add(protocol)
    if len(by_model) != 13 or any(value != {"A", "B"} for value in by_model.values()):
        raise ValueError(f"{path} must describe both protocols for exactly 13 models")
    slugs = [Run(model, "A", 0).relative_dir.parts[0] for model in by_model]
    if len(set(slugs)) != 13:
        raise ValueError("model names do not have unique filesystem-safe forms")
    return cells

def select_runs(cells, models=None, protocols=None, seeds=None) -> list[Run]:
    available = {model for model, _ in cells}
    unknown = set(models or ()) - available
    if unknown:
        raise ValueError("unknown model filter(s): " + ", ".join(sorted(unknown)))
    chosen_models = set(models or available)
    chosen_protocols = set(protocols or ("A", "B"))
    chosen_seeds = set(seeds or SEEDS)
    return [Run(model, protocol, seed) for model, protocol in cells
            if model in chosen_models and protocol in chosen_protocols
            for seed in SEEDS if seed in chosen_seeds]

def complete_result(path: Path, run: Run, fingerprint: str = "",
                    input_hashes: dict[str, str] | None = None) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        score = float(result.get("test", {}).get("macro_f1"))
        artifacts = result.get("artifacts", [])
        artifact_names = set(artifacts) if isinstance(artifacts, list) else set()
        artifact_paths = {name: path.parent / name for name in REQUIRED_ARTIFACTS}
        nonempty = all(item.is_file() and item.stat().st_size > 0
                       for item in artifact_paths.values())
        csv_rows = all(csv_has_data_row(artifact_paths[name])
                       for name in ("history.csv", "predictions.csv"))
        valid = (str(result.get("status", "")).lower() in COMPLETE_STATUSES
                 and result.get("model") == run.model
                 and result.get("protocol") == run.protocol
                 and int(result.get("seed")) == run.seed
                 and math.isfinite(score) and 0.0 <= score <= 1.0
                 and (not fingerprint or result.get("recipe_fingerprint") == fingerprint)
                 and artifact_names == REQUIRED_ARTIFACTS
                 and nonempty and csv_rows)
        if input_hashes:
            valid = valid and all(
                result.get("inputs", {}).get(key) == value
                for key, value in input_hashes.items()
            )
        return valid
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error):
        return False

def command_for(run: Run, args: argparse.Namespace, run_dir: Path,
                image_check: str = "size") -> list[str]:
    manifest = args.manifest_a if run.protocol == "A" else args.manifest_b
    return [
        sys.executable, str(args.reproduce),
        "--images", str(args.images), "--manifest", str(manifest),
        "--recipes", str(args.recipes), "--model", run.model,
        "--protocol", run.protocol, "--seed", str(run.seed),
        "--output-dir", str(run_dir), "--workers", str(args.workers),
        "--device", args.device, "--image-check", image_check, "--overwrite",
    ]

def expected_result(run: Run, args: argparse.Namespace):
    return (
        args.fingerprints.get((run.model, run.protocol), ""),
        {"recipes_sha256": args.recipes_sha256,
         "manifest_sha256": args.manifest_sha256[run.protocol]},
    )

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def launch(run: Run, args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.output / run.relative_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with ((run_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
          (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr):
        completed = subprocess.run(command_for(run, args, run_dir),
                                   stdout=stdout, stderr=stderr, check=False)
    valid = complete_result(run_dir / "result.json", run, *expected_result(run, args))
    return {
        "model": run.model, "protocol": run.protocol, "seed": run.seed,
        "run_dir": run.relative_dir.as_posix(), "returncode": completed.returncode,
        "status": "succeeded" if completed.returncode == 0 and valid else "failed",
        "result_valid": valid,
    }

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path, help="LettuceDeF archive root")
    p.add_argument("--output", type=Path, default=Path("benchmark_results"))
    p.add_argument("--reproduce", type=Path, default=HERE / "reproduce.py")
    p.add_argument("--recipes", type=Path, default=HERE / "S6_selected_recipes.csv")
    p.add_argument("--manifest-a", type=Path, default=HERE / "manifest_protocol_A.csv")
    p.add_argument("--manifest-b", type=Path, default=HERE / "manifest_protocol_B.csv")
    p.add_argument("--model", action="append", help="model name; repeat to select several")
    p.add_argument("--protocol", action="append", choices=("A", "B"))
    p.add_argument("--seed", action="append", type=int, choices=SEEDS)
    p.add_argument("--jobs", type=int, default=1, help="concurrent local runs (default: 1)")
    p.add_argument("--workers", type=int, default=24, help="data-loader workers per run")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true", help="write the plan without training")
    p.add_argument("--resume", action="store_true", help="skip valid completed results")
    return p

def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.jobs < 1 or args.workers < 0:
        raise SystemExit("--jobs must be positive and --workers cannot be negative")
    if args.jobs > 1 and args.device != "cpu":
        raise SystemExit("--jobs > 1 requires --device cpu; runs cannot share cuda:0")
    try:
        runs = select_runs(read_cells(args.recipes), args.model, args.protocol, args.seed)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    required = [args.reproduce, args.recipes]
    required += [args.manifest_a] if any(run.protocol == "A" for run in runs) else []
    required += [args.manifest_b] if any(run.protocol == "B" for run in runs) else []
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("required file(s) not found: " + ", ".join(missing))
    with args.recipes.open(newline="", encoding="utf-8") as handle:
        args.fingerprints = {
            (row["model"], row["protocol"]): row.get("final_config_fingerprint", "")
            for row in csv.DictReader(handle)
        }
    args.recipes_sha256 = sha256_file(args.recipes)
    args.manifest_sha256 = {
        protocol: sha256_file(args.manifest_a if protocol == "A" else args.manifest_b)
        for protocol in {run.protocol for run in runs}
    }
    args.output.mkdir(parents=True, exist_ok=True)

    actions = []
    for run in runs:
        result_path = args.output / run.relative_dir / "result.json"
        if result_path.exists() and not args.resume:
            raise SystemExit(f"existing result: {result_path}; use --resume or a new --output")
        action = "skip_complete" if args.resume and complete_result(
            result_path, run, *expected_result(run, args)
        ) else "run"
        actions.append((run, action))
    plan = {
        "schema_version": 1,
        "design": {"cells": 26, "seeds": list(SEEDS), "full_run_count": 312},
        "inputs": {"recipes_sha256": args.recipes_sha256,
                   "manifest_sha256": args.manifest_sha256},
        "selected_run_count": len(runs),
        "runs": [{
            "model": run.model, "protocol": run.protocol, "seed": run.seed,
            "run_dir": run.relative_dir.as_posix(), "action": action,
            "command": command_for(run, args, args.output / run.relative_dir),
        } for run, action in actions],
    }
    write_json(args.output / "plan.json", plan)
    pending = [run for run, action in actions if action == "run"]
    skipped = len(actions) - len(pending)
    if args.dry_run:
        summary = {
            "schema_version": 1, "mode": "dry_run", "selected": len(runs),
            "would_launch": len(pending), "skipped_complete": skipped,
            "succeeded": 0, "failed": 0, "runs": [],
        }
        write_json(args.output / "run_summary.json", summary)
        print(f"Planned {len(pending)} run(s); skipped {skipped}; no subprocesses launched.")
        return 0

    # Hash every image once per selected protocol. Individual runs then repeat
    # the inexpensive size check without rereading the full archive 312 times.
    for protocol in sorted({run.protocol for run in pending}):
        example = next(run for run in pending if run.protocol == protocol)
        preflight = command_for(example, args, args.output / ".preflight", "full") + ["--dry-run"]
        if subprocess.run(preflight, check=False).returncode:
            raise SystemExit(f"Protocol {protocol} input verification failed")

    outcomes = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(launch, run, args) for run in pending]
        for index, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            outcomes.append(outcome)
            print(f"[{index}/{len(pending)}] {outcome['status']}: "
                  f"{outcome['model']} {outcome['protocol']} seed {outcome['seed']}")
    outcomes.sort(key=lambda row: (str(row["model"]).casefold(), row["protocol"], row["seed"]))
    failed = sum(outcome["status"] == "failed" for outcome in outcomes)
    summary = {
        "schema_version": 1, "mode": "execute", "selected": len(runs),
        "launched": len(pending), "skipped_complete": skipped,
        "succeeded": len(outcomes) - failed, "failed": failed, "runs": outcomes,
    }
    write_json(args.output / "run_summary.json", summary)
    print(f"Finished: {summary['succeeded']} succeeded, {failed} failed, {skipped} skipped.")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
