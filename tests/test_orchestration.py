from __future__ import annotations

import csv
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark
import summarize_results


class RepositoryFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recipes = self.root / "S6.csv"
        with self.recipes.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "protocol", "family"])
            writer.writeheader()
            for index in range(13):
                for protocol in ("A", "B"):
                    writer.writerow({
                        "model": f"Model {index}", "protocol": protocol, "family": "CNN"
                    })
        self.manifest_a = self.root / "manifest_A.csv"
        self.manifest_b = self.root / "manifest_B.csv"
        self.manifest_a.touch()
        self.manifest_b.touch()

    def tearDown(self):
        self.temporary.cleanup()

    def write_result(
        self, model: str, protocol: str, seed: int, macro_f1: float,
        accuracy: float, status: str = "complete",
    ) -> Path:
        path = self.root / model / protocol / str(seed) / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "status": status,
            "model": model,
            "family": "CNN",
            "protocol": protocol,
            "seed": seed,
            "test": {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "per_class_f1": {"healthy": macro_f1 - 0.1},
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        }), encoding="utf-8")
        return path


class LauncherTests(RepositoryFixture):
    def test_full_design_and_filters_have_stable_directories(self):
        cells = run_benchmark.read_cells(self.recipes)
        runs = run_benchmark.select_runs(cells, None, None, None)
        self.assertEqual(len(runs), 312)
        selected = run_benchmark.select_runs(cells, ["Model 2"], ["B"], [7, 42])
        self.assertEqual(
            [run.relative_dir.as_posix() for run in selected],
            ["model_2/protocol_B/seed_7", "model_2/protocol_B/seed_42"],
        )

    def test_dry_run_writes_plan_without_creating_run_directories(self):
        output = self.root / "benchmark"
        status = run_benchmark.main([
            "--images", str(self.root / "images"),
            "--recipes", str(self.recipes),
            "--manifest-a", str(self.manifest_a),
            "--output", str(output),
            "--model", "Model 0", "--protocol", "A", "--seed", "13",
            "--dry-run",
        ])
        self.assertEqual(status, 0)
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["design"]["full_run_count"], 312)
        self.assertEqual(plan["runs"][0]["run_dir"], "model_0/protocol_A/seed_13")
        self.assertEqual(summary["would_launch"], 1)
        self.assertFalse((output / "model_0").exists())

    def test_local_subprocess_result_and_resume(self):
        fake = self.root / "fake_reproduce.py"
        fake.write_text(textwrap.dedent("""
            import argparse, hashlib, json
            from pathlib import Path
            parser = argparse.ArgumentParser()
            parser.add_argument("--model")
            parser.add_argument("--protocol")
            parser.add_argument("--seed", type=int)
            parser.add_argument("--output-dir", type=Path)
            parser.add_argument("--manifest", type=Path)
            parser.add_argument("--recipes", type=Path)
            parser.add_argument("--dry-run", action="store_true")
            args, _ = parser.parse_known_args()
            if args.dry_run:
                raise SystemExit(0)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoint.pt").write_text("fixture")
            (args.output_dir / "history.csv").write_text("epoch,val_loss\\n1,0.5\\n")
            (args.output_dir / "predictions.csv").write_text("file_name,prediction\\na.jpg,FN\\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (args.output_dir / "result.json").write_text(json.dumps({
                "status": "complete", "model": args.model,
                "protocol": args.protocol, "seed": args.seed,
                "test": {"macro_f1": 0.8, "accuracy": 0.75},
                "inputs": {"manifest_sha256": digest(args.manifest),
                           "recipes_sha256": digest(args.recipes)},
                "artifacts": ["result.json", "checkpoint.pt", "history.csv", "predictions.csv"]
            }))
        """), encoding="utf-8")
        output = self.root / "runs"
        common = [
            "--images", str(self.root), "--recipes", str(self.recipes),
            "--manifest-b", str(self.manifest_b),
            "--reproduce", str(fake), "--output", str(output),
            "--model", "Model 1", "--protocol", "B", "--seed", "55",
        ]
        self.assertEqual(run_benchmark.main(common), 0)
        result = output / "model_1/protocol_B/seed_55/result.json"
        self.assertTrue(result.is_file())
        self.assertEqual(run_benchmark.main(common + ["--resume"]), 0)
        summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["launched"], summary["skipped_complete"]), (0, 1))

        # An incomplete artifact must be rerun rather than silently resumed.
        (result.parent / "predictions.csv").write_text("", encoding="utf-8")
        self.assertEqual(run_benchmark.main(common + ["--resume"]), 0)
        summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["launched"], summary["skipped_complete"]), (1, 0))

        # A changed input invalidates the old result instead of silently
        # treating it as resumable.
        self.recipes.write_text(self.recipes.read_text() + "\n", encoding="utf-8")
        self.assertEqual(run_benchmark.main(common + ["--resume"]), 0)
        summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["launched"], summary["skipped_complete"]), (1, 0))


class SummarizerTests(RepositoryFixture):
    def test_cell_statistics_and_complete_seed_pairs(self):
        self.write_result("Example", "A", 0, 0.9, 0.8)
        self.write_result("Example", "A", 7, 0.8, 0.7)
        self.write_result("Example", "B", 0, 0.7, 0.6)
        self.write_result("Example", "B", 7, 0.2, 0.2, status="failed")
        output = self.root / "tables"
        payload = summarize_results.write_outputs(self.root, output)

        self.assertEqual(payload["completed_results"], 3)
        a_cell = next(row for row in payload["cells"] if row["protocol"] == "A")
        self.assertAlmostEqual(a_cell["metrics"]["macro_f1"]["mean"], 0.85)
        self.assertAlmostEqual(
            a_cell["metrics"]["macro_f1"]["sample_std"], 0.07071067811865474
        )
        self.assertEqual(len(payload["paired_seed_gaps"]), 1)
        self.assertAlmostEqual(
            payload["paired_seed_gaps"][0]["gaps_a_minus_b"]["macro_f1"], 0.2
        )
        paired_summary = payload["paired_gap_summary"][0]
        self.assertEqual(paired_summary["n_pairs"], 1)
        self.assertAlmostEqual(paired_summary["metrics"]["macro_f1"]["mean"], 0.2)
        self.assertTrue((output / "summary.json").is_file())
        self.assertIn("macro_f1_sample_std", (output / "cell_summary.csv").read_text())
        self.assertIn("macro_f1_a_minus_b", (output / "paired_seed_gaps.csv").read_text())
        self.assertTrue((output / "paired_gap_summary.csv").is_file())

    def test_top_level_metric_aliases(self):
        metrics = summarize_results.metrics_from({"test_macro_f1": 0.5, "test_accuracy": 0.6})
        self.assertEqual(metrics, {"macro_f1": 0.5, "accuracy": 0.6})

    def test_out_of_range_classification_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "out-of-range macro_f1"):
            summarize_results.metrics_from({"test": {"macro_f1": 1.1}})


if __name__ == "__main__":
    unittest.main()
