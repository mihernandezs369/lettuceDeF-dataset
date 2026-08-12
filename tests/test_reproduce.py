"""Tests for the single-run reproducer."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import reproduce


def recipe(**changes):
    row = {
        "model": "ResNet50",
        "family": "CNN",
        "protocol": "A",
        "timm_identifier": "resnet50",
        "final_config_fingerprint": "abc123",
        "learning_rate": "0.0003",
        "weight_decay": "0.0001",
        "layer_decay": "",
        "warmup_epochs": "0",
        "epoch_cap": "100",
        "early_stopping_patience": "10",
        "augmentation": "none",
        "randaugment_num_ops": "",
        "randaugment_magnitude": "",
        "randaugment_num_magnitude_bins": "",
        "mixup_alpha": "",
        "cutmix_alpha": "",
        "batch_mix_application_probability": "",
        "mixup_selection_probability": "",
        "cutmix_selection_probability": "",
        "batch_mix_mode": "",
        "label_smoothing": "",
        "effective_resolution_px": "32",
        "batch_size": "4",
        "interpolation": "bicubic",
        "normalization_mean": "[0.485, 0.456, 0.406]",
        "normalization_std": "[0.229, 0.224, 0.225]",
        "configured_trainable_percent": "100",
        "measured_trainable_percent": "",
        "verified_unfrozen_blocks": "",
        "representation_layers": "not applicable",
        "effective_token_pooling": "checkpoint-native final representation",
        "head_construction": "native classifier replaced by Linear(4)",
    }
    row.update({key: str(value) for key, value in changes.items()})
    return row


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TokenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(4))

    def forward(self, tokens):
        return tokens + self.offset


class TinyBackbone(nn.Module):
    """Small timm-shaped model that avoids checkpoint downloads in tests."""

    def __init__(self, classes):
        super().__init__()
        self.num_features = 4
        self.num_prefix_tokens = 1
        self.blocks = nn.ModuleList([TokenBlock() for _ in range(3)])
        self.norm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, classes) if classes else nn.Identity()

    def get_classifier(self):
        return self.classifier

    def forward_features(self, images):
        tokens = torch.ones(images.shape[0], 5, 4, device=images.device)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, images):
        return self.classifier(self.forward_features(images).mean(1))


class RecipeAndModelTests(unittest.TestCase):
    def test_recipe_lookup_accepts_all_13_models_and_both_protocols(self):
        models = {
            "SigLIP", "EVA-02", "DINOv3", "BEiT-3", "ConvNeXt Base",
            "DenseNet201", "EfficientNet-B3", "InceptionV3", "MnasNet-100",
            "MobileNetV2", "ResNet50", "VGG16", "Xception-41",
        }
        rows = [recipe(model=model, protocol=protocol) for model in models for protocol in "AB"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S6.csv"
            write_csv(path, rows)
            for model in models:
                for protocol in "AB":
                    self.assertEqual(reproduce.read_recipe(path, model, protocol)["model"], model)

    def test_native_and_linear_probe_construction(self):
        calls = []

        def factory(identifier, pretrained, num_classes):
            calls.append((identifier, pretrained, num_classes))
            return TinyBackbone(num_classes)

        native = reproduce.build_model(recipe(configured_trainable_percent=0), False, factory)
        self.assertIsInstance(native, TinyBackbone)
        self.assertTrue(all(value.requires_grad for value in native.classifier.parameters()))

        probe_recipe = recipe(
            family="vFM", timm_identifier="tiny_vit",
            head_construction="LinearProbe ending in Linear(4)",
            representation_layers="last2", effective_token_pooling="patch-token mean",
            verified_unfrozen_blocks=1,
        )
        probe = reproduce.build_model(probe_recipe, False, factory)
        self.assertIsInstance(probe, reproduce.LinearProbe)
        self.assertEqual(probe.indices, [1, 2])
        self.assertEqual(probe.token, "patch")
        self.assertFalse(next(probe.backbone.blocks[0].parameters()).requires_grad)
        self.assertTrue(next(probe.backbone.blocks[-1].parameters()).requires_grad)
        self.assertEqual(tuple(probe(torch.zeros(2, 3, 8, 8)).shape), (2, 4))
        self.assertEqual(calls[-1], ("tiny_vit", False, 0))

    def test_zero_block_probe_keeps_backbone_norm_frozen(self):
        probe_recipe = recipe(
            family="vFM", timm_identifier="tiny_vit",
            head_construction="LinearProbe ending in Linear(4)",
            representation_layers="last", effective_token_pooling="CLS token",
            verified_unfrozen_blocks=0,
        )
        probe = reproduce.build_model(
            probe_recipe, False,
            lambda _identifier, pretrained, num_classes: TinyBackbone(num_classes),
        )
        self.assertFalse(next(probe.backbone.norm.parameters()).requires_grad)
        self.assertTrue(next(probe.head.parameters()).requires_grad)

    def test_native_head_includes_head_norm_and_trainable_check(self):
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(4)
                self.fc = nn.Linear(4, 4)

        model = TinyBackbone(4)
        model.classifier = nn.Identity()
        model.head = Head()
        reproduce.adapt_native_model(model, recipe(configured_trainable_percent=0))
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))
        expected = 100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(
            p.numel() for p in model.parameters())
        reproduce.verify_trainable_percentage(
            model, recipe(measured_trainable_percent=f"{expected:.4f}")
        )
        with self.assertRaisesRegex(RuntimeError, "trainable-parameter check failed"):
            reproduce.verify_trainable_percentage(
                model, recipe(measured_trainable_percent="99.9999")
            )


class AugmentationTests(unittest.TestCase):
    def test_randaugment_and_batch_mix_values_come_from_s6(self):
        row = recipe(
            augmentation="randaugment", randaugment_num_ops=2,
            randaugment_magnitude=9, randaugment_num_magnitude_bins=31,
            mixup_alpha=0.2, cutmix_alpha=1.0,
            batch_mix_application_probability=1.0,
            mixup_selection_probability=0.5, cutmix_selection_probability=0.5,
            batch_mix_mode="batch", label_smoothing=0.1,
        )
        config = reproduce.Augmentation.from_recipe(row)
        self.assertEqual((config.rand_ops, config.rand_magnitude, config.rand_bins), (2, 9, 31))
        self.assertEqual(
            (config.mixup_alpha, config.cutmix_alpha, config.application_probability,
             config.mixup_probability, config.cutmix_probability, config.label_smoothing),
            (0.2, 1.0, 1.0, 0.5, 0.5, 0.1),
        )
        training, evaluation = reproduce.build_transforms(row, config)
        self.assertIsInstance(training.transforms[1], reproduce.transforms.PILToTensor)
        self.assertIsInstance(training.transforms[2], reproduce.transforms.RandAugment)
        self.assertEqual(training.transforms[2].num_ops, 2)
        self.assertEqual(len(evaluation.transforms), 3)

    def test_manual_mixup_uses_one_lambda_and_permutation_per_batch(self):
        config = reproduce.Augmentation.from_recipe(recipe(
            mixup_alpha=0.2, cutmix_alpha=1.0,
            batch_mix_application_probability=1,
            mixup_selection_probability=0.5, cutmix_selection_probability=0.5,
            batch_mix_mode="batch", label_smoothing=0.1,
        ))
        images = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 3])
        with patch.object(np.random, "rand", side_effect=[0.0, 0.0]), \
             patch.object(np.random, "beta", return_value=0.25):
            mixed, first, second, coefficient, active = reproduce.BatchMixer(config)(images, labels)
        self.assertTrue(active)
        self.assertEqual(coefficient, 0.25)
        self.assertEqual(first.tolist(), labels.tolist())
        self.assertEqual(sorted(second.tolist()), labels.tolist())
        self.assertEqual(tuple(mixed.shape), tuple(images.shape))

    def test_manual_cutmix_recomputes_lambda_from_the_clipped_box(self):
        config = reproduce.Augmentation.from_recipe(recipe(
            mixup_alpha=0.2, cutmix_alpha=1.0,
            batch_mix_application_probability=1,
            mixup_selection_probability=0.5, cutmix_selection_probability=0.5,
            batch_mix_mode="batch",
        ))
        images = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 3])
        permutation = torch.tensor([1, 0, 3, 2])
        with patch.object(np.random, "rand", side_effect=[0.0, 0.9]), \
             patch.object(np.random, "beta", return_value=0.25), \
             patch.object(np.random, "randint", side_effect=[2, 2]), \
             patch.object(torch, "randperm", return_value=permutation):
            mixed, _first, second, coefficient, active = reproduce.BatchMixer(config)(
                images.clone(), labels
            )
        self.assertTrue(active)
        self.assertEqual(second.tolist(), labels[permutation].tolist())
        self.assertEqual(coefficient, 0.75)
        self.assertFalse(torch.equal(mixed, images))

    def test_manual_cutmix_preserves_rectangular_image_shape(self):
        config = reproduce.Augmentation.from_recipe(recipe(
            cutmix_alpha=1.0, batch_mix_application_probability=1,
            mixup_selection_probability=0, cutmix_selection_probability=1,
            batch_mix_mode="batch",
        ))
        images = torch.arange(4 * 3 * 6 * 10, dtype=torch.float32).reshape(4, 3, 6, 10)
        labels = torch.tensor([0, 1, 2, 3])
        with patch.object(np.random, "rand", return_value=0.0), \
             patch.object(np.random, "beta", return_value=0.25), \
             patch.object(np.random, "randint", side_effect=[5, 3]):
            mixed, _first, _second, coefficient, active = reproduce.BatchMixer(config)(
                images.clone(), labels
            )
        self.assertTrue(active)
        self.assertEqual(tuple(mixed.shape), (4, 3, 6, 10))
        self.assertAlmostEqual(coefficient, 1 - 32 / 60)

    def test_invalid_batch_mix_probabilities_fail_early(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            reproduce.Augmentation.from_recipe(recipe(
                mixup_alpha=0.2, cutmix_alpha=1,
                batch_mix_application_probability=1,
                mixup_selection_probability=0.8, cutmix_selection_probability=0.8,
                batch_mix_mode="batch",
            ))


class ManifestAndOutputTests(unittest.TestCase):
    def test_dataset_resolves_archive_and_flat_image_layouts(self):
        row = {"file_name": "sample.jpg", "class": "FN", "split": "test"}
        layouts = [
            Path("lettuceDeF_protocol_B/test/FN/sample.jpg"),
            Path("test/FN/sample.jpg"), Path("FN/sample.jpg"),
        ]
        for relative in layouts:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / relative
                path.parent.mkdir(parents=True)
                Image.new("RGB", (5, 3)).save(path)
                dataset = reproduce.ImageDataset([row], Path(directory), lambda image: image.size, "B")
                size, label, index = dataset[0]
                self.assertEqual((size, label, index), ((5, 3), 0, 0))

    def test_output_contract_writes_json_csv_and_checkpoint(self):
        args = SimpleNamespace(
            model="ResNet50", protocol="B", seed=42,
            manifest_sha256="a" * 64, recipes_sha256="b" * 64,
            image_check="full", parameters_total=10, parameters_trainable=4,
        )
        history = [{
            "epoch": 1, "train_loss": 1.0, "val_loss": 0.5,
            "val_macro_f1": 0.75, "learning_rate": 0.0003,
        }]
        metrics = {
            "loss": 0.4, "accuracy": 1.0, "macro_f1": 1.0,
            "per_class_f1": {name: 1.0 for name in reproduce.CLASSES},
            "confusion_matrix": [[1, 0, 0, 0]] * 4,
        }
        evaluation = (metrics, [0], [0], [[0.9, 0.05, 0.03, 0.02]], [0])
        rows = [{"file_name": "sample.jpg", "class": "FN", "split": "test"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reproduce.save_outputs(
                output, args, recipe(protocol="B"), history,
                {"weight": torch.tensor([1.0])}, rows, evaluation,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"result.json", "history.csv", "predictions.csv", "checkpoint.pt"},
            )
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["test"]["macro_f1"], 1.0)
            self.assertEqual(result["best_epoch"], 1)

    def test_cli_accepts_canonical_output_dir_and_the_twelve_seeds(self):
        common = [
            "--images", "images", "--manifest", "manifest.csv", "--recipes", "S6.csv",
            "--model", "ResNet50", "--protocol", "A", "--output-dir", "run",
        ]
        for seed in reproduce.SEEDS:
            self.assertEqual(reproduce.parse_args(common + ["--seed", str(seed)]).seed, seed)

    def test_tiny_cpu_run_exercises_the_training_and_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split_index, split in enumerate(("train", "val", "test")):
                for index in range(2):
                    name = f"{split}_{index}.jpg"
                    path = root / "images" / split / "FN" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8),
                              color=(20 + 80 * index, 30 + 70 * split_index,
                                     40 + 20 * split_index + 10 * index)).save(path)
                    rows.append({
                        "file_name": name, "class": "FN", "split": split,
                        "bytes": str(path.stat().st_size),
                        "sha256": reproduce.sha256_file(path),
                    })

            manifest = root / "manifest.csv"
            recipes = root / "S6.csv"
            write_csv(manifest, rows)
            write_csv(recipes, [recipe(epoch_cap=1, batch_size=2)])
            args = SimpleNamespace(
                images=root / "images", manifest=manifest, recipes=recipes,
                model="ResNet50", protocol="A", seed=0,
                output_dir=root / "output", workers=0, device="cpu", dry_run=False,
                image_check="full", allow_custom_manifest=True, overwrite=False,
            )
            with patch.object(reproduce, "build_model", return_value=TinyBackbone(4)):
                reproduce.run(args)

            result = json.loads((args.output_dir / "result.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["epochs_completed"], 1)
            with (args.output_dir / "predictions.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_manifest_rejects_protocol_mismatch_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            rows = [{
                "file_name": "../sample.jpg", "class": "FN", "split": "train",
                "bytes": "10", "sha256": "a" * 64,
            }]
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "unsafe manifest"):
                reproduce.read_manifest(path, "A", allow_custom=True)
            with self.assertRaisesRegex(ValueError, "does not match the Protocol A"):
                reproduce.read_manifest(path, "A", allow_custom=False)

    def test_custom_manifest_rejects_cross_split_image_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            rows = [
                {"file_name": "same.jpg", "class": "FN", "split": split,
                 "bytes": "10", "sha256": ("a" if split == "train" else "b") * 64}
                for split in ("train", "val")
            ]
            rows.append({"file_name": "other.jpg", "class": "FN", "split": "test",
                         "bytes": "10", "sha256": "c" * 64})
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                reproduce.read_manifest(path, "A", allow_custom=True)

            rows[1]["file_name"] = "renamed.jpg"
            rows[1]["sha256"] = rows[0]["sha256"]
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate image content"):
                reproduce.read_manifest(path, "A", allow_custom=True)


if __name__ == "__main__":
    unittest.main()
