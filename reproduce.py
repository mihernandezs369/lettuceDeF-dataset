#!/usr/bin/env python3
"""Train and evaluate one LettuceDeF model--protocol--seed run.

Inputs are the images, split manifest, and Supplementary Table S6. Outputs
are a checkpoint, epoch history, test predictions, and machine-readable metrics.

    python reproduce.py --images /data/lettuceDeF \\
      --manifest manifest_protocol_B.csv --recipes S6_selected_recipes.csv \\
      --model "ResNet50" --protocol B --seed 42 --output-dir runs/resnet50_B_42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASSES = ["FN", "K-deficiency", "N-deficiency", "P-deficiency"]
SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123, 256, 512]
MINIMUM_LR = 1e-5
REFERENCE_MANIFESTS = {
    "A": "26ee4f6cac887fc23c08499f1b1eb2647c1f585079aed95d553380fa7f6785d0",
    "B": "6d1e0f219e3e6d6cfc422abc6bbc840b4b31054471a24c3b5c5731915c885808",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def optional_number(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key) or "").strip()
    return float(value) if value else default

@dataclass(frozen=True)
class Augmentation:
    """Image- and batch-level augmentation values stored in S6."""
    name: str
    rand_ops: int
    rand_magnitude: int
    rand_bins: int
    mixup_alpha: float
    cutmix_alpha: float
    application_probability: float
    mixup_probability: float
    cutmix_probability: float
    mode: str
    label_smoothing: float
    @classmethod
    def from_recipe(cls, row: Mapping[str, str]) -> "Augmentation":
        mixup = optional_number(row, "mixup_alpha")
        cutmix = optional_number(row, "cutmix_alpha")
        enabled = mixup > 0 or cutmix > 0
        both = mixup > 0 and cutmix > 0
        result = cls(
            name=str(row.get("augmentation") or "none").lower(),
            rand_ops=int(optional_number(row, "randaugment_num_ops", 2)),
            rand_magnitude=int(optional_number(row, "randaugment_magnitude", 9)),
            rand_bins=int(optional_number(row, "randaugment_num_magnitude_bins", 31)),
            mixup_alpha=mixup, cutmix_alpha=cutmix,
            application_probability=optional_number(
                row, "batch_mix_application_probability", 1.0 if enabled else 0.0
            ),
            mixup_probability=optional_number(
                row, "mixup_selection_probability", 0.5 if both else float(mixup > 0)
            ),
            cutmix_probability=optional_number(
                row, "cutmix_selection_probability", 0.5 if both else float(cutmix > 0)
            ),
            mode=str(row.get("batch_mix_mode") or ("batch" if enabled else "none")).lower(),
            label_smoothing=optional_number(row, "label_smoothing"),
        )
        result.validate()
        return result

    @property
    def uses_randaugment(self) -> bool:
        return self.name == "randaugment"
    @property
    def uses_batch_mix(self) -> bool:
        return self.mixup_alpha > 0 or self.cutmix_alpha > 0

    def validate(self) -> None:
        valid_randaugment = self.rand_ops >= 1 and self.rand_magnitude >= 0 and self.rand_bins >= 2
        if self.name not in {"none", "randaugment"} or not valid_randaugment:
            raise ValueError("invalid image augmentation values in S6")
        probabilities = (self.application_probability, self.mixup_probability,
                         self.cutmix_probability)
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("batch-mixing probabilities must be in [0, 1]")
        if self.mixup_alpha < 0 or self.cutmix_alpha < 0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("invalid Mixup/CutMix alpha or label-smoothing value")
        valid_mix = self.mode == "batch" and math.isclose(
            self.mixup_probability + self.cutmix_probability, 1.0)
        if self.uses_batch_mix and not valid_mix:
            raise ValueError("batch mode is required and method probabilities must sum to one")

def read_recipe(path: Path, model: str, protocol: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle) if
            row["model"] == model and row["protocol"] == protocol
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one S6 recipe for {model!r}/{protocol}, found {len(matches)}")
    return matches[0]

def read_manifest(path: Path, protocol: str, allow_custom: bool = False) -> dict[str, list[dict[str, str]]]:
    digest = sha256_file(path)
    if not allow_custom and digest != REFERENCE_MANIFESTS[protocol]:
        raise ValueError(
            f"{path} does not match the Protocol {protocol} manifest "
            f"(SHA-256 {digest})"
        )
    required = {"file_name", "class", "split", "bytes", "sha256"}
    splits: dict[str, list[dict[str, str]]] = defaultdict(list)
    identities = set()
    digests = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest must contain {sorted(required)}")
        for row in reader:
            name = row["file_name"].strip()
            if not name or Path(name).name != name or Path(name).is_absolute():
                raise ValueError(f"unsafe manifest file_name: {name!r}")
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"unknown manifest split: {row['split']!r}")
            if row["class"] not in CLASSES:
                raise ValueError(f"unknown class: {row['class']!r}")
            identity = (row["class"], name)
            if identity in identities:
                raise ValueError(f"image appears more than once in manifest: {'/'.join(identity)!r}")
            identities.add(identity)
            try:
                if int(row["bytes"]) <= 0:
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"invalid byte count for {name!r}") from error
            if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
                raise ValueError(f"invalid SHA-256 for {name!r}")
            if row["sha256"] in digests:
                raise ValueError(f"duplicate image content in manifest: {name!r}")
            digests.add(row["sha256"])
            splits[row["split"]].append(row)
    if any(not splits[name] for name in ("train", "val", "test")):
        raise ValueError("manifest must have non-empty train, val, and test splits")
    return dict(splits)

def image_candidates(root: Path, row: Mapping[str, str], protocol: str) -> list[Path]:
    relative = Path(row["split"]) / row["class"] / row["file_name"]
    candidates = [
        root / f"lettuceDeF_protocol_{protocol}" / relative, root / f"protocol_{protocol}" / relative,
        root / protocol / relative, root / relative,
        root / row["class"] / row["file_name"],
    ]
    return list(dict.fromkeys(candidates))

def resolve_image(root: Path, row: Mapping[str, str], protocol: str) -> Path:
    for path in image_candidates(root, row, protocol):
        if path.is_file():
            return path
    tried = "\n  ".join(map(str, image_candidates(root, row, protocol)))
    raise FileNotFoundError(f"could not find {row['file_name']!r}; tried:\n  {tried}")

def verify_images(splits, root: Path, protocol: str, check: str) -> None:
    """Verify every image before model construction."""
    for rows in splits.values():
        for row in rows:
            path = resolve_image(root, row, protocol)
            expected_size = int(row["bytes"])
            if path.stat().st_size != expected_size:
                raise ValueError(f"size mismatch for {row['file_name']!r}")
            if check == "full" and sha256_file(path) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {row['file_name']!r}")

class ImageDataset(Dataset):
    def __init__(self, rows, root: Path, transform, protocol: str):
        self.rows, self.root, self.transform, self.protocol = rows, root, transform, protocol
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(resolve_image(self.root, row, self.protocol)) as source:
            image = source.convert("RGB")
        return self.transform(image), CLASSES.index(row["class"]), index

def build_transforms(recipe: Mapping[str, str], augmentation: Augmentation):
    """Build the exact resize, normalization, and optional RandAugment paths in S6."""
    size = int(recipe["effective_resolution_px"])
    interpolation = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST,
    }.get(recipe["interpolation"].lower())
    if interpolation is None:
        raise ValueError(f"unsupported interpolation: {recipe['interpolation']!r}")
    mean, std = json.loads(recipe["normalization_mean"]), json.loads(recipe["normalization_std"])
    normalise = transforms.Normalize(mean, std)
    common = [transforms.Resize((size, size), interpolation=interpolation)]
    evaluation = transforms.Compose(common + [transforms.ToTensor(), normalise])
    if augmentation.uses_randaugment:
        training = transforms.Compose(common + [
            transforms.PILToTensor(),
            transforms.RandAugment(num_ops=augmentation.rand_ops,
                                   magnitude=augmentation.rand_magnitude,
                                   num_magnitude_bins=augmentation.rand_bins),
            transforms.ConvertImageDtype(torch.float32), normalise,
        ])
    else:
        training = evaluation
    return training, evaluation

def layer_indices(description: str, block_count: int) -> list[int]:
    value = description.strip().lower()
    if value not in {"", "last", "last2", "last_2"}:
        raise ValueError(f"unsupported representation layers in S6: {description!r}")
    count = 2 if value in {"last2", "last_2"} else 1
    return list(range(max(0, block_count - count), block_count))

class LinearProbe(nn.Module):
    """Single linear head over the configured CLS- or patch-token representation."""
    def __init__(self, backbone: nn.Module, recipe: Mapping[str, str]):
        super().__init__()
        self.backbone = backbone
        self.indices = layer_indices(recipe["representation_layers"], len(backbone.blocks))
        pooling = recipe["effective_token_pooling"].lower()
        self.token = "patch" if "patch" in pooling else "cls"
        self.prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
        self.captured: dict[int, torch.Tensor] = {}
        for index in self.indices:
            backbone.blocks[index].register_forward_hook(self._capture(index))
        self.head = nn.Linear(int(backbone.num_features), len(CLASSES))
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        unfrozen = int(float(recipe["verified_unfrozen_blocks"] or 0))
        for block in list(backbone.blocks)[-unfrozen:] if unfrozen else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # A frozen-backbone probe trains only its new linear head. The final
        # normalization joins the trainable set when Transformer blocks do.
        if unfrozen and isinstance(getattr(backbone, "norm", None), nn.Module):
            for parameter in backbone.norm.parameters():
                parameter.requires_grad = True
    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self.captured[index] = output[0] if isinstance(output, (tuple, list)) else output
        return hook
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.captured.clear()
        self.backbone.forward_features(images)
        representations = []
        for index in self.indices:
            tokens = self.captured[index]
            if tokens.ndim != 3:
                raise RuntimeError(f"block {index} returned {tuple(tokens.shape)}, expected BxTxC")
            if self.token == "cls" and self.prefix_tokens:
                representations.append(tokens[:, 0])
            else:
                representations.append(tokens[:, self.prefix_tokens:].mean(1))
        return self.head(torch.stack(representations, dim=1).mean(1))

def classifier_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the complete classification head, including any head norm."""
    modules = []
    for name in ("head", "classifier", "fc"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        modules.append(classifier)
    unique = {}
    for module in modules:
        for parameter in module.parameters():
            unique[id(parameter)] = parameter
    return list(unique.values())

def adapt_native_model(model: nn.Module, recipe: Mapping[str, str]) -> None:
    head_ids = {id(parameter) for parameter in classifier_parameters(model)}
    blocks = recipe["verified_unfrozen_blocks"].strip()
    if recipe["family"] == "vFM" and blocks:
        for parameter in model.parameters():
            parameter.requires_grad = False
        unfrozen = list(model.blocks)[-int(float(blocks)):] if float(blocks) else []
        for block in unfrozen:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # Native ViT heads may normalize with either `norm` or `fc_norm`.
        for name in ("norm", "fc_norm"):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    else:
        parameters = list(model.parameters())
        percent = float(recipe["configured_trainable_percent"])
        trainable_count = 0 if percent <= 0 else max(1, int(len(parameters) * percent / 100))
        trainable_count = len(parameters) if percent >= 100 else trainable_count
        first_trainable = len(parameters) - trainable_count
        for index, parameter in enumerate(parameters):
            parameter.requires_grad = index >= first_trainable
    for parameter in model.parameters():
        if id(parameter) in head_ids:
            parameter.requires_grad = True

def verify_trainable_percentage(model: nn.Module, recipe: Mapping[str, str]) -> None:
    """Fail early if model construction differs from the S6 configuration."""
    expected = str(recipe.get("measured_trainable_percent") or "").strip()
    if not expected:
        return
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    measured = 100 * trainable / total
    if f"{measured:.4f}" != f"{float(expected):.4f}":
        raise RuntimeError(
            f"trainable-parameter check failed: S6 expects {float(expected):.4f}% "
            f"but this model build has {measured:.4f}% ({trainable:,}/{total:,}). "
            "Use the dependency versions in requirements.txt."
        )

def build_model(recipe: Mapping[str, str], pretrained: bool = True, factory=None) -> nn.Module:
    factory = factory or timm.create_model
    identifier = recipe["timm_identifier"]
    if "linearprobe" in recipe["head_construction"].lower():
        backbone = factory(identifier, pretrained=pretrained, num_classes=0)
        model = LinearProbe(backbone, recipe)
    else:
        model = factory(identifier, pretrained=pretrained, num_classes=len(CLASSES))
        adapt_native_model(model, recipe)
    verify_trainable_percentage(model, recipe)
    return model

def optimizer_groups(model: nn.Module, recipe: Mapping[str, str]):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    value = recipe["layer_decay"].strip()
    if not value:
        return trainable
    try:
        from timm.optim import param_groups_layer_decay
    except ImportError:  # timm < 1.0
        from timm.optim.optim_factory import param_groups_layer_decay
    no_decay = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = param_groups_layer_decay(model, weight_decay=float(recipe["weight_decay"]),
                                      layer_decay=float(value), no_weight_decay_list=no_decay)
    learning_rate = float(recipe["learning_rate"])
    return [
        {"params": [parameter for parameter in group["params"] if parameter.requires_grad],
         "lr": learning_rate * group.get("lr_scale", 1.0),
         "weight_decay": group.get("weight_decay", float(recipe["weight_decay"]))}
        for group in groups if any(parameter.requires_grad for parameter in group["params"])
    ]

class BatchMixer:
    """The manual batch-level Mixup/CutMix implementation used by the final runs."""
    def __init__(self, config: Augmentation):
        self.config = config
    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        cfg = self.config
        if np.random.rand() >= cfg.application_probability:
            return images, labels, labels, 1.0, False
        use_mixup = cfg.mixup_alpha > 0
        if cfg.mixup_alpha > 0 and cfg.cutmix_alpha > 0:
            use_mixup = np.random.rand() < cfg.mixup_probability
        alpha = cfg.mixup_alpha if use_mixup else cfg.cutmix_alpha
        coefficient = float(np.random.beta(alpha, alpha))
        order = torch.randperm(images.size(0), device=images.device)
        if use_mixup:
            images = coefficient * images + (1 - coefficient) * images[order]
        else:
            height, width = images.shape[-2:]
            ratio = math.sqrt(1 - coefficient)
            cut_width, cut_height = int(width * ratio), int(height * ratio)
            center_x, center_y = np.random.randint(width), np.random.randint(height)
            x1, x2 = np.clip([center_x - cut_width // 2,
                              center_x + cut_width // 2], 0, width)
            y1, y2 = np.clip([center_y - cut_height // 2,
                              center_y + cut_height // 2], 0, height)
            images[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
            coefficient = 1 - ((x2 - x1) * (y2 - y1) / (height * width))
        return images, labels, labels[order], float(coefficient), True

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def amp_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=enabled)

def amp_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

def classification_metrics(targets: list[int], predictions: list[int]):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    per_class = {}
    for index, name in enumerate(CLASSES):
        denominator = matrix[index, :].sum() + matrix[:, index].sum()
        per_class[name] = float(2 * matrix[index, index] / denominator) if denominator else 0.0
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean(list(per_class.values()))),
        "per_class_f1": per_class,
        "confusion_matrix": matrix.tolist(),
    }

def evaluate(model, loader, device, criterion=None, channels_last: bool = False):
    model.eval()
    targets, predictions, probabilities, row_indices = [], [], [], []
    loss_total, samples_seen = 0.0, 0
    with torch.no_grad():
        for images, labels, indices in loader:
            images, labels = images.to(device), labels.to(device)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits during evaluation")
            if criterion is not None:
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss during evaluation")
                loss_total += loss.float().item() * len(labels)
                samples_seen += len(labels)
            probs = logits.float().softmax(1).cpu()
            targets.extend(labels.cpu().tolist())
            predictions.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
            row_indices.extend(indices.tolist())
    result = classification_metrics(targets, predictions)
    result["loss"] = loss_total / samples_seen if criterion is not None else None
    return result, targets, predictions, probabilities, row_indices

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def save_outputs(output: Path, args, recipe, history, state, test_rows, evaluation) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metrics, targets, predicted, probabilities, indices = evaluation
    prediction_rows = []
    for target, prediction, probability, index in zip(targets, predicted, probabilities, indices):
        source = test_rows[index]
        row = {"file_name": source["file_name"], "true_class": CLASSES[target],
               "predicted_class": CLASSES[prediction]}
        row.update({f"probability_{name}": value for name, value in zip(CLASSES, probability)})
        prediction_rows.append(row)
    write_csv(output / "history.csv", history)
    write_csv(output / "predictions.csv", prediction_rows)
    torch.save({"model_state_dict": state, "model": args.model,
                "protocol": args.protocol, "seed": args.seed, "classes": CLASSES,
                "recipe_fingerprint": recipe["final_config_fingerprint"]},
               output / "checkpoint.pt")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "model": args.model,
        "timm_id": recipe["timm_identifier"],
        "family": recipe["family"],
        "protocol": args.protocol,
        "seed": args.seed,
        "recipe_fingerprint": recipe["final_config_fingerprint"],
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "recipes_sha256": args.recipes_sha256,
            "image_check": args.image_check,
        },
        "parameters": {
            "total": args.parameters_total,
            "trainable": args.parameters_trainable,
        },
        "epochs_completed": len(history),
        "best_epoch": min(history, key=lambda row: row["val_loss"])["epoch"],
        "best_validation_loss": min(row["val_loss"] for row in history),
        "test": metrics,
        "artifacts": ["result.json", "checkpoint.pt", "history.csv", "predictions.csv"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "device": getattr(args, "resolved_device", getattr(args, "device", "unknown")),
        },
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

def run(args) -> None:
    seed_everything(args.seed)

    required_outputs = ("result.json", "checkpoint.pt", "history.csv", "predictions.csv")
    if not args.dry_run and not args.overwrite and any(
        (args.output_dir / name).exists() for name in required_outputs
    ):
        raise FileExistsError(f"{args.output_dir} already contains a run; use --overwrite")

    args.manifest_sha256 = sha256_file(args.manifest)
    args.recipes_sha256 = sha256_file(args.recipes)
    recipe = read_recipe(args.recipes, args.model, args.protocol)
    splits = read_manifest(args.manifest, args.protocol, args.allow_custom_manifest)
    augmentation = Augmentation.from_recipe(recipe)
    train_transform, evaluation_transform = build_transforms(recipe, augmentation)
    verify_images(splits, args.images, args.protocol, args.image_check)
    if args.dry_run:
        print(f"configuration valid: {args.model} / protocol {args.protocol} / seed {args.seed}")
        print(f"manifest and images valid: {sum(map(len, splits.values()))} images ({args.image_check})")
        return
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device_name = "cpu" if device_name == "auto" else device_name
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    loaders = {}
    batch_size = int(recipe["batch_size"])
    for split, rows in splits.items():
        generator = torch.Generator().manual_seed(args.seed)
        loaders[split] = DataLoader(
            ImageDataset(rows, args.images,
                         train_transform if split == "train" else evaluation_transform,
                         args.protocol),
            batch_size=batch_size, shuffle=split == "train", num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=(split == "train" and len(rows) > batch_size
                       and len(rows) % batch_size == 1),
            generator=generator, worker_init_fn=seed_worker,
            persistent_workers=(split == "train" and args.workers > 0
                                and augmentation.uses_randaugment),
        )
    args.resolved_device = str(device)
    model = build_model(recipe).to(device)
    channels_last = recipe["family"] == "CNN" and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    args.parameters_total = sum(parameter.numel() for parameter in model.parameters())
    args.parameters_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    learning_rate = float(recipe["learning_rate"])
    weight_decay = float(recipe["weight_decay"])
    optimizer = torch.optim.AdamW(optimizer_groups(model, recipe), lr=learning_rate,
                                  weight_decay=weight_decay)
    epochs = int(recipe["epoch_cap"])
    warmup = int(recipe["warmup_epochs"])

    def schedule(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return MINIMUM_LR / learning_rate + (1 - MINIMUM_LR / learning_rate) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.CrossEntropyLoss(label_smoothing=augmentation.label_smoothing)
    mixer = BatchMixer(augmentation) if augmentation.uses_batch_mix else None
    amp = device.type == "cuda"
    scaler = amp_scaler(amp)
    patience = int(recipe["early_stopping_patience"])
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    print(f"{args.model} | protocol {args.protocol} | seed {args.seed} | {device}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss, samples_seen = 0.0, 0
        for images, labels, _indices in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            mixed = False
            if mixer is not None:
                images, first, second, coefficient, mixed = mixer(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(amp):
                logits = model(images)
                loss = criterion(logits, labels)
                if mixed:
                    loss = coefficient * criterion(logits, first) + (1 - coefficient) * criterion(logits, second)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss during training")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.detach().float().item() * len(labels)
            samples_seen += len(labels)
        scheduler.step()
        validation, *_ = evaluate(model, loaders["val"], device, criterion, channels_last)
        val_loss = validation["loss"]
        history.append({"epoch": epoch, "train_loss": training_loss / max(samples_seen, 1),
                        "val_loss": val_loss, "val_macro_f1": validation["macro_f1"],
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:3d}  val_loss={val_loss:.4f}  val_macro_f1={validation['macro_f1']:.4f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        elif epoch > warmup:
            stale += 1
            if stale >= patience:
                print(f"early stopping after {epoch} epochs")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    evaluation = evaluate(model, loaders["test"], device, criterion, channels_last)
    save_outputs(args.output_dir, args, recipe, history, best_state, splits["test"], evaluation)
    print(f"test macro-F1={evaluation[0]['macro_f1']:.4f}; outputs: {args.output_dir}")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="image archive root")
    parser.add_argument("--manifest", type=Path, required=True, help="protocol manifest CSV")
    parser.add_argument("--recipes", type=Path, required=True, help="Supplementary Table S6 CSV")
    parser.add_argument("--model", required=True, help="model name exactly as written in S6")
    parser.add_argument("--protocol", choices=["A", "B"], required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path,
                        required=True, help="directory for this run")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-check", choices=["full", "size"], default="full",
                        help="verify image hashes (full) or sizes only (size)")
    parser.add_argument("--allow-custom-manifest", action="store_true",
                        help="allow a non-release manifest (not a benchmark reproduction)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing run in --output-dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate recipe, manifest, and image paths without training")
    return parser.parse_args(argv)

if __name__ == "__main__":
    run(parse_args())
