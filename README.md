# LettuceDeF benchmark reproducer

This repository implements the four-class Protocol A/B benchmark reported in
*Beyond random splits: Temporal robustness of vision models for lettuce
nutrient-restriction classification*.

The workflow has three commands:

1. `reproduce.py` trains and evaluates one model--protocol--seed run.
2. `run_benchmark.py` launches any subset of the 312 benchmark runs.
3. `summarize_results.py` summarizes completed runs and pairs Protocol A with
   Protocol B by model and seed.

## Installation

Requirements:

- Python 3.11 or 3.12
- PyTorch and torchvision
- timm, NumPy, and Pillow
- a CUDA GPU is recommended

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA 12.6 on Tesla V100 GPUs:

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

`timm` downloads the pretrained weights on first use. `PRETRAINED_WEIGHTS.csv`
lists the source repository, revision, filename, and SHA-256 for each checkpoint.

`requirements.txt` pins the tested release environment. The experiments reported
in the manuscript used Python 3.11.5, PyTorch 2.1.2, torchvision 0.16.2,
timm 1.0.27, NumPy 1.26.0, CUDA 12.2, and a Tesla V100-SXM2-32GB. Use 24
data-loader workers to match the reported setup; `--workers` can be lowered on
smaller machines.

## Dataset

The repository includes:

- `manifest_protocol_A.csv`
- `manifest_protocol_B.csv`
- `S6_selected_recipes.csv`

Download [LettuceDeF from Zenodo](https://doi.org/10.5281/zenodo.21898482),
extract `LettuceDeF_images_v1.0.0.zip`, and pass the extracted `images/` directory
to `--images`:

```text
images/
  FN/
  K-deficiency/
  N-deficiency/
  P-deficiency/
```

The manifests and Table S6 recipe file define the evaluated splits and
configurations. The scripts validate these inputs automatically before training.

## Run one model

```bash
python reproduce.py \
  --images /path/to/images \
  --manifest manifest_protocol_B.csv \
  --recipes S6_selected_recipes.csv \
  --model MobileNetV2 \
  --protocol B \
  --seed 42 \
  --output-dir results/mobilenetv2/protocol_B/seed_42
```

Each run uses the selected configuration from Table S6 and writes `result.json`,
per-image test predictions, training history, and the best validation checkpoint.

## Run the benchmark

Inspect the complete plan without starting training:

```bash
python run_benchmark.py \
  --images /path/to/images \
  --manifest-a manifest_protocol_A.csv \
  --manifest-b manifest_protocol_B.csv \
  --recipes S6_selected_recipes.csv \
  --output results \
  --dry-run
```

Remove `--dry-run` to execute the plan. The complete benchmark contains 13 models,
two protocols, and 12 seeds (312 training runs). Use `--model`, `--protocol`, or
`--seed` to run a subset, and `--resume` to continue an interrupted benchmark.

For one GPU, keep the default of one process. Use separate launcher invocations to
assign work to multiple GPUs.

## Summarize results

```bash
python summarize_results.py --input results --output summary
```

The summary reports model/protocol means and standard deviations, paired A-minus-B
differences by seed, and the aggregate paired difference.

## Citation

Please cite the associated preprint:

> Hernández-Sandoval, M., Salazar-Colores, S., Ramírez-Pedraza, A.,
> Martínez-Ponce, G., Rodríguez-Reséndiz, J., and Valentín-Coronado, L. (2026).
> *Beyond random splits: Temporal robustness of vision foundation models for
> lettuce nutritional diagnosis*. https://doi.org/10.2139/ssrn.6876395

The same citation is available in `CITATION.cff`.

## License

The code is released under the MIT License. The LettuceDeF dataset is distributed
separately on [Zenodo](https://doi.org/10.5281/zenodo.21898482) under CC BY 4.0.
