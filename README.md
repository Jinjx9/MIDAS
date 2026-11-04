# MIDAS

Official implementation of MIDAS for multi-stage surgical phase segmentation.

MIDAS estimates temporally coherent class evidence between decoder stages using
Kalman filtering and Rauch--Tung--Striebel smoothing. It reuses ASFormer's encoder
and refinement decoders, adds no learnable parameters, and uses offline temporal
context. This repository provides method code, configurations, data preparation,
and training/inference/evaluation commands.

## Installation

Python 3.10 is the tested version. Install a compatible PyTorch build for your
machine first; the verified core environment is PyTorch 2.5.1 and NumPy 1.26.4.

```bash
git clone https://github.com/Jinjx9/MIDAS.git
cd MIDAS
python -m pip install -r requirements-eval-tested.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

Datasets are not included. See the dataset guides below for preparation instructions.

## Dataset guides

| Dataset | Guide | Evaluation |
| --- | --- | --- |
| AutoLaparo | [Preparation and commands](docs/AUTOLAPARO.md) | pooled one-to-one F1, video Edit, frame accuracy |
| Cholec80 | [Preparation and commands](docs/CHOLEC80.md) | same surgical segmentation evaluator |
| 50Salads | [Custom split and recipes](docs/50SALADS.md) | explicitly selected official-ASFormer protocol |
| SurgVU24 | [New main-method run](docs/SURGVU24.md) | dataset-level temporal detection AP |

All datasets must be obtained under their own terms. Do not redistribute
restricted clinical data, extracted features or annotations without permission.
Precomputed features are required; this repository does not bundle image/video
backbones or claim to regenerate every historical feature extraction.

## Segmentation quick start

Expected input layout:

```text
DATA_ROOT/
  features/video01.npy       # float32, channels x time
  groundTruth/video01.txt    # one class name per step
  mapping.txt               # contiguous IDs: 0 class_name
  splits/train.bundle
  splits/val.bundle
  splits/test.bundle
```

```bash
python -m midas train --data-root DATA_ROOT --train-split DATA_ROOT/splits/train.bundle --val-split DATA_ROOT/splits/val.bundle --output runs/example --input-dim 2048 --classes 7 --layers 10 --width 64 --lam 0.0001 --epochs 60 --seed 20260303 --device cuda:0
python -m midas predict --features DATA_ROOT/features --split DATA_ROOT/splits/test.bundle --checkpoint runs/example/best_val.pt --output predictions.json --device cuda:0
python -m midas evaluate --predictions predictions.json --labels DATA_ROOT/groundTruth --mapping DATA_ROOT/mapping.txt --output metrics.json
```

This is an interface example, not a universal paper recipe. Parent initialization,
backend and training budgets are dataset-specific; follow the guides. Prediction
does not read labels. Checkpoint selection uses validation data. Training output
directories must be new. Optimizer/RNG resume is not implemented.

## SurgVU main-method run

```bash
python -m midas.tal --help
python -m midas.tal train --annotations ANNOTATIONS.json --features FEATURES --config configs/surgvu24/midas.json --output runs/surgvu_midas_seed19980125 --device cuda:0 --threads 4
```

Run the preparation and preflight steps in [the guide](docs/SURGVU24.md) first.
The localization pipeline uses the same MIDAS class as segmentation; it does not
substitute the early ConvNet experiment. The new 30-epoch run completed on a
V100 16GB with Python 3.11.8, PyTorch 2.5.1+cu121 and NumPy 1.26.4.
Validation selected epoch 30; test average mAP is 49.89 under the documented
dataset-level AP evaluator. See [the run record](results/surgvu24_run_20261021.json).
This is a new run, not a claim to reproduce the historical paper-table score.
The matched no-MIDAS control is also included: average mAP 46.70 versus MIDAS
49.89. The [unified comparison](results/surgvu24_unified_comparison_20261021.json)
records all six methods under the same detection AP evaluator.

## Models and reproducibility

The [SurgVU checkpoint](weights/surgvu24_midas_seed19980125_best_val.pt) is included
in this checkout; its SHA256 and evaluation commands are in the dataset guide.
Other checkpoints are not yet hosted; their documented training commands remain
the entry point. Existing
AutoLaparo and 50Salads fixed-checkpoint predictions were locally checked against
retained outputs. The new SurgVU run has a saved checkpoint and evaluation record.
See [verification scope](docs/REPRODUCTION_STATUS.md).

## Repository layout

- `midas/`: ASFormer wrapper, segmentation CLI, localization pipeline and metrics.
- `core/`: sequential/fast state inference and retained attention utilities.
- `configs/`: dataset-specific recipes and permitted split identifiers.
- `tests/`: numerical, data-contract and end-to-end CPU checks.
- `docs/`: data and execution guides.
- `LICENSES/`: third-party notices.

## License and attribution

New project code is released under the MIT license. ASFormer retains its original
MIT copyright notice; see [third-party attribution](docs/THIRD_PARTY.md).
The original algorithms and backbone architectures remain credited to their
authors. No accepted venue, DOI or publication date is asserted by this repository.
