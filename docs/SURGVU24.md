# SurgVU24: main-method MIDAS run

This pipeline reuses `midas.model.MIDAS`, including its ASFormer encoder, three
decoders and state inference after the first two decoders. It does not use the
early residual-ConvNet experiment. This is a reconstructed, newly specified run,
not recovered original code or a promise to reproduce a historical score.

## Data access and format

Obtain SurgVU data under the dataset's terms and authorization. We do not bundle
videos, clinical annotations, frames, features or account credentials. Existing
authorized 1-fps DINOv3 ViT-L/16 features have 1024 channels. Each `.npy` is
float32 `(time, 1024)`, named `case_000.npy`, etc. Use the exact authorized
tasks7 annotation JSON and extractor checkpoint, not unrelated clip features.
The JSON has `database[video]` records with `subset` (train/val/test), `duration`
in seconds, `fps: 1`, and `annotations: [{label, segment: [start,end]}]`.

The retained split has 100 training, 25 validation and 30 test cases. Preparation
preserves these assignments. Seven foreground classes are listed in the config;
background is class 7. Do not create a new random split or infer class IDs from
the test data. Feature length must equal `int(duration)`; mismatches fail rather
than silently crop. Resolve any failure against feature provenance before running.

Features are normalized per video/channel with `(x-mean)/(std+1e-6)` over time,
including the full offline video. Annotation endpoints are floored for framewise
training targets. The authorized original annotation has cross-class overlaps in
case_049 and case_056. The config explicitly retains the old target-construction
rule, `last_annotation_wins`, in JSON annotation order. Evaluation retains ALL
original fractional intervals. Original annotations are never overwritten.

```bash
python -m midas.tal prepare --annotations /data/surgvu/tasks7_1fps.json --config configs/surgvu24/midas.json --output /data/surgvu/prepared
python -m midas.tal preflight --annotations /data/surgvu/tasks7_1fps.json --features /data/surgvu/features --config configs/surgvu24/midas.json --output preflight.json
```

Preflight checks every case and writes split counts, lengths and SHA256 hashes.
It does not evaluate a model. `prepare` writes only split bundles and metadata,
not copies of restricted annotations or features. Use fresh output paths.

## Train

```bash
python -m midas.tal train --annotations /data/surgvu/tasks7_1fps.json --features /data/surgvu/features --config configs/surgvu24/midas.json --output runs/surgvu_midas_seed19980125 --device cuda:0 --threads 4
```

The fixed configuration uses seed 19980125, scratch initialization, 30 epochs,
Adam lr 1e-4/weight decay 1e-5, gradient norm clipping at 5, full sequences and
the existing MIDAS multi-stage CE + 0.15 TMSE objective. No additional TAL loss
or learnable module is introduced. The ASFormer backbone has 10 layers, 64
channels and 3 refinement decoders; the unchanged backbone's channel dropout
is 0.3. Fast RTS uses lambda 1e-4 and the retained unfold attention implementation.
TF32 is disabled and the configured precision/runtime choices are recorded.

Model selection uses strictly highest validation average AP; ties retain the
earlier epoch. Test features and test predictions are not used by training.
The run produces `config.json`, `run_manifest.json`, `metrics.jsonl`, `last.pt`,
`best_val.pt`, and `selection.json`. Run metadata binds source, train/val input,
annotation and selected-checkpoint hashes. No optimizer/RNG resume is advertised;
never restart training into an existing output directory.

## Predict, then evaluate once

```bash
python -m midas.tal predict --features /data/surgvu/features --split /data/surgvu/prepared/test.bundle --checkpoint runs/surgvu_midas_seed19980125/best_val.pt --output surgvu_predictions.json --device cuda:0 --threads 4
python -m midas.tal evaluate --predictions surgvu_predictions.json --annotations /data/surgvu/tasks7_1fps.json --subset test --output surgvu_metrics.json
```

Prediction requires no labels or annotations. The checkpoint carries its model,
class, normalization and readout settings. The output records its SHA256 and
selected epoch. Evaluation rejects missing, extra or duplicated case IDs.

Readout applies foreground softmax probability >0.5, forms half-open connected
intervals lasting at least 3 seconds, scores each with its mean probability and
keeps the highest-scoring 100 per class/video. AP ranks detections across the
entire subset separately for each class, uses one-to-one same-video matching,
and integrates the all-points precision envelope at tIoU .3/.4/.5/.6/.7. Classes
without ground truth are explicitly excluded. This is dataset-level detection
AP, not an average of per-video AP. The evaluator has a named protocol version;
its values must not be silently mixed with another implementation.

## Completed run and checkpoint

The new run completed all 30 epochs on a Tesla V100 PCIe 16GB, with Python
3.11.8, PyTorch 2.5.1+cu121 and NumPy 1.26.4. Full input preflight verified all
155 cases (100/25/30); the original feature archive checksum and the case071
repair checksum matched. Epoch 30 was selected using validation average mAP
(52.046387635205306). Its test average mAP is 49.88950544500655, with mAP at
0.3/0.4/0.5/0.6/0.7 equal to 60.66259101540698 / 57.83398706330461 /
54.727194038058755 / 41.86647859466592 / 34.35727651359645.

Checkpoint: `weights/surgvu24_midas_seed19980125_best_val.pt` (included in Git).
SHA256: `bd12425de79fe4137be3c90e21d7e73344a55f724f28dafd3fe3d4b586c8c484`.
This documents a new run; it does not establish equivalence to historical table
entries or baseline results computed with another evaluator.

The matched no-MIDAS control has also completed 30 epochs with the same seed,
data and settings, with only `model.mid` disabled. Validation selected epoch18
(validation average mAP 49.69126152279366). Its test average mAP is
46.69627470011926, compared with MIDAS 49.88950544500655 (gain 3.19 points).
MIDAS is lower at tIoU0.6 (41.87 vs 43.31); gains are not uniform across thresholds.
Control config: `configs/surgvu24/asformer.json`; checkpoint:
`weights/surgvu24_asformer_seed19980125_best_val.pt`; SHA256:
`2582ccca208c50cbff7229a7f11e416ec507056e0e061587ae489d7e05d4cb11`.
Use the same prediction command below with this checkpoint to evaluate the control.

All six methods' retained detections were evaluated with the same released AP
implementation. The new final-checkpoint exports give average mAP 5.47 for
ActionFormer, 22.95 for TemporalMaxer, and 31.40 for OphBiWSSD; retained TriDet
detections give 32.34. See `results/surgvu24_unified_comparison_20261021.json`.
Method-specific postprocessing and validation-selected weights are retained.
ActionFormer uses the final epoch35 EMA checkpoint; its earlier raw export is
not the final comparison. OphBiWSSD retains the ordinary state dictionary used
by its archived final-test script; TemporalMaxer retains EMA weights.

With authorized features and the prepared split, evaluate the included model:

```bash
python -m midas.tal predict --features /data/surgvu/features --split configs/surgvu24/splits/test.bundle --checkpoint weights/surgvu24_midas_seed19980125_best_val.pt --output surgvu_predictions.json --device cuda:0 --threads 4
python -m midas.tal evaluate --predictions surgvu_predictions.json --annotations /data/surgvu/tasks7_1fps.json --subset test --output surgvu_metrics.json
```

Verify the checkpoint before use with `sha256sum` (Linux) or
`Get-FileHash -Algorithm SHA256` (PowerShell). Predictions and restricted inputs
are retained privately and are not distributed in this repository.

## Checklist for another GPU run

The synthetic end-to-end test is not a benchmark run. On the actual server:

1. Install the pinned core environment; run all tests.
2. Download only authorized data through your own authenticated storage client.
3. Run preparation and full-data preflight; inspect counts and any alignment error.
4. Time an initial training/validation epoch and inspect peak VRAM before estimating
   the full budget. The old ConvNet timing is NOT an ASFormer timing estimate.
5. Freeze the config before test evaluation. Preserve the complete run directory,
   test predictions, metrics and source commit; copy them off the rented disk.

Do not change sampling, sequence lengths, epoch budget, feature type or precision
to meet a deadline without recording a new experimental configuration. Results
for another run must be measured independently; no exact retraining score is promised.
