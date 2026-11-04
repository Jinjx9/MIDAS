# 50Salads custom fold2 experiment

This is a 32/8/10 subject-disjoint training/validation/test split using the fold2
test videos. It is NOT the standard five-fold average or the original 19-class
benchmark. The final experiment uses only the split files in
`configs/50salads/splits/`: 32 training, eight validation, and ten test videos.
Test videos are excluded from training and model selection; checkpoints are
selected using validation F1@50 only.

## Required authorized inputs

Obtain the original full-resolution I3D feature sequences and 19-class labels
under the dataset's terms. Each feature file is float32 (2048, time). The original
mapping is IDs0–18; action_start17 and action_end18 are merged into background17.
Features are not subsampled, chunked or re-extracted by the commands below.
The dataset, raw labels and feature files are not bundled in this repository.
The selected checkpoints were recovered locally but a distribution URL has not
yet been approved; this remains a release-access gate.

## Prepare labels without modifying the originals

```bash
python -m midas.prepare --raw-root RAW_50SALADS --output PREPARED_LABELS
```

RAW_50SALADS contains mapping.txt, groundTruth/*.txt and features/*.npy. Preparation
merges labels only and keeps the timeline unchanged. Keep features at their original
location. Existing outputs are not overwritten. A partial local dataset prepares
only its available labels; full training requires all 50 videos.

## Evaluate the fixed selected weights

```bash
python -m midas predict --features RAW_50SALADS/features --split configs/50salads/splits/test.bundle --checkpoint MIDAS_BEST.pt --recipe configs/50salads/midas.json --output midas_test.json --device cuda:0
python -m midas evaluate --predictions midas_test.json --labels PREPARED_LABELS/groundTruth --mapping PREPARED_LABELS/mapping.txt --metric-protocol official_asformer --background 17 --output midas_metrics.json
```

For ASFormer use configs/50salads/asformer.json and its own selected checkpoint.
Recipes verify checkpoint SHA256 before inference; they set the historical fast
smoother and unfold attention implementation explicitly and disable both matmul
and cuDNN TF32, matching the formal test program. Leaving cuDNN TF32 at its system
default produced five differing labels in 115,955 time steps on the tested setup;
the declared precision policy is part of the recipe. Do not use the old epoch40
weights: the formal selected epochs are MIDAS73 and ASFormer78.

The official-ASFormer metric endpoint and greedy matching behavior are preserved
separately from the half-open evaluator used for the recovered surgical results.
Background is included in Acc, excluded from F1 and Edit. Expected unrounded metrics
are recorded in each recipe; fixed-checkpoint inference status is in
REPRODUCTION_STATUS.md. Jaccard is not a reported metric for this experiment.

## Training settings recovered from the formal run

```bash
python -m midas train --data-root PREPARED_LABELS --features RAW_50SALADS/features --train-split configs/50salads/splits/train.bundle --val-split configs/50salads/splits/val.bundle --output RUN_MIDAS --device cuda:0 --input-dim 2048 --classes 18 --layers 10 --width 64 --seed 19980125 --lam .0003 --backend historical_fast --attention-impl unfold --lr .0005 --epochs 120 --patience 0 --grad-clip 0 --shuffle python --deterministic --metric-protocol official_asformer --background 17
```

Omit parent initialization (scratch). Add --no-mid for the paired ASFormer run.
Both runs use Adam, weight decay1e-5, stage CE plus TMSE0.15, train-loss plateau
scheduler (factor0.5, patience3), full sequences, batch1 and dropout0.3. Save the
strictly best validation F1@50; retain the earlier epoch in ties. Never use test
metrics for selection. A fresh full 120-epoch training run has not been repeated
as part of repository packaging; identical training trajectories across PyTorch
versions/hardware are not promised. Historical training used PyTorch2.2.1+cu121;
current evaluation tests use PyTorch2.5.1+cu121.
