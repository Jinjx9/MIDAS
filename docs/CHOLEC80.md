# Cholec80

Use authorized 1-fps, 2048-dimensional ResNet50 features. The fixed split is
videos 01--32 training, 33--40 validation, 41--80 testing. Feature arrays use
(channels,time). Do not mix these features with the SurgVU DINOv3 inputs.

## Explicit alignment

The retained input export has exactly one more terminal label than feature steps.
Preparation permits that case only when explicitly requested and records all
original/evaluated lengths. Other mismatches fail; source annotations are unchanged.

```bash
python -m midas.prepare --dataset cholec80 --raw-root RAW_C80 --features RAW_C80/features/asformer --drop-one-terminal --output PREPARED_C80
```

## Train

The recovered s44 recipe initializes from its ASFormer epoch70 parent, uses
seed44, Adam lr1e-4/weight decay1e-5, a 20-epoch cap and patience15. Both the parent
and selected MIDAS checkpoint must be separately obtained; downloads are pending.

```bash
python -m midas train --data-root PREPARED_C80 --features RAW_C80/features/asformer --train-split RAW_C80/splits/train.split1.bundle --val-split RAW_C80/splits/val.split1.bundle --parent ASFORMER_EPOCH70.model --output runs/c80_midas_s44 --seed 44 --epochs 20 --patience 15 --lr .0001 --lam .01 --backend historical_fast --input-dim 2048 --classes 7 --device cuda:0
```

## Predict and evaluate

```bash
python -m midas predict --features RAW_C80/features/asformer --split RAW_C80/splits/test.split1.bundle --checkpoint C80_S44_BEST.pt --lam .01 --backend historical_fast --output c80_predictions.json --device cuda:0
python -m midas evaluate --predictions c80_predictions.json --labels PREPARED_C80/groundTruth --mapping PREPARED_C80/mapping.txt --metric-protocol half_open --output c80_metrics.json
```

The released surgical evaluator uses pooled one-to-one segment F1, mean
segment-sequence Edit and frame-weighted Acc. Jaccard outputs name video-macro
and global-class-macro aggregation separately. Use a consistent aggregation
across comparisons; do not combine values from different definitions.
