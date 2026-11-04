# AutoLaparo

Use authorized pre-extracted RGB+Flow I3D features, concatenated into 2048
channels at 1 fps. Input arrays are `(2048,time)`. The fixed video split is
01--10 training, 11--14 validation, 15--21 testing; class IDs must follow the
authorized seven-class mapping, not a new alphabetical ordering.

```bash
python -m midas.prepare --dataset autolaparo --raw-root RAW_AL --features FEATURES --output PREPARED_AL
python -m midas train --data-root PREPARED_AL --features FEATURES --train-split RAW_AL/splits/train.split1.bundle --val-split RAW_AL/splits/val.split1.bundle --parent PARENT_ASFORMER.model --output runs/al_midas --input-dim 2048 --classes 7 --layers 10 --width 64 --lam .0001 --backend sequential --seed 20260303 --epochs 60 --patience 15 --device cuda:0
python -m midas predict --features FEATURES --split RAW_AL/splits/test.split1.bundle --checkpoint SELECTED_MIDAS.pt --lam .0001 --backend sequential --output al_predictions.json --device cuda:0
python -m midas evaluate --predictions al_predictions.json --labels PREPARED_AL/groundTruth --mapping PREPARED_AL/mapping.txt --output al_metrics.json
```

The train command documents the portable recipe interface with a 60-epoch cap;
the exact selected historical parent and original complete training budget are
not established by this command. Do not claim scratch reproduction of the paper
from this example. Selected-weight inference was independently verified on all
seven retained test inputs. Parent and selected-weight download links are pending.

The preparation command requires original mapping and groundTruth folders;
it never resamples features or silently crops misaligned labels. Surgical F1
uses pooled one-to-one matching; use the named video-macro Jaccard output when
comparing against video-macro Jaccard reports.
