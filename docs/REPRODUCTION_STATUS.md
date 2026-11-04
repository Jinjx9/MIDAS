# Verification scope

## Local checks

- Independent synthetic CPU training, saving, label-free prediction and evaluation.
- AutoLaparo: retained selected-checkpoint predictions match all seven test
  videos (28,060 steps). This checks existing inputs, not fresh feature extraction.
- Cholec80: selected checkpoint loads strictly and runs with its declared fast
  backend. The released evaluator is the unified surgical segmentation evaluator.
  See the private audit for historical reporting provenance; do not infer all
  historical table entries from a generic test success.
- 50Salads: both selected models match retained predictions on all ten test
  videos (115,955 steps each). This is the documented custom split, not a
  standard five-fold average. Full scratch training has not been rerun here.
- SurgVU24: the new 30-epoch main-method run completed; full input preflight,
  selected checkpoint, predictions and test metrics are retained. The checkpoint
  is included in `weights/`; see `results/surgvu24_run_20261021.json`.
  This is separate from historical paper-table reproduction.

## Distribution

The repository provides code, commands and the new SurgVU checkpoint.
Other dataset weights are not yet hosted.
Videos, annotations and features are not included. Users must obtain authorized
inputs. No optimizer/RNG resume or bitwise cross-hardware training reproduction
is promised. License: MIT for new code, preserved upstream notices for ASFormer.

The repository stays private until the author approves final publication after
the new run and release checks. Never advertise every historical paper table as
independently reproduced on the basis of synthetic tests.
