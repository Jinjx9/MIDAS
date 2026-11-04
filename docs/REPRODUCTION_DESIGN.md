# Paper reproduction release: proposed acceptance contract

Status: original audit contract, retained as historical design. Later on
2026-10-20 the author approved a reconstructed main-method SurgVU pipeline and
a NEW run when the historical code/weights could not be recovered. The new
pipeline is not claimed to recover an exact historical score. Public runtime
keeps only unified surgical segmentation evaluation; legacy evidence is retained
privately outside this checkout. See SURGVU24.md for the current run contract.

## Objective

A reader with authorized datasets can clone this repository, install a documented
environment, prepare the exact inputs, run fixed checkpoints, and compare outputs
and metrics with the paper. Training recipes must preserve historical choices;
successful synthetic tests alone do not establish benchmark reproduction.

## Approaches considered

1. Recommended: recover the historical pipelines and bind each result to its
   checkpoint, features, split, evaluator and configuration. Verify inference
   first, then package training and preparation. This preserves scientific provenance.
2. Checkpoint-only release: quicker, but cannot meet the requested training/data
   preparation completeness; useful only as an intermediate milestone.
3. Retrain new configurations to obtain convenient reproducible results: rejected
   for this task. New results cannot silently replace the reported experiments.

## Deliverables and verification gates

### A. Result registry and source binding

For AutoLaparo, Cholec80, 50Salads and SurgVU24, record checkpoint SHA256, selected
epoch, seed, architecture, smoothing backend and lambda, exact input feature
provenance, train/validation/test lists, class/background mapping, readout and
metric definitions. Each registry entry has a status: verified, discrepant or
missing asset. Never label an entry verified from rounded table values alone.

Existing findings requiring investigation:

- AutoLaparo has formal weights and recovered per-video inputs locally; current
  synthetic equivalence is not yet full-video equivalence.
- Cholec80's historical 86.12 F1@50 report and a later archived fixed-s44 evaluation
  differ. The historical report uses a different F1 implementation. Feature and
  runtime binding must also be checked because accuracy differs between records.
  Do not infer that the metric implementation alone explains all differences.
- The 50Salads summary preserves splits, counts and result bindings; locate the
  actual paired-run code, selected weights and feature manifests before claiming
  that this repository reproduces it. Preserve the custom-split and label-mapping
  specification; this experiment does not report the standard five-fold average.
- SurgVU24 has both convolutional and ASFormer-related historical sources.
  Identify the model actually used by the final table, weights and unified AP
  evaluator. Do not substitute whichever runtime is easiest to reconstruct.

### B. Portable evaluation

Provide explicit per-dataset configurations, sequential/fast backend selection,
strict checkpoint loading, label-free inference, segmentation metrics including
the actual Jaccard aggregation, background-aware 50Salads evaluation, and the
verified localization readout/AP implementation. Validation and test must remain
separate. Unexpected IDs, mismatched times and missing videos fail explicitly.

Compare actual full-test predictions with archived outputs where available;
otherwise report the exact limitation. Compare raw logits within a stated
tolerance, discrete predictions exactly, and unrounded metrics before display
rounding. A mismatch produces a report, not a new checkpoint selection.

### C. Data preparation and model access

Document legitimate dataset access, exact feature extraction checkpoints,
sampling, temporal alignment, preprocessing and output layout. Provide portable
preparation/extraction commands and integrity manifests without distributing
restricted videos, surgical images, labels or feature caches without permission.

Selected model weights require a deliberately chosen distribution mechanism and
permission review. Do not upload all local weights or publish them automatically.
If an asset is unavailable, state that reproducibility remains incomplete.

### D. Training, environment and fresh-clone checks

Package the actual recovered recipes, parent initialization, optimizer/scheduler,
stopping/selection policy, budgets and seeds. Add optimizer/RNG resume only if
its semantics are documented and tested. Do not claim bitwise training equality
across hardware. Provide a tested dependency lock plus platform instructions,
small CPU integration tests, backend equivalence tests and clean-clone checks.
Do not spend rented GPU money or launch full retraining without further approval.

### E. Release and manuscript handoff

README supplies complete commands, expected outputs, resource requirements,
result registry, reproduction status and known limitations. Keep attribution and
upstream notices; the author must approve the license for new code. A citation
entry must not invent an accepted venue, DOI or author list.

Keep the repository private throughout this task. Public release, weight hosting,
an anonymous review copy and manuscript link placement need separate decisions.
Do not change paper scores or scientific claims to conceal a discrepancy.

## Scope and stopping conditions

Reproduce MIDAS and its claimed ablations/backbone references, not every competing
method's training implementation. Existing historical baseline evidence remains
private and intact. No dataset deletion or legacy cleanup during this work.

If historical code, inputs or weights cannot be recovered, identify the exact
missing asset and last completed gate. If the paper and recovered evidence
disagree, present the discrepancy for author review before changing the paper.
Completion requires verified result bindings, runnable commands and available
authorized assets; documentation alone is not sufficient.
