# Release checklist

Completed implementation scope:
- Main MIDAS model, segmentation train/predict/evaluate and label preparation.
- Dataset guides and retained 50Salads recipes.
- SurgVU main-method training, label-free prediction and dataset-level AP.
- CPU tests, CI configuration, MIT license and upstream notices.

Before public release:
- SurgVU full preflight and 30-epoch GPU run are complete; selected weights,
  run metadata, predictions and metrics are retained, with the selected model
  and its SHA256 included in the repository.
- Other dataset weights remain unhosted; their guides state the current scope.
- Inspect the final tracked tree and Git history for credentials and restricted data.
- Author approval to change visibility. No public visibility change is automatic.

Private historical audit records remain separate from the quick-start guide.
