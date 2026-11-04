# C2F-TCN AutoLaparo fixed-run protocol

Date 2026-10-13. Seed 20260303. No MIDAS modifications. No test-driven tuning.

Data: existing 2048-dimensional I3D RGB+flow at 1Hz; original21 feature files restored from validated pre-decommission backup. Train video01–10, val11–14, test15–21. GT/feature lengths must match exactly. No alignment truncation. Training may read metadata for test length checking, not predictions or selection metrics.

Upstream https://github.com/dipika-singhania/C2F-TCN. Server clone commit and license retained separately. Original C2F_TCN network, AugmentDataset temporal augmentation, CriterionClass CE+0.15 TMSE and ensemble weights [1,1,1,1,0,0] retained. Only unused imports are stripped using AST for environment compatibility; adapted sources and diffs saved. No testing-time augmentation.

500epochs; Adam lr0.0005, weight_decay0.0003, StepLR step500 gamma0.5 (the official small-dataset optimizer preset, selected before any run). Batch1 for memory compatibility;10 optimizer steps/epoch. Official original video-frame chunk sizes are not transplanted: chunk1 corresponds to1second, augmentation can choose chunk1–2 using official zoom0.5–2. Fixed max_frames6400 covers observed longest6388seconds, so no video splitting or tail cropping. Padding ignored in CE and temporal mask per original criterion; validation/test counts remove only loader padding, not original video samples.

Selection: validation F1@50 descending, Edit descending, earlier epoch tie.500epochs execute fixed budget; best and last saved, optimizer/scheduler/RNG states support resume. Test only after budget ends using selected best. Validation measured by project metric_reference_source.py half-open evaluator snapshot. No Jaccard claim.

Before formal initialization: verify every transferred SHA256, all split sets, mapping and feature shapes.12-step smoke fit on first512samples of trainvideo01 must have finite loss, nonzero finite gradient, correct outputshape, reduced loss. Formal model and RNG reinitialized after smoke. If smoke/OOM fails, stop and report, not publish shortened run as formal.

Outputs: /hy-tmp/tas_reproduction_20261013/results/C2F-TCN/autolaparo_seed20260303. Ensemble probabilities explicitly named probability, not logits. Post-run labels and all test probabilities saved. Internal-event/omission analysis and per-video metric exports remain the subsequent evaluation step, not accomplished merely by starting training.

Environment isolated venv with existing server torch2.2.1 CUDA12.1 exposed via system-site-packages; does not modify the server's original torch environment. GPU V10032GB. Epoch time to be measured on first formal epoch; not invented.
