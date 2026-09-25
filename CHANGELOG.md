# Changelog

## 2.2.1+pmf.1 — 2026-09-25

- Add PMF with masked mean pooling, train-only standardization, three 128-dimensional projections, fusion MLP and polarity/magnitude heads.
- Restore the original implementations of the other 14 previously modified models.
- Record raw Acc3, MacroF1, signed-intensity MAE and Corr; keep undefined Corr missing.
- Add independent per-seed CSV rows and summaries for valid/test/optional attachment-3 test2.
- Preserve per-run/per-seed checkpoints and full JSON configuration; load saved normalization during inference.
- Fix CPU execution, device-independent checkpoint loading and repeated logging handlers.
- Add standalone training and prediction scripts with optional external test2 input; declare missing einops dependency and tested core constraints.
- Add unit/integration tests, portable documentation, data/output ignore rules and test-only CI.
