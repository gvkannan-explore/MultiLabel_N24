# Tests

The tests run offline on CPU and never download pretrained weights. Where a
model is needed they build a tiny RoBERTa on synthetic splits.

- `tests/test_baseline_model.py`: data preparation (sorted sections,
  whitespace cleaning, rejected records) and the trainer (metrics match
  scikit-learn; a smoke run writes `best_model/` and `metrics.json`).
- `tests/test_import_run.py`: the MLflow import logs scores, parameters, tags,
  and curves, and skips a run that is already in the store.
- `tests/test_deployment_gate.py`: the eval harness's results file,
  repeatability, and refusal to overwrite, and every pass, fail, and refusal
  path of the deployment gate.
- `tests/test_deploy.py`: freezing a run copies only what the inference API
  reads, serves from the copy, and rejects other weights or a non-encoder
  checkpoint.
- `tests/test_privacy.py`: the privacy scanner and the identity guard.
- `tests/test_legacy.py`: the current behavior of the legacy code.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python -m unittest discover -s tests -v
```
