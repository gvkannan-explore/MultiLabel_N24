# Deploying to Modal

A push to `main` that adds or changes one file under `results/nlp/` runs
`.github/workflows/deploy.yml`, which calls `bins/ci_tasks/deploy.sh`. Only
trained encoder runs write there. Zero-shot results go to `results/llm/`,
outside the workflow's path filter, so they never deploy, even when they score
higher ([ADR 0008](adr/0008-frozen-modal-image-deploys-encoder-runs-only.md)).

1. The deployment gate judges the results file. Any exit other than 0 stops
   the deploy, and the HTML report is uploaded as the `gate-report` artifact.
2. `modal deploy -m n24.deploy.modal_app` builds a frozen image. Its build
   step (`n24/deploy/freeze.py`) copies the run's `metrics.json` and
   `best_model/` out of the `n24-runs` volume into the image, after checking
   that the checkpoint's `model_type` is a BERT-family encoder and that its
   weights sha256 is the one the deployment gate judged. A deployed version
   never reads the volume again, so retraining under the same name cannot
   change what it serves.
3. `bins/api/check_api.sh` checks the live endpoint, including that a request
   without proxy auth is rejected with 401.

A push that changes two `results/nlp/` files fails: one deployment serves one
run. Push each results file separately.

One-time setup:

- A Modal token for CI (`uv run modal token new`, or a service token) and a
  proxy auth token (Modal dashboard, Settings, Proxy Auth Tokens).
- A GitHub environment named `production` with the secrets
  `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_PROXY_KEY`, and
  `MODAL_PROXY_SECRET`.

Call the deployed API with the proxy auth headers:

```bash
curl -X POST "$URL/predict" \
    -H "Modal-Key: $MODAL_PROXY_KEY" -H "Modal-Secret: $MODAL_PROXY_SECRET" \
    -H 'content-type: application/json' \
    -d '{"abstracts": ["The team won the final."]}'
```
