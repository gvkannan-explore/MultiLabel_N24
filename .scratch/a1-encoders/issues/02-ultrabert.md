# ultrabert

Status: blocked
Blocked by: 01

After 01 passes, verify an UltraBERT model ID and CPU loader; otherwise record a small encoder substitution. Snapshot 01, then repeat its acceptance checks for the third encoder.

## Setup result
`dmis-lab/UltraBERT-150M` resolves as model type `ultrabert` and loaded with `AutoModelForSequenceClassification` on CPU. A two-example forward pass produced finite logits of shape `(2, 24)`. No substitution is needed. Step 1b remains blocked until step 1a passes full CI.
