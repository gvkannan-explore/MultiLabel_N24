# A1 skill tally

This tally uses the current conversation's execution record and branch artifacts. Reading or copying a skill is not evidence that its full workflow ran. Counts refer to unique skill names, not invocation counts; there is no structured skill-invocation telemetry for this run.

**12 unique skills read: 6 applied, 3 adapted, 2 partially applied, 1 read only. Eight were copied into the A1 arm.**

| Skill | Copied into A1 | Use and evidence |
| --- | --- | --- |
| `tdd` | Yes | Applied. Failing ModernBERT regression, minimal production fixes, passing regression; retained tests. Also used for the run-file contracts. |
| `principle-prove-it-works` | Yes | Applied. Actual five-step CPU run, recorded hashes, evaluation logits, frozen API response; `.audit/modernbert-artifact-check.json`. |
| `code-review` | Yes | Applied. Two read-only review agents covered Standards and Spec. Fixed device reporting and the missing test annotation. |
| `to-spec` | Yes | Adapted. Local spec at `.scratch/a1-encoders/spec.md`; autonomous execution replaced its interactive seam confirmation. |
| `implement-spec` | Yes | Adapted. Two dependent local tickets and one integration branch. Implementation was performed directly; no implementer-worktree or merger-agent workflow ran. |
| `principle-subtract-before-you-add` | Yes | Partial. Used the existing generic loaders and avoided duplicate encoder paths. No distinct subtraction pass was performed. |
| `blast-radius` | Yes | Partial. Checked downstream loader/API use and proved the train-to-freeze-to-predict boundary. No separate complete blast-radius report was produced. |
| `deslop` | Yes | Applied at commit preparation. Reviewed the final diff and expanded collapsed shell commands for readability without changing behavior. |
| `setup-matt-pocock-skills` | No | Adapted. Local markdown tracker/domain docs and CLAUDE instructions were added using the user's preselected local-markdown preference. Full interactive setup did not run. |
| `agent-skills:git-workflow-and-versioning` | No | Applied. Read the workflow and used the isolated branch, bounded staging and diff checks. Committing was authorized explicitly after implementation; no push was requested. |
| `describe-change` | No | Applied at commit preparation. Drafted the commit message from the staged diff and observed verification results. |
| `html-plan` | No | Read only during initial setup planning. No HTML plan was authored; do not count it as applied. |

Thus **9 workflows were applied or adapted**, with **2 additional partial applications**. The copied arm contains eight skills, but that is not eight completed workflows.

## Sources and protocol limits

Skill source commit: `538f0b51e5c23399e944abfaebc36fd4e9bb0bde`. That source checkout was dirty. Per-file hashes, source status, and the background skill directory inventory are recorded in `.audit/a1-exposure.json`; the inventory also contains a `.DS_Store` entry, which is not a skill.

This was an existing conversation with extra global skills available, not a fresh blinded subject thread. The setup probe was not the first message. The copied skills were added to the implementation commit after the run, rather than frozen in `exp/a1-base` before it; work started at the user-approved `15a97d3`, which predates the A0 history rewrite. No clean A0-versus-A1 causal comparison can be claimed from this run.

`advisor`, `grill-with-docs`, and `sharpen` were not used. No advisor was consulted. UltraBERT implementation and training remain pending behind the step-1a CI gate.

Before commit, the generic API route in the copied TDD mocking example was renamed to avoid a privacy-scanner false positive. This is a documented local difference from the source skill, with no instruction change. Original and final skill hashes remain in `.audit/a1-exposure.json`.
