# A1: Additional encoders

## Problem Statement
The baseline model workflow must train, evaluate, and serve additional encoders without parallel implementations.

## Solution
Use the existing generic trainer, eval harness, and inference API for ModernBERT, then UltraBERT after its CPU loader is verified.

## User Stories
1. As a model developer, I can train ModernBERT with the existing model argument.
2. As a model developer, I can score the saved checkpoint through the unchanged eval harness.
3. As an API caller, I can receive section predictions from a frozen ModernBERT run.
4. As a model developer, I can repeat the same workflow with a verified UltraBERT checkpoint.

## Implementation Decisions
Keep generic Hugging Face loaders and the existing API contract. Extend the encoder deployment allowlist only as required by verified model configurations. Preserve RoBERTa defaults. Run all smoke training on CPU for exactly five steps. Do not commit or push implementation changes without explicit authorization.

## Testing Decisions
Test the public train-to-freeze-to-predict boundary with a tiny offline encoder. Keep decoder rejection and weight-hash checks. Run full CI before and after implementation, plus a five-step run on the prepared dataset for each real model. Snapshot the staged step 1a diff outside the repository before step 1b.

## Out of Scope
Full training, model quality comparisons, changing evaluation criteria, legacy behavior, or deployment to a paid service.

## Further Notes
The supplied pilot plan is authoritative. Base commit and skill exposure are recorded in the ignored audit folder. The A1 checkout was prepared in an existing conversation, so this is not a fresh blinded subject thread. Skill files were staged during implementation. The user subsequently authorized committing all A1 changes before starting training.
