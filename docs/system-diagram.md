# System diagram

This page draws every command, artifact, and guardrail in the baseline
branch, from the raw N24News articles to a deployment. The
[README](../README.md) has the four-step overview and the commands. Terms
follow the [glossary](../GLOSSARY.md).

## From raw articles to a deployment

```mermaid
flowchart TB
    subgraph S0["0 · Pre-register, before any test score exists"]
        criteria[/"gate/criteria.toml<br/>eval_version · test_split_sha256<br/>accuracy minimums · logits_atol"/]
    end

    subgraph S1["1 · Data preparation (local)"]
        raw[("datasets/news/<br/>nytimes_{train,dev,test}.json")]
        prep("n24.data.prepare_data<br/>bins/data/data_prep.sh")
        prepared[("datasets/prepared/<br/>DatasetDict: train · dev · test<br/>label = ClassLabel of 24 sections")]
        dataVol[("Modal volume n24-data<br/>/prepared")]
        raw --> prep --> prepared
        prepared -- "modal volume put" --> dataVol
    end

    subgraph S2["2 · Training (Modal GPU: L4, else A10G)"]
        train("bins/train/train_modal.sh &lt;run&gt;<br/>n24.train.modal_train → train_baseline<br/>best epoch chosen on dev macro F1")
        runDir[("Modal volume n24-runs<br/>/news24-runs/&lt;run&gt;/<br/>best_model/ · metrics.json<br/>training_report.pdf")]
        smoke("bins/train/smoke_test.sh<br/>local, --max-steps 5")
        smokeDir[("experiment_runs/smoke-1/")]
        train --> runDir
        smoke --> smokeDir
    end
    dataVol -- "train + dev only" --> train
    prepared -. "train + dev" .-> smoke

    subgraph S3["3 · Tracking (local)"]
        metricsFile[/"metrics/&lt;run&gt;.json"/]
        importRun("n24.track.import_run")
        mlflow[("experiment_runs/mlflow.db<br/>MLflow store")]
        metricsFile --> importRun --> mlflow
    end
    runDir -- "train_modal.sh fetches<br/>metrics.json" --> metricsFile

    subgraph S4["4 · Evaluation (test split only)"]
        localRun[("runs_modal/&lt;run&gt;/")]
        evalRun("n24.eval.run<br/>CPU · deterministic · 2 passes")
        results[/"results/nlp/&lt;run&gt;.json<br/>test metrics · hashes · repeatability<br/>per-article predictions"/]
        zeroShot("n24.eval.zero_shot<br/>vLLM on Modal L4 · 2 passes")
        zsResults[/"results/llm/zs-&lt;model&gt;.json"/]
        localRun --> evalRun --> results
        zeroShot --> zsResults
    end
    runDir -- "modal volume get" --> localRun
    prepared -- "test" --> evalRun
    prepared -- "test" --> zeroShot

    subgraph S5["5 · Commit and push (guardrails)"]
        hooks{{"git hooks: pre-commit · commit-msg · pre-push<br/>identity · privacy · secrets · 20 KB size · perform_ci.sh"}}
        mainBranch[("main<br/>fast-forward push")]
        ci("GitHub Actions ci.yml<br/>bins/ci_tasks/perform_ci.sh")
        hooks --> mainBranch --> ci
    end
    criteria --> hooks
    metricsFile --> hooks
    results --> hooks
    zsResults --> hooks

    subgraph S6["6 · Deployment gate (GitHub Actions deploy.yml → bins/ci_tasks/deploy.sh)"]
        oneFile{"exactly one results/nlp/*.json<br/>added or changed?"}
        gate("n24.gate.report<br/>7 checks")
        verdict{"exit code"}
        gateReport[/"gate/reports/&lt;run&gt;.html<br/>uploaded as the gate-report artifact"/]
        oneFile -- "yes" --> gate --> verdict
        gate --> gateReport
    end
    mainBranch -- "path filter results/nlp/*.json" --> oneFile
    criteria --> gate
    results --> gate

    stopMany["stop<br/>push each results file separately"]
    refused["exit 2 · refused<br/>other eval version or test split"]
    failed["exit 1 · not deployable<br/>at least one check failed"]
    oneFile -- "no" --> stopMany
    verdict -- "2" --> refused
    verdict -- "1" --> failed

    subgraph S7["7 · Deployment (Modal)"]
        deploy("modal deploy -m n24.deploy.modal_app")
        freeze("n24/deploy/freeze.py at image build<br/>BERT-family encoder?<br/>weights sha256 = the gated hash?")
        image[("frozen image<br/>best_model/ · metrics.json · code")]
        api(["n24-api · POST /predict<br/>behind Modal proxy auth"])
        check("bins/api/check_api.sh<br/>200 argmax · 422 bad input · 401 no auth")
        deploy --> freeze --> image --> api --> check
    end
    verdict -- "0 · deployable" --> deploy
    runDir -- "copied once, at build" --> freeze

    classDef tracked fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef ignored fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef modal fill:#ede9fe,stroke:#6d28d9,color:#2e1065
    classDef ciArtifact fill:#ffedd5,stroke:#c2410c,color:#431407
    classDef stop fill:#fee2e2,stroke:#b91c1c,color:#450a0a
    class criteria,metricsFile,results,zsResults,mainBranch tracked
    class raw,prepared,smokeDir,mlflow,localRun ignored
    class dataVol,runDir,image,api modal
    class gateReport ciArtifact
    class stopMany,refused,failed stop
```

| Color | Where the artifact lives |
|---|---|
| Green | Committed to git. Git is the source of truth. |
| Gray | Local and git-ignored (`datasets/`, `experiment_runs/`, `runs_modal/`). Rerun the stage to rebuild it. |
| Purple | On Modal, as a volume, the frozen image, or the live app |
| Orange | A GitHub Actions artifact from the deploy run |
| Red | A stop. Nothing deploys. |

Parallelograms are files, cylinders are stores and directories, rounded boxes
are commands, and diamonds are decisions. The hexagon is the set of git hooks.

## The deployment gate runs seven checks

```mermaid
flowchart LR
    results[/"results/nlp/&lt;run&gt;.json"/] --> match{"eval_version and<br/>test_split_sha256<br/>match criteria.toml?"}
    criteria[/"gate/criteria.toml"/] --> match
    match -- "no" --> refused["exit 2 · refused"]
    match -- "yes" --> checks

    subgraph checks["All seven must pass"]
        direction TB
        subgraph accuracy["Accuracy"]
            f1["test macro F1 ≥ min_test_f1_macro"]
            acc["test accuracy ≥ min_test_accuracy"]
            sec["worst section F1 ≥ min_section_f1"]
        end
        subgraph repeat["Repeatability"]
            passes["two passes: same sections,<br/>logits within logits_atol"]
        end
        subgraph repro["Reproducibility"]
            weights["recomputed weights sha256<br/>= metrics.json"]
            data["recomputed dataset sha256<br/>= metrics.json"]
            commit["git commit recorded<br/>and not -dirty"]
        end
    end

    checks -- "any fails" --> failed["exit 1 · not deployable"]
    checks -- "all pass" --> ok["exit 0 · deployable"]
    checks --> report[/"gate/reports/&lt;run&gt;.html<br/>each row links to its evidence"/]

    classDef tracked fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef stop fill:#fee2e2,stroke:#b91c1c,color:#450a0a
    class results,criteria,report tracked
    class refused,failed stop
```
