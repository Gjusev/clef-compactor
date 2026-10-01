# evals

Reproducible evaluation harness for clef-compactor: compaction quality,
latency and cost on a committed dataset with gold relevance labels.

## Dataset

`data/compaction_suite.jsonl` holds 20 cases (104 chunks) across support,
technical, finance, legal and science topics. Each row carries the query, the
retrieved chunks and a boolean `relevant` label per chunk. Labels were written
by hand against the query; they are the ground truth for every metric.

## Run it

```bash
# Pipeline validation: real compaction code, deterministic simulated scorer
python evals/run_eval.py

# Real quality numbers: needs CLEF_ACCOUNT_ID and CLEF_API_TOKEN
python evals/run_eval.py --mode live

# CI gate: exit code 1 when chunk accuracy drops below the threshold
python evals/run_eval.py --min-accuracy 0.9

# Published Clef vs laya numbers, side by side
python evals/run_eval.py --compare-laya
```

Every run writes `results/results.json`: aggregate metrics plus one entry per
case. The committed file was produced by `--mode replay`, so it is
reproducible byte for byte given the same seed.

## What the modes mean

`replay` runs the real compaction pipeline (batching, ranking, budget) against
a seeded simulated scorer whose noisy probabilities overlap the decision
threshold. It validates the machinery and gives stable numbers for CI, but it
says nothing about Clef's actual accuracy. Treat it as a regression gate.

`live` calls Cloudflare's hosted Clef endpoint. That is the run that produces
quality and latency numbers worth quoting. It needs credentials and network
access, so CI keeps it behind the `integration` marker and a manual workflow.

Latency in replay mode measures client-side overhead only. Quoting latency
requires a live run.

## Reproducing on Kaggle

For heavier reproductions (or the open-weights Clef models from
https://huggingface.co/Cloudflare/clef), the pattern from
https://github.com/Gjusev/laya-evals works unchanged: push a kernel folder
with `kernel-metadata.json` plus a script that clones this repo, installs it
and runs `evals/run_eval.py --mode live` with credentials injected as Kaggle
secrets. Outputs land in `/kaggle/working/` and come back with
`kaggle kernels output`. Keep API tokens out of the kernel source.
