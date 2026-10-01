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

# Real model weights, no Cloudflare account: 9B on a local GPU (T4 pair tested)
python evals/run_eval.py --mode local --model-path Cloudflare/clef-flash

# Hosted endpoint: the published prices and latencies apply
python evals/run_eval.py --mode live

# CI gate: exit code 1 when chunk accuracy drops below the threshold
python evals/run_eval.py --min-accuracy 0.9

# Published Clef vs laya numbers, side by side
python evals/run_eval.py --compare-laya
```

## Measured so far

`results/results.json` is the replay baseline (simulated scorer).
`results/results-local-t4x2.json` is the real open-weights clef-flash 9B
measured on a Kaggle T4 pair via the
[`clef-compactor-evals`](https://www.kaggle.com/code/gjusev/clef-compactor-evals)
kernel: chunk accuracy 0.712, kept F1 0.758, 38.4% of context tokens removed,
scoring latency ~1.27 s p50 (T4, torch attention fallback). The 27B model
needs ~54 GB and does not fit on that hardware.

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
