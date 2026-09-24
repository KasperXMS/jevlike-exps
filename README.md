# Jev-like Decision MVP

Milestone 1 compares three ways to answer one hand-written multiple-choice question
with `Qwen/Qwen3-0.6B-Base`:

- greedy autoregressive generation;
- a restricted softmax over the next-token logits for option labels;
- length-normalized option continuation log likelihood.

The label-logit implementation inspects both bare (`A`) and leading-space (` A`)
forms. It only runs when every label has a distinct, single-token representation
that remains stable at the prompt boundary. Otherwise the JSONL result explicitly
marks the method as unsupported.

## Run

Python 3.10 or newer is required.

From this directory:

```powershell
python -m pip install -r requirements.txt
python scripts/run_milestone1.py
```

The model and device are configured in `configs/qwen3_0.6b.yaml`. CUDA is selected
when available, with a float32 CPU fallback. Raw per-method output is written to
`results/milestone1.jsonl`.

## Verified smoke result

The manual example was run successfully with Qwen3-0.6B-Base using Transformers
4.57.6 and CPU inference. All three methods selected `C. Paris`:

| Method | Prediction | Confidence | Latency |
| --- | --- | ---: | ---: |
| Generation | C | n/a | 977 ms |
| Label logit | C | 0.9774 | 181 ms |
| Option likelihood | C | 0.9987 | 563 ms |

These are smoke-test timings from one cold run, not benchmark measurements.

## ARC-Challenge evaluation

Milestone 2 evaluates a deterministic random sample of 100 examples from the
ARC-Challenge test split:

```powershell
python scripts/run_eval.py --limit 100
```

The command writes raw per-sample results to `results/arc_challenge.jsonl`, the
exact sampled IDs to `results/arc_challenge_sample_ids.json`, and aggregate
accuracy to `results/arc_challenge_summary.json`.

The verified seed-42 test sample produced:

| Method | Accuracy | Parsed/supported | Mean latency |
| --- | ---: | ---: | ---: |
| Generation | 54% | 85/100 | 1060 ms |
| Label logit | 57% | 100/100 | 341 ms |
| Option likelihood | 39% | 100/100 | 1131 ms |

All 15 generation parse failures began generating explanatory prose instead of
an option label and remain in the raw results as incorrect, as intended. These
CPU timings are diagnostic only; the dedicated benchmark belongs to Milestone 6.

MMLU, CommonsenseQA, permutation tests, calibration, and the throughput harness
are intentionally left for the subsequent milestones.
