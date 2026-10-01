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

## Milestone 2.5

Milestone 2.5 reuses the exact 100 IDs in
`results/arc_challenge_sample_ids.json`. It tightens generation parsing, records
candidate-label vocabulary mass, and compares legacy and exact-text likelihood
prompts under both summed and mean token log likelihood:

```powershell
python scripts/run_milestone25.py
```

The verified CPU run produced:

| Metric | Result |
| --- | ---: |
| Generation end-to-end accuracy | 53% |
| Generation parse rate | 84% |
| Generation accuracy when parsed | 63.10% |
| Label-logit accuracy | 57% |
| Mean candidate-label mass | 25.27% |
| Legacy likelihood, mean | 39% |
| Legacy likelihood, summed | 42% |
| Exact-text likelihood, mean | 38% |
| Exact-text likelihood, summed | 33% |

Generation versus label-logit paired counts were `G+L+ = 50`, `G+L- = 3`,
`G-L+ = 7`, and `G-L- = 40`. Raw results are written to
`results/arc_challenge_milestone25.jsonl`; the distinct summary and paired sample
IDs are stored in `results/arc_challenge_milestone25_summary.json` and
`results/arc_challenge_milestone25_paired.json`. The original Milestone 2 files
are not overwritten.

## Milestone 3: cross-dataset validation

Milestone 3 evaluates the frozen ARC split, five 100-example MMLU subjects, and
100 frozen CommonsenseQA validation examples:

```powershell
python scripts/run_milestone3.py --config configs/qwen3_0.6b.yaml
```

The requested broad MMLU names map explicitly to standard configs:
`computer_science -> college_computer_science`, `physics -> college_physics`,
`history -> high_school_world_history`, and
`psychology -> high_school_psychology`.

| Dataset | Generation | Gen. parsed | Gen. conditional | Label logit | Legacy mean/sum | Exact mean/sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 53.0% | 84.0% | 63.1% | 57.0% | 39.0% / 42.0% | 38.0% / 33.0% |
| MMLU | 35.0% | 69.4% | 50.4% | 42.6% | 36.6% / 37.8% | 35.0% / 33.0% |
| CommonsenseQA | 46.0% | 93.0% | 49.5% | 54.0% | 35.0% / 42.0% | 25.0% / 34.0% |

Label-logit leads generation end to end on all three aggregate datasets, so the
ARC result generalizes beyond one benchmark. It does not win every MMLU subject:
computer science is 26% versus generation's 27%, and psychology ties at 66%.
Generation conditional accuracy is consistently above end-to-end accuracy,
confirming that answer-format failures remain material.

`G+L-` remains uncommon: 3/100 on ARC, 18/500 on MMLU, and 3/100 on
CommonsenseQA. `G-L+` occurs on every dataset (7, 56, and 11 respectively), which
is evidence that direct label decisions recover useful cases rather than merely
matching parsed generation.

Mean candidate-label mass is 25.3% on ARC, 33.3% on MMLU, and 22.7% on
CommonsenseQA. The scale is similar across datasets, but it is not stable per
sample: MMLU spans 4.6% at p10 to 77.2% at p90. Restricted probabilities must
therefore remain interpreted as forced-choice probabilities, not calibrated
full-vocabulary confidence.

Option likelihood remains a generally weaker baseline and is sensitive to prompt
and aggregation choice. Overall, Qwen3-0.6B shows a useful cross-task label-logit
decision interface, but variable candidate mass and uneven subject performance
are not yet evidence for a robust or calibrated general-purpose decision head.

Milestone 3 outputs `results/milestone3_results.jsonl`,
`results/milestone3_summary.json`, `results/milestone3_paired.json`, and
`results/milestone3_candidate_mass.json`. MMLU and CommonsenseQA frozen IDs are
stored separately and reused by subsequent runs.

## Milestone 4: permutation robustness and calibration

Milestone 4 reuses all 700 Milestone 3 samples. It evaluates label-logit on the
original option order plus three deterministic permutations, then calibrates
only the 700 original-order results with a frozen, subject-stratified 20/80
calibration/test split:

```powershell
python scripts/run_milestone4_permutation.py --config configs/qwen3_0.6b.yaml
python scripts/run_milestone4_calibration.py
```

Permutation aggregate accuracy did not decline, but semantic stability was
limited:

| Dataset | Original accuracy | Non-original accuracy | Semantic consistency |
| --- | ---: | ---: | ---: |
| ARC-Challenge | 57.0% | 57.3% | 61.7% |
| MMLU | 42.6% | 46.1% | 48.3% |
| CommonsenseQA | 54.0% | 55.7% | 59.0% |

The flat aggregate accuracy therefore does not mean that predictions reliably
follow option semantics. Position effects are substantial: MMLU predictions
select B 45.4% of the time versus a 24.6% gold frequency, while CommonsenseQA
selects D 36.5% of the time versus an 18.5% gold frequency. Label-conditioned
accuracy also varies sharply. Qwen3-0.6B's label-logit interface is useful, but
is not permutation invariant and has clear label/position bias.

On the held-out calibration splits, scalar temperature scaling preserved every
argmax as required, but did not improve calibration consistently:

| Dataset | Temperature | Raw NLL / Brier / ECE | Scaled NLL / Brier / ECE |
| --- | ---: | ---: | ---: |
| ARC-Challenge | 0.944 | 0.960 / 0.514 / 0.120 | 0.959 / 0.513 / 0.121 |
| MMLU | 1.271 | 1.168 / 0.633 / 0.092 | 1.169 / 0.632 / 0.081 |
| CommonsenseQA | 1.429 | 1.049 / 0.537 / 0.084 | 1.118 / 0.564 / 0.157 |
| Pooled | 1.238 | 1.121 / 0.602 / 0.079 | 1.129 / 0.604 / 0.076 |

The restricted probabilities carry useful ranking information, especially on
MMLU and CommonsenseQA, but are not reliably calibrated by a small scalar-fit
split. Restricted-confidence AUROC for correctness is 0.727 on ARC, 0.747 on
MMLU, and 0.803 on CommonsenseQA.

Candidate mass is stable under permutation at the output-mode level: mean
within-sample ranges are 0.053 on ARC, 0.045 on MMLU, and 0.057 on
CommonsenseQA. Its usefulness for correctness is dataset dependent. Candidate-
mass AUROC is 0.525 on ARC, 0.629 on MMLU, and 0.572 on CommonsenseQA; MMLU
fixed-bin accuracy rises from roughly 34% below mass 0.25 to 67% above 0.75,
but ARC is not monotonic. Candidate mass can be an auxiliary readiness signal,
not a standalone confidence measure.

These results support moving to a model-scaling milestone as a controlled test
of whether size reduces semantic and position instability. They do not support
calling the 0.6B interface robust or calibrated already. Full manifests, raw
records, summaries, frozen calibration IDs, temperatures, candidate-mass
analysis, and reliability diagrams are under `results/` without overwriting any
earlier milestone output.

## Milestone 5: invariance scaling

### Research question

Milestone 5 tests whether the label/position sensitivity found in Milestone 4
is mainly a small-model capability limit or a structural failure of label-token
scoring. It compares three public, non-gated checkpoints from the same base
model family: Qwen3-0.6B-Base, Qwen3-1.7B-Base, and Qwen3-4B-Base. The loaded
models contain 0.600B (nominal), 1.721B, and 4.022B parameters respectively.

### Setup

The experiment reuses the exact 700 Milestone 3 sample IDs, all 2,800 entries
in the Milestone 4 permutation manifest, the frozen calibration/test split,
prompts, parser, and label-token inspection. New models run generation and
label-logit on every original sample and label-logit on all orderings. The 0.6B
records are derived without modification from the M3/M4 raw files.

```powershell
python scripts/run_milestone5.py --config configs/qwen3_0.6b.yaml
```

Original-order results show clear scaling of the label-logit interface. Gen.
conditional accuracy is reported only over successfully parsed generations.

| Model | Dataset | Gen. E2E | Parse | Gen. conditional | Label logit |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B | ARC | 53.0% | 84.0% | 63.1% | 57.0% |
| 0.6B | MMLU | 35.0% | 69.4% | 50.4% | 42.6% |
| 0.6B | CSQA | 46.0% | 93.0% | 49.5% | 54.0% |
| 1.7B | ARC | 66.0% | 87.0% | 75.9% | 77.0% |
| 1.7B | MMLU | 36.4% | 61.2% | 59.5% | 56.6% |
| 1.7B | CSQA | 57.0% | 76.0% | 75.0% | 76.0% |
| 4B | ARC | 84.0% | 97.0% | 86.6% | 86.0% |
| 4B | MMLU | 40.0% | 58.4% | 68.5% | 67.0% |
| 4B | CSQA | 72.0% | 100.0% | 72.0% | 73.0% |

### Invariance results

Scaling improves every primary robustness signal. The aggregate row is
sample-weighted, so the 500 MMLU examples retain their intended weight.

| Model | Semantic consistency | C-to-W | W-to-C | Mean gold-prob. range |
| --- | ---: | ---: | ---: | ---: |
| 0.6B | 51.8% | 30.9% | 31.7% | 0.243 |
| 1.7B | 67.2% | 19.4% | 27.0% | 0.197 |
| 4B | 77.0% | 11.7% | 31.2% | 0.184 |

At 4B, semantic consistency is 85.7% on ARC, 75.3% on MMLU, and
77.0% on CommonsenseQA. Correct-to-wrong flips fall from 24.6%/33.8%/25.9%
at 0.6B to 6.6%/12.8%/12.3% at 4B. Wrong-to-correct flips remain substantial,
so unchanged aggregate accuracy would still be an inadequate robustness
measure. The separately reported directions expose that cancellation.

Position preference also weakens. The 0.6B model predicts B on 45.4% of MMLU
orderings; the 4B distribution is A/B/C/D = 28.9%/27.7%/26.9%/16.5%.
CommonsenseQA's 0.6B D preference (36.5%) becomes a much flatter
24.0%/23.3%/21.2%/17.2%/14.2% distribution at 4B. Residual D/E under-selection
means the interface is improved, not perfectly permutation invariant.

### Calibration observation

No new calibration method is fitted. On the unchanged held-out split, pooled
raw ECE is 0.079, 0.046, and 0.067 from 0.6B through 4B. Restricted-confidence
AUROC rises from 0.763 to 0.797 to 0.850. Candidate-mass AUROC is non-monotonic
(0.604, 0.717, 0.642), reinforcing that candidate mass is an exploratory
readiness signal rather than calibrated confidence or uncertainty.

### Interpretation and limitations

The evidence meets Route A: semantic consistency reaches 77% overall at 4B,
correct-to-wrong flips fall by more than half, gold-probability instability
declines, and position preference becomes materially flatter. In this tested
range, most label-token instability is capability-limited rather than an
unchanging structural ceiling. It is not fully eliminated, especially on MMLU
and five-option CommonsenseQA.

This conclusion is limited to one model family, three sizes, one prompt, three
deterministic non-original permutations, and these frozen datasets. MMLU
dominates the aggregate by design. Calibration estimates use only 560 held-out
examples, and these evaluation timings are not performance benchmarks.

The decision gate therefore selects **Route A / Milestone 6A**: benchmark
generation versus label-logit on CUDA with synchronized timing, warm-up,
multiple batch sizes, throughput, and peak-memory measurement. All M5 raw
records, summaries, paired groups, invariance metrics, calibration observations,
and seven scaling figures are in `results/milestone5/`.

## Milestone 6A: CUDA efficiency

### Research question and setup

Milestone 6A asks whether the now-useful zero-decoding interface produces a
real systems benefit. It benchmarks generation and label-logit on an RTX 4060
Ti 16 GB using the same 0.6B, 1.7B, and 4B models. A deterministic set of 64
four-option prompts is drawn from the frozen M3 pool (58 MMLU and 6 ARC); each
batch size uses the same nested prefix for every model and method.

Generation performs one prefill plus fixed-step greedy decoding to exactly
eight new tokens, matching the evaluation token cap but deliberately disabling
early EOS termination. Label-logit performs one no-cache forward and a
restricted softmax. Every successful combination uses three warm-ups and 20
timed trials. CUDA events are synchronized, and mean/P50/P95, throughput, and
peak CUDA allocation are saved per combination.

```powershell
python scripts/run_milestone6a.py --config configs/qwen3_0.6b.yaml
python scripts/validate_milestone6a.py
```

### Results

Latency speedup is generation total latency divided by label-logit total
latency; the throughput ratio is numerically identical because both methods use
the same batch size.

| Model | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B | 7.46x | 2.34x | 1.48x | 1.42x |
| 1.7B | 7.50x | 1.65x | 1.27x | 1.23x |
| 4B | 7.43x | 1.42x | 1.66x | memory-limited / label only |

At batch 1, label-logit raises throughput from 2.32 to 17.30 samples/s for
0.6B, 2.33 to 17.50 for 1.7B, and 1.87 to 13.88 for 4B. At batch 32 the
generation/label throughputs are 14.30/21.17, 8.77/11.14, and 2.93/4.86
samples/s. Restricted scoring itself costs below 0.2 ms in every measured
combination; almost all label latency is the shared model forward.

The memory benefit grows with batch size. At batch 64, generation versus label
peak allocation is 7.35/2.38 GiB for 0.6B and 10.61/5.58 GiB for 1.7B. For 4B
at batch 32 it is 12.14/9.16 GiB. The 4B batch-64 generation warm-up saturated
16077/16380 MiB physical VRAM under Windows WDDM and entered sustained shared-
memory paging, so it is recorded as `memory_limit`, not mislabeled as a CUDA
OOM. Batch-64 label-logit completed at 11.04 GiB and 4.47 samples/s.

### Interpretation and limitations

Zero decoding provides a large low-batch latency gain and a smaller but still
real throughput gain after prefill dominates at larger batches. It also avoids
the KV-cache growth that made the largest 4B generation workload unusable on
this GPU. The result supports a decision interface for workloads that need a
small fixed answer rather than generated prose; it does not imply a universal
7x speedup.

The benchmark uses one consumer GPU under WDDM, one nested prompt sample, no
CUDA graphs, no specialized serving engine, and a fixed eight-token decode.
Real generation that stops before eight tokens would reduce the advantage;
longer answers would increase it. The four-option timing set excludes
CommonsenseQA because keeping candidate tensor shape fixed avoids changing the
scoring workload by method. Reported numbers are batch latency, not per-sample
latency, and are not mixed with earlier CPU smoke timings.

These results justify entering **Milestone 7**. The next research question is
whether a shared encoded state can answer multiple independent decision
questions without repeating context prefill. Raw trials, the explicit
memory-limit record, environment metadata, summary, validator, and separate
speedup/throughput/memory plots are in `results/milestone6a/`.
