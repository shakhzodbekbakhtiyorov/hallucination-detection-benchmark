# Hallucination Detection Benchmark

How well can an LLM-based evaluator tell whether a RAG answer is **faithful** to
its retrieved context? This repo benchmarks four faithfulness/hallucination
detectors on [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench)
(Ravi et al., 2024): ~15k question/context/answer triples from six sources,
each labelled PASS (faithful) or FAIL (hallucinated).

Each method outputs both a **decision** (PASS/FAIL) and a **continuous
hallucination score** in [0, 1], so they are compared on thresholded metrics
(accuracy, F1) and threshold-free ranking metrics (ROC-AUC, PR-AUC), with
bootstrap confidence intervals, cost, and latency.

## Methods

| Method | How it works | Continuous score | API calls / sample |
|---|---|---|---|
| `llm_judge` | One holistic PASS/FAIL verdict with the HaluBench/Lynx evaluation prompt. | P(FAIL) from the verdict token's logprobs | 1 |
| `llm_claims` | Same criteria, but the judge splits the answer into atomic claims and gives a verdict per claim, in one call. | Fraction of claims judged FAIL | 1 |
| `geval` | G-Eval (Liu et al., 2023): chain-of-thought rubric, 1–5 score. | 1 − normalised E[score] from the score token's logprobs | 1 |
| `nli` | LLM extracts atomic claims; a local NLI cross-encoder (`cross-encoder/nli-deberta-v3-large`) checks each claim against the context. | Fraction of claims not entailed | 1 + local model |

The same judge model (default `gpt-4o-mini`) is used for every LLM step.

## Design choices

- **Stratified, seeded sampling.** HaluBench rows are grouped by label, so taking
  the first *N* rows of a subset can return a single class (e.g. only
  hallucinated answers), which makes precision, recall and F1 meaningless. Samples are
  drawn per label with a fixed seed. By default the subset's own label ratio is kept;
  `--balanced` draws 50/50. The exact sample ids are saved to `samples.json`.
- **Fair comparison.** Metrics are computed only on samples that *every* method
  scored successfully, and the paired accuracy difference to the holistic judge is
  reported with a paired-bootstrap CI.
- **Errors are not "faithful".** A failed API call, unparseable reply or missing
  score is recorded as an error, excluded from metrics, and counted in the report.
  (`--retry-errors` re-runs only those.)
- **Long contexts in NLI.** NLI cross-encoders see ~512 tokens. The context is split
  into overlapping windows, each paired with the claim as a proper (premise,
  hypothesis) pair, and a claim counts as entailed if any window entails it. Naively
  concatenating and truncating would cut off the claim itself on long passages.
- **Thresholds are fixed in advance** (see `halubench_eval/config.py`), not tuned
  on the evaluation data. ROC-AUC/PR-AUC show performance independent of the cutoff.
- **Resumable.** Predictions are appended to `predictions/<method>.jsonl` after every
  sample; re-running the same command skips finished samples.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # put your key in .env (never commit it)
export $(grep -v '^#' .env | xargs)

python run_benchmark.py --dry-run                     # show the sample, no API calls
python run_benchmark.py --n 20 --out-dir results/smoke # small smoke run
python run_benchmark.py --out-dir results/main         # 200 per subset, all methods
```

The report is written to `<out-dir>/REPORT.md` (tables) and `<out-dir>/summary.json`
(all numbers). Rebuild it without API calls: `python run_benchmark.py --out-dir results/main --report-only`.

Useful options: `--methods llm_judge llm_claims`, `--subsets RAGTruth DROP`,
`--model gpt-4o`, `--nli-device mps` (Apple silicon) or `cuda`, `--workers 8`, `--n 0` (full subsets).

## Results

All runs: judge model `gpt-4o-mini`, temperature 0. Full tables with per-subset numbers:
[`results/main/REPORT.md`](results/main/REPORT.md) and
[`results/ragtruth_holdout/REPORT.md`](results/ragtruth_holdout/REPORT.md). Full write-up:
[`report/report.pdf`](report/report.pdf).

### 1. Main benchmark: 1,198 samples across all six subsets

200 per subset, stratified by label, seed 42. Two samples where a method errored were excluded
from every method.

| Method | Accuracy | Balanced acc. | F1 | ROC-AUC | PR-AUC | USD / 1k samples |
|---|---:|---:|---:|---:|---:|---:|
| LLM judge (holistic) | **82.2** [79.9, 84.5] | **82.0** | **80.0** | 83.9 [81.4, 86.2] | 78.0 | 0.24 |
| LLM-Claims (per-claim judge) | 80.8 [78.5, 83.1] | 80.0 | 77.1 | **84.9** [82.6, 86.9] | 77.0 | 0.30 |
| G-Eval (logprob-weighted) | 79.6 [77.4, 81.9] | 79.2 | 76.7 | 84.4 [82.1, 86.6] | 78.0 | 0.26 |
| Claims + NLI (DeBERTa-v3-large) | 59.2 [56.4, 61.7] | 59.9 | 59.4 | 61.5 [58.4, 64.1] | 54.0 | 0.06 + local model |

Values in %, 95% bootstrap CIs in brackets.

- **The three LLM-based detectors are statistically tied.** Their ROC-AUCs (83.9–84.9) have
  overlapping CIs. Paired accuracy differences against the holistic judge: LLM-Claims −1.4
  [−3.6, 0.8] (not distinguishable), G-Eval −2.6 [−4.8, −0.4] (slightly worse, weak evidence).
  With `gpt-4o-mini`, neither claim decomposition nor logprob weighting beats a single holistic
  verdict on HaluBench as a whole.
- **For context:** the holistic `gpt-4o-mini` judge's macro accuracy (82.2) is in the range that
  Ravi et al. (2024) report for Lynx-8B (82.9) and Llama-3-70B (80.1), below GPT-4o (86.5). Their
  numbers are on the full benchmark, so this is indicative, not a head-to-head comparison.
- **Claim extraction + off-the-shelf NLI is far worse** (−23.0 accuracy points vs. the judge,
  CI [−26.3, −19.9]; ROC-AUC at chance on FinanceBench). It flags 47% of faithful answers as
  hallucinated, mostly because correct but paraphrased or multi-sentence claims come back as
  "neutral" rather than "entailed" (195 of 312 false alarms); spurious contradictions account
  for the rest.
- **RAGTruth needs different metrics.** It is the only subset with naturally occurring
  hallucinations (18% of answers), so "always faithful" already scores 82% accuracy. On
  RAGTruth, LLM-Claims ranked hallucinations best (ROC-AUC 81.8 vs. 72.1 for the judge) but its
  "> 50% of claims fail" rule caught only 3 of 36 hallucinations. Because this was noticed on
  the evaluation data, it was tested separately.

### 2. Pre-registered held-out test: 700 unseen RAGTruth samples

Hypotheses and decision rules were written down before scoring
([`docs/ragtruth_holdout.md`](docs/ragtruth_holdout.md)); the samples exclude every row used above.
124 of 700 answers are hallucinated.

| Rule | Precision | Recall | F1 | Balanced acc. | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LLM judge (own verdict) | 48.4 | 36.3 | 41.5 [33.2, 49.4] | 64.0 | 73.2 [67.7, 78.4] |
| LLM-Claims, FAIL if > 50% of claims fail | 56.0 | 11.3 | 18.8 [11.0, 27.3] | 54.7 | 81.1 [76.8, 85.3] |
| **LLM-Claims, FAIL if any claim fails** | 50.0 | **76.6** | **60.5** [53.8, 66.7] | **80.1** | **81.1** [76.8, 85.3] |
| G-Eval | 55.8 | 23.4 | 33.0 [24.0, 41.8] | 59.7 | 76.0 [70.9, 80.5] |

All three hypotheses were supported (paired bootstrap, 2,000 resamples):

| Hypothesis | Difference [95% CI] |
|---|---:|
| H1: LLM-Claims ranks hallucinations better than the judge (ROC-AUC) | +7.9 [2.4, 13.4] |
| H2: "any claim fails" beats "> 50% fail" (F1) | +41.7 [32.0, 51.2] |
| H3: LLM-Claims ("any") beats the judge (F1) | +19.0 [11.2, 27.3] |

**Takeaway:** on RAGTruth's naturally occurring hallucinations, a claim-level judge with an
"any failing claim" rule caught 77% of hallucinated answers versus 36% for the holistic
judge's verdict, at similar precision (50% vs. 48%), for about 1.9× the cost. Even with its
threshold tuned on the test data (a best case), the holistic judge reached at most 47% recall at
50% precision, so the advantage is not an artifact of the judge's default cut-off. A likely
reason: natural hallucinations are mostly one or a few wrong claims in an otherwise correct
answer, which a single overall verdict often lets through. The held-out numbers closely match
the development sample (ROC-AUC 81.8 → 81.1, F1 61.4 → 60.5).

**Scope:** this is one dataset of natural hallucinations (124 positive examples held out), one
judge model (`gpt-4o-mini`) and one prompt per detector. The "any failing claim" rule was not
tested on the five synthetic subsets, where it may raise false alarms. At 50% precision it suits
a screening step followed by review, not automatic rejection.

## Reproducing the held-out test

```bash
python run_benchmark.py --subsets RAGTruth --n 0 --exclude-from results/main \
    --methods llm_judge llm_claims geval --claims-threshold 0 --out-dir results/ragtruth_holdout
python run_benchmark.py --out-dir results/ragtruth_holdout --report-only --bootstrap 2000
```

`--exclude-from` holds out every sample an earlier run used. Whenever LLM-Claims is included,
the report also compares both of its decision rules (> 50% and any failing claim) on the same
scores, with paired-bootstrap CIs. The pre-registered plan is in
[`docs/ragtruth_holdout.md`](docs/ragtruth_holdout.md).

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests run offline with a fake LLM client (no API key or model download needed).
They cover sampling, metrics (ROC-AUC with ties, PR-AUC, bootstrap CIs, single-class
subsets), response parsing, each method's scoring logic, and resume/retry behaviour.

## Project layout

```
run_benchmark.py            CLI entry point
halubench_eval/
  config.py                 all settings and thresholds
  data.py                   HaluBench loading + stratified sampling
  prompts.py                judge, claims, claim-extraction and G-Eval prompts
  llm.py                    OpenAI wrapper: retries, token usage, latency
  methods/                  llm_judge, llm_claims, geval, nli
  metrics.py                metrics + bootstrap CIs
  runner.py                 resumable execution
  report.py                 summary.json + REPORT.md, incl. decision-rule comparison
docs/                       pre-registered follow-up experiment
report/                     LaTeX write-up; make_figures.py builds its figures from results/
tests/                      offline unit and end-to-end tests
```

## Limitations

- A few hundred samples per subset give CIs of several points; use `--n 0` for full subsets.
- The published baselines were measured on the full benchmark, with other models; they
  are context, not a head-to-head comparison.
- One judge model (`gpt-4o-mini`) was used throughout; a stronger judge may narrow the
  holistic-vs-claims gap.
- LLM judges are sensitive to prompt wording and model version; results are for the
  prompts in `prompts.py` and the model named in the report.
- HaluBench's hallucinated answers are partly synthetic perturbations, which may be
  easier to detect than naturally occurring RAG hallucinations.

## References

- Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024).
  *Lynx: An Open Source Hallucination Evaluation Model.* arXiv:2407.08488.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023).
  *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP.
- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual
  Precision in Long Form Text Generation.* EMNLP.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). *RAGAS: Automated
  Evaluation of Retrieval Augmented Generation.* EACL (demo).
