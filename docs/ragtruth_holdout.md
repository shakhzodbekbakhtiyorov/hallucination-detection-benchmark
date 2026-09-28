# Pre-registered follow-up: LLM-Claims on held-out RAGTruth

**Written 2026-09-27, before any held-out sample was scored.** Nothing below may be
changed after the held-out run; deviations go in a separate "Deviations" section.

## Motivation (from the main run, `results/main`)

On the 200 RAGTruth samples of the main run (36 hallucinated):

- LLM-Claims ranked hallucinations better than the holistic judge
  (ROC-AUC 81.8 vs 72.1), but had the lowest F1 (14.6 vs 40.6).
- Its decision rule ("hallucinated if **more than 50%** of claims FAIL") let almost
  every real hallucination through: only 3/36 hallucinated answers crossed 50%,
  while 27/36 had **at least one** FAIL claim (vs 25/164 faithful answers).
- RAGTruth hallucinations are naturally occurring and usually affect one claim
  among several correct ones, so "any FAIL claim" is the more natural rule.

Both the ROC-AUC gap and the "any FAIL claim" rule were noticed *on that data*, so
they are hypotheses, not results. This follow-up tests them on data not yet seen.

## Data

- HaluBench, `source_ds == RAGTruth`, **excluding every id in `results/main/samples.json`**
  (`--exclude-from results/main`), all remaining rows (`--n 0`): ~700 samples.
- Label counts are whatever the data contains; they are reported, not chosen.

## Methods (unchanged from the main run)

`llm_judge`, `llm_claims`, `geval`; judge model `gpt-4o-mini`, temperature 0, prompts
as in `halubench_eval/prompts.py` at the time of writing.

Decision rules, fixed now:

| Name | Rule |
|---|---|
| `judge` | the judge's own PASS/FAIL verdict |
| `claims_0.5` | LLM-Claims, hallucinated if fraction of FAIL claims > 0.5 (original rule) |
| `claims_any` | LLM-Claims, hallucinated if fraction of FAIL claims > 0 (**new rule**) |
| `geval` | G-Eval, hallucinated if 1 − normalised score > 0.5 (original rule) |

## Hypotheses (primary)

Each is tested with a paired bootstrap (2,000 resamples, seed 0) over the held-out
samples; "supported" means the 95% CI of the difference excludes 0 in the stated direction.

- **H1 (replication, threshold-free):** ROC-AUC(`llm_claims`) > ROC-AUC(`llm_judge`).
- **H2 (decision rule):** F1(`claims_any`) > F1(`claims_0.5`).
- **H3 (practical):** F1(`claims_any`) > F1(`judge`).

Balanced accuracy and PR-AUC are reported alongside as secondary outcomes.
Everything else (G-Eval comparisons, per-error inspection) is exploratory.

## Procedure

```bash
python run_benchmark.py --subsets RAGTruth --n 0 --exclude-from results/main \
    --methods llm_judge llm_claims geval --claims-threshold 0 \
    --out-dir results/ragtruth_holdout --workers 4
python run_benchmark.py --subsets RAGTruth --n 0 --exclude-from results/main \
    --methods llm_judge llm_claims geval --claims-threshold 0 \
    --out-dir results/ragtruth_holdout --workers 4 --retry-errors   # once, if errors
python scripts/compare_rules.py results/ragtruth_holdout --bootstrap 2000 \
    --rule judge=llm_judge --rule claims_0.5=llm_claims@0.5 \
    --rule claims_any=llm_claims@0 --rule geval=geval
```

Samples that still fail after one retry are excluded from all rules and reported.
Results are reported whatever the outcome, including if no hypothesis is supported.

## Results

Run on 2026-09-27, after this document was written. 700 held-out RAGTruth samples
(124 hallucinated); 0 errors, so no sample was excluded and no retry was needed.
Full output: `results/ragtruth_holdout/REPORT.md` (section "Decision rules").

| Rule | Precision | Recall | F1 | Balanced acc. | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `judge` | 48.4 | 36.3 | 41.5 [33.2, 49.4] | 64.0 [59.7, 68.4] | 73.2 [67.7, 78.4] | 40.5 |
| `claims_0.5` | 56.0 | 11.3 | 18.8 [11.0, 27.3] | 54.7 [52.2, 57.6] | 81.1 [76.8, 85.3] | 46.9 |
| `claims_any` | 50.0 | 76.6 | 60.5 [53.8, 66.7] | 80.1 [75.9, 84.0] | 81.1 [76.8, 85.3] | 46.9 |
| `geval` | 55.8 | 23.4 | 33.0 [24.0, 41.8] | 59.7 [56.1, 63.6] | 76.0 [70.9, 80.5] | 39.2 |

| Hypothesis | Difference [95% CI] | Outcome |
|---|---:|---|
| H1: ROC-AUC(`llm_claims`) − ROC-AUC(`llm_judge`) | +7.9 [2.4, 13.4] | Supported |
| H2: F1(`claims_any`) − F1(`claims_0.5`) | +41.7 [32.0, 51.2] | Supported |
| H3: F1(`claims_any`) − F1(`judge`) | +19.0 [11.2, 27.3] | Supported |

Secondary: balanced accuracy agrees with H3 (+16.1 [11.4, 20.8]). PR-AUC points the same way
(46.9 vs. 40.5) but was not tested as a paired difference.

Exploratory (not pre-registered as hypotheses): G-Eval's F1 did not differ from the judge's
(−8.5 [−16.8, 0.0]); its ROC-AUC was between the two (76.0).

The held-out estimates closely match the development sample from the main run
(`claims_any` F1 61.4 → 60.5; ROC-AUC 81.8 → 81.1 for LLM-Claims, 72.1 → 73.2 for the judge).

**Conclusion.** On naturally occurring RAG hallucinations, evaluating each claim and flagging
an answer when any claim fails roughly doubles recall over a holistic judge (77% vs. 36%) at
the same precision. This is established for RAGTruth with `gpt-4o-mini` only; the rule was not
tested on the synthetic HaluBench subsets or with other judge models.

## Deviations

None in the analysis. Afterwards, the rule comparison was moved from `scripts/compare_rules.py`
into the standard report (`python run_benchmark.py --out-dir results/ragtruth_holdout
--report-only --bootstrap 2000`). The recomputed numbers are identical to the ones above.
