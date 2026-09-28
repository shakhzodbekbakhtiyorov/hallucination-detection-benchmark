# Results

Judge model: `gpt-4o-mini` · samples per subset: all (natural label ratio) · seed 42. Held out from: results/main.

Decision rules: LLM-Claims FAIL if fraction of FAIL claims > 0.0; G-Eval FAIL if 1 - normalised score > 0.5; NLI FAIL on any contradiction or neutral fraction > 0.6.

Values in %, 95% bootstrap CIs in brackets; positive class = hallucinated. Only samples that every method scored successfully are included.

## Overall

| Method | Accuracy | Macro acc. | Bal. acc. | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LLM judge | 81.9 [78.9, 84.7] | 81.9 | 64.0 [59.7, 68.4] | 41.5 [33.2, 49.4] | 73.2 [67.7, 78.4] | 40.5 [32.4, 49.4] |
| LLM-Claims | 82.3 [79.3, 85.1] | 82.3 | 80.1 [75.9, 84.0] | 60.5 [53.8, 66.7] | 81.1 [76.8, 85.3] | 46.9 [38.7, 56.7] |
| G-Eval | 83.1 [80.4, 86.0] | 83.1 | 59.7 [56.1, 63.6] | 33.0 [24.0, 41.8] | 76.0 [70.9, 80.5] | 39.2 [31.8, 48.1] |

n = 700 samples.

## Accuracy per subset

| Subset | LLM judge | LLM-Claims | G-Eval |
|---|---:|---:|---:|
| RAGTruth (n=700, 124 halluc.) | 81.9 [78.9, 84.7] | 82.3 [79.3, 85.1] | 83.1 [80.4, 86.0] |

## F1 per subset

| Subset | LLM judge | LLM-Claims | G-Eval |
|---|---:|---:|---:|
| RAGTruth (n=700, 124 halluc.) | 41.5 [33.2, 49.4] | 60.5 [53.8, 66.7] | 33.0 [24.0, 41.8] |

## ROC-AUC per subset

| Subset | LLM judge | LLM-Claims | G-Eval |
|---|---:|---:|---:|
| RAGTruth (n=700, 124 halluc.) | 73.2 [67.7, 78.4] | 81.1 [76.8, 85.3] | 76.0 [70.9, 80.5] |

## Paired accuracy difference vs. the LLM judge

| Comparison | Δ accuracy | 95% CI |
|---|---:|---:|
| llm_claims - llm_judge | 0.4 | [-2.9, 3.7] |
| geval - llm_judge | 1.3 | [-0.9, 3.4] |

A CI that contains 0 means the difference is not distinguishable from noise.

## Decision rules (same scores, different thresholds)

Pooled over this run's samples: n = 700 (124 hallucinated).

| Rule | Precision | Recall | F1 | Bal. acc. | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| judge | 48.4 [38.4, 58.7] | 36.3 [27.6, 44.7] | 41.5 [33.2, 49.4] | 64.0 [59.7, 68.4] | 73.2 [67.7, 78.4] | 40.5 [32.4, 49.4] |
| claims_0.5 | 56.0 [36.4, 75.0] | 11.3 [6.3, 17.2] | 18.8 [11.0, 27.3] | 54.7 [52.2, 57.6] | 81.1 [76.8, 85.3] | 46.9 [38.7, 56.7] |
| claims_any | 50.0 [42.9, 57.2] | 76.6 [68.9, 84.2] | 60.5 [53.8, 66.7] | 80.1 [75.9, 84.0] | 81.1 [76.8, 85.3] | 46.9 [38.7, 56.7] |
| geval | 55.8 [41.8, 69.2] | 23.4 [16.4, 31.0] | 33.0 [24.0, 41.8] | 59.7 [56.1, 63.6] | 76.0 [70.9, 80.5] | 39.2 [31.8, 48.1] |

| A | B | Metric | Δ | 95% CI |
|---|---:|---:|---:|---:|
| judge | claims_0.5 | f1 | 22.7 | [12.0, 32.7] |
| judge | claims_0.5 | balanced_accuracy | 9.3 | [4.8, 14.1] |
| judge | claims_0.5 | roc_auc | -7.9 | [-13.4, -2.4] |
| judge | claims_any | f1 | -19.0 | [-27.3, -11.2] |
| judge | claims_any | balanced_accuracy | -16.1 | [-20.8, -11.4] |
| judge | claims_any | roc_auc | -7.9 | [-13.4, -2.4] |
| judge | geval | f1 | 8.5 | [-0.0, 16.8] |
| judge | geval | balanced_accuracy | 4.3 | [0.2, 8.4] |
| judge | geval | roc_auc | -2.8 | [-7.7, 2.7] |
| claims_0.5 | claims_any | f1 | -41.7 | [-51.2, -32.0] |
| claims_0.5 | claims_any | balanced_accuracy | -25.4 | [-30.0, -21.0] |
| claims_0.5 | claims_any | roc_auc | 0.0 | [0.0, 0.0] |
| claims_0.5 | geval | f1 | -14.2 | [-23.7, -4.2] |
| claims_0.5 | geval | balanced_accuracy | -5.0 | [-8.9, -1.2] |
| claims_0.5 | geval | roc_auc | 5.1 | [0.8, 9.6] |
| claims_any | geval | f1 | 27.6 | [18.3, 36.7] |
| claims_any | geval | balanced_accuracy | 20.4 | [16.1, 25.0] |
| claims_any | geval | roc_auc | 5.1 | [0.8, 9.6] |

## Cost and latency

| Method | API calls | Input tok. | Output tok. | USD | USD / 1k | API s / sample | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM judge | 700 | 426,169 | 91,063 | 0.12 | 0.169 | 1.925 | 0 |
| LLM-Claims | 700 | 466,069 | 250,449 | 0.22 | 0.315 | 3.241 | 0 |
| G-Eval | 700 | 500,369 | 77,447 | 0.12 | 0.174 | 1.946 | 0 |

At $0.15/M input and $0.6/M output tokens. Local NLI inference time is not included.

## Published HaluBench accuracies (context)

From Ravi et al. (2024) on the full subsets; this run uses a sample, so the comparison is indicative only.

| Model | HaluEval | RAGTruth | FinanceBench | DROP | CovidQA | PubMedQA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LLM judge** (this run) | – | 81.9 | – | – | – | – | 81.9 |
| **LLM-Claims** (this run) | – | 82.3 | – | – | – | – | 82.3 |
| **G-Eval** (this run) | – | 83.1 | – | – | – | – | 83.1 |
| GPT-4o | 87.9 | 84.3 | 85.3 | 84.3 | 95.0 | 82.1 | 86.5 |
| GPT-4-Turbo | 86.0 | 85.0 | 82.2 | 84.8 | 90.6 | 83.5 | 85.0 |
| Claude-3-Sonnet | 84.5 | 79.1 | 69.7 | 84.3 | 95.0 | 82.9 | 78.8 |
| Llama-3-Instruct-70B | 87.0 | 83.8 | 72.7 | 69.4 | 85.0 | 82.6 | 80.1 |
| Llama-3-Instruct-8B | 83.1 | 80.0 | 55.0 | 58.2 | 75.2 | 70.7 | 70.4 |
| Mistral-Instruct-7B | 78.3 | 77.7 | 56.3 | 56.3 | 71.7 | 77.9 | 69.4 |
| RAGAS Faithfulness | 70.6 | 75.8 | 59.5 | 59.6 | 75.0 | 67.7 | 66.9 |
| GPT-3.5-Turbo | 62.2 | 50.7 | 60.9 | 57.2 | 56.7 | 62.8 | 58.7 |
| Lynx (8B) | 85.7 | 80.0 | 72.5 | 77.8 | 96.3 | 85.2 | 82.9 |
| Lynx (70B) | 88.4 | 80.2 | 81.4 | 86.4 | 97.5 | 90.4 | 87.4 |
