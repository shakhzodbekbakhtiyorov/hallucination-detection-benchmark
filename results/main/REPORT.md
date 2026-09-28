# Results

Judge model: `gpt-4o-mini` · samples per subset: 200 (natural label ratio) · seed 42.

Decision rules: LLM-Claims FAIL if fraction of FAIL claims > 0.5; G-Eval FAIL if 1 - normalised score > 0.5; NLI FAIL on any contradiction or neutral fraction > 0.6.

Values in %, 95% bootstrap CIs in brackets; positive class = hallucinated. Only samples that every method scored successfully are included.

## Overall

| Method | Accuracy | Macro acc. | Bal. acc. | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LLM judge | 82.2 [79.9, 84.5] | 82.2 | 82.0 [79.6, 84.2] | 80.0 [77.1, 82.5] | 83.9 [81.4, 86.2] | 78.0 [74.4, 81.3] |
| LLM-Claims | 80.8 [78.5, 83.1] | 80.8 | 80.0 [77.7, 82.3] | 77.1 [74.3, 80.0] | 84.9 [82.6, 86.9] | 77.0 [73.7, 80.5] |
| G-Eval | 79.6 [77.4, 81.9] | 79.6 | 79.2 [76.9, 81.5] | 76.7 [73.9, 79.4] | 84.4 [82.1, 86.6] | 78.0 [74.4, 81.4] |
| Claims + NLI | 59.2 [56.4, 61.7] | 59.2 | 59.9 [57.2, 62.3] | 59.4 [55.9, 62.4] | 61.5 [58.4, 64.1] | 54.0 [50.4, 57.6] |

n = 1198 samples.

## Accuracy per subset

| Subset | LLM judge | LLM-Claims | G-Eval | Claims + NLI |
|---|---:|---:|---:|---:|
| HaluEval (n=199, 99 halluc.) | 92.5 [88.4, 95.5] | 85.9 [80.9, 90.5] | 86.9 [81.9, 91.5] | 70.4 [64.3, 76.4] |
| RAGTruth (n=200, 36 halluc.) | 81.0 [75.5, 86.5] | 82.5 [77.5, 87.5] | 83.0 [78.0, 88.0] | 56.5 [49.0, 63.5] |
| FinanceBench (n=199, 100 halluc.) | 76.4 [70.4, 81.9] | 76.9 [70.9, 82.4] | 72.4 [65.8, 78.4] | 47.7 [40.7, 54.3] |
| DROP (n=200, 100 halluc.) | 76.5 [70.5, 82.5] | 74.0 [67.0, 79.5] | 73.0 [66.5, 78.5] | 58.0 [51.5, 65.0] |
| CovidQA (n=200, 100 halluc.) | 84.5 [79.0, 89.5] | 85.5 [80.5, 90.5] | 83.0 [77.5, 88.5] | 59.0 [52.5, 66.0] |
| PubMedQA (n=200, 100 halluc.) | 82.5 [77.0, 87.5] | 80.0 [74.5, 85.5] | 79.5 [74.0, 84.5] | 63.5 [56.5, 70.0] |

## F1 per subset

| Subset | LLM judge | LLM-Claims | G-Eval | Claims + NLI |
|---|---:|---:|---:|---:|
| HaluEval (n=199, 99 halluc.) | 92.1 [87.6, 95.4] | 84.3 [78.0, 89.7] | 86.0 [80.0, 90.7] | 70.4 [63.0, 76.8] |
| RAGTruth (n=200, 36 halluc.) | 40.6 [24.1, 54.6] | 14.6 [0.0, 30.4] | 32.0 [15.0, 49.1] | 30.4 [18.6, 39.7] |
| FinanceBench (n=199, 100 halluc.) | 78.1 [71.7, 83.6] | 79.3 [72.8, 84.3] | 74.9 [68.0, 81.1] | 50.0 [41.1, 57.5] |
| DROP (n=200, 100 halluc.) | 78.1 [71.9, 83.7] | 73.2 [65.6, 79.6] | 75.7 [68.9, 81.0] | 64.4 [57.1, 71.2] |
| CovidQA (n=200, 100 halluc.) | 82.7 [76.6, 88.4] | 83.8 [77.6, 89.6] | 80.7 [73.9, 86.9] | 55.9 [46.6, 64.0] |
| PubMedQA (n=200, 100 halluc.) | 82.6 [76.5, 87.9] | 78.9 [72.1, 84.8] | 78.8 [71.5, 84.5] | 70.9 [64.1, 76.9] |

## ROC-AUC per subset

| Subset | LLM judge | LLM-Claims | G-Eval | Claims + NLI |
|---|---:|---:|---:|---:|
| HaluEval (n=199, 99 halluc.) | 94.3 [90.1, 97.4] | 90.2 [85.8, 94.1] | 92.1 [87.6, 95.9] | 71.7 [65.3, 78.4] |
| RAGTruth (n=200, 36 halluc.) | 72.1 [61.6, 81.5] | 81.8 [74.0, 90.0] | 77.9 [68.9, 86.4] | 60.9 [49.6, 71.0] |
| FinanceBench (n=199, 100 halluc.) | 78.4 [72.0, 84.1] | 77.1 [71.3, 82.3] | 75.5 [68.5, 81.5] | 47.7 [40.6, 54.6] |
| DROP (n=200, 100 halluc.) | 71.7 [64.5, 78.4] | 77.4 [70.7, 83.3] | 76.6 [69.8, 83.0] | 59.1 [53.0, 65.9] |
| CovidQA (n=200, 100 halluc.) | 90.5 [85.9, 94.8] | 90.3 [86.0, 94.4] | 88.2 [83.1, 92.8] | 59.4 [52.3, 66.9] |
| PubMedQA (n=200, 100 halluc.) | 87.0 [82.2, 91.8] | 84.9 [79.3, 89.7] | 88.0 [83.0, 92.3] | 70.2 [63.3, 76.5] |

## Paired accuracy difference vs. the LLM judge

| Comparison | Δ accuracy | 95% CI |
|---|---:|---:|
| llm_claims - llm_judge | -1.4 | [-3.6, 0.8] |
| geval - llm_judge | -2.6 | [-4.8, -0.4] |
| nli - llm_judge | -23.0 | [-26.3, -19.9] |

A CI that contains 0 means the difference is not distinguishable from noise.

## Decision rules (same scores, different thresholds)

Pooled over this run's samples: n = 1198 (535 hallucinated).

| Rule | Precision | Recall | F1 | Bal. acc. | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| judge | 80.5 [77.1, 83.7] | 79.4 [75.9, 82.8] | 80.0 [77.1, 82.5] | 82.0 [79.6, 84.2] | 83.9 [81.4, 86.2] | 78.0 [74.4, 81.3] |
| claims_0.5 | 82.5 [79.2, 85.9] | 72.3 [68.5, 76.1] | 77.1 [74.3, 80.0] | 80.0 [77.7, 82.3] | 84.9 [82.6, 86.9] | 77.0 [73.7, 80.5] |
| claims_any | 75.2 [71.9, 78.5] | 86.4 [82.9, 89.3] | 80.4 [77.7, 82.8] | 81.7 [79.4, 83.9] | 84.9 [82.6, 86.9] | 77.0 [73.7, 80.5] |
| geval | 78.5 [75.0, 81.8] | 75.0 [71.4, 78.5] | 76.7 [73.9, 79.4] | 79.2 [76.9, 81.5] | 84.4 [82.1, 86.6] | 78.0 [74.4, 81.4] |

| A | B | Metric | Δ | 95% CI |
|---|---:|---:|---:|---:|
| judge | claims_0.5 | f1 | 2.9 | [0.2, 5.6] |
| judge | claims_0.5 | balanced_accuracy | 2.0 | [-0.4, 4.2] |
| judge | claims_0.5 | roc_auc | -0.9 | [-3.3, 1.3] |
| judge | claims_any | f1 | -0.5 | [-2.9, 1.9] |
| judge | claims_any | balanced_accuracy | 0.2 | [-2.1, 2.3] |
| judge | claims_any | roc_auc | -0.9 | [-3.3, 1.3] |
| judge | geval | f1 | 3.3 | [0.8, 5.8] |
| judge | geval | balanced_accuracy | 2.8 | [0.6, 5.0] |
| judge | geval | roc_auc | -0.5 | [-2.8, 1.7] |
| claims_0.5 | claims_any | f1 | -3.3 | [-5.4, -1.1] |
| claims_0.5 | claims_any | balanced_accuracy | -1.7 | [-3.5, 0.2] |
| claims_0.5 | claims_any | roc_auc | 0.0 | [0.0, 0.0] |
| claims_0.5 | geval | f1 | 0.4 | [-2.1, 3.2] |
| claims_0.5 | geval | balanced_accuracy | 0.8 | [-1.3, 3.1] |
| claims_0.5 | geval | roc_auc | 0.4 | [-1.5, 2.4] |
| claims_any | geval | f1 | 3.7 | [1.2, 6.4] |
| claims_any | geval | balanced_accuracy | 2.5 | [0.2, 4.9] |
| claims_any | geval | roc_auc | 0.4 | [-1.5, 2.4] |

## Cost and latency

| Method | API calls | Input tok. | Output tok. | USD | USD / 1k | API s / sample | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM judge | 1200 | 1,353,105 | 147,648 | 0.29 | 0.243 | 1.986 | 0 |
| LLM-Claims | 1200 | 1,421,505 | 253,040 | 0.36 | 0.304 | 2.364 | 1 |
| G-Eval | 1200 | 1,480,305 | 146,631 | 0.31 | 0.258 | 2.217 | 0 |
| Claims + NLI | 1199 | 223,990 | 60,997 | 0.07 | 0.058 | 1.044 | 1 |

At $0.15/M input and $0.6/M output tokens. Local NLI inference time is not included.

## Published HaluBench accuracies (context)

From Ravi et al. (2024) on the full subsets; this run uses a sample, so the comparison is indicative only.

| Model | HaluEval | RAGTruth | FinanceBench | DROP | CovidQA | PubMedQA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LLM judge** (this run) | 92.5 | 81.0 | 76.4 | 76.5 | 84.5 | 82.5 | 82.2 |
| **LLM-Claims** (this run) | 85.9 | 82.5 | 76.9 | 74.0 | 85.5 | 80.0 | 80.8 |
| **G-Eval** (this run) | 86.9 | 83.0 | 72.4 | 73.0 | 83.0 | 79.5 | 79.6 |
| **Claims + NLI** (this run) | 70.4 | 56.5 | 47.7 | 58.0 | 59.0 | 63.5 | 59.2 |
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
