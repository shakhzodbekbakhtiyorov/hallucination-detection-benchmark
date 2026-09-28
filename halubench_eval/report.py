import itertools
import json
from pathlib import Path
from typing import List

import numpy as np

from .config import ALL_METHODS
from .metrics import evaluate, paired_diff

# Accuracy (%) on the full HaluBench subsets from Ravi et al. (2024), arXiv:2407.08488.
PAPER_SUBSETS = ["HaluEval", "RAGTruth", "FinanceBench", "DROP", "CovidQA", "PubMedQA"]
PAPER_RESULTS = {  # per subset in PAPER_SUBSETS order, then the overall mean
    "GPT-4o":               [87.9, 84.3, 85.3, 84.3, 95.0, 82.1, 86.5],
    "GPT-4-Turbo":          [86.0, 85.0, 82.2, 84.8, 90.6, 83.5, 85.0],
    "Claude-3-Sonnet":      [84.5, 79.1, 69.7, 84.3, 95.0, 82.9, 78.8],
    "Llama-3-Instruct-70B": [87.0, 83.8, 72.7, 69.4, 85.0, 82.6, 80.1],
    "Llama-3-Instruct-8B":  [83.1, 80.0, 55.0, 58.2, 75.2, 70.7, 70.4],
    "Mistral-Instruct-7B":  [78.3, 77.7, 56.3, 56.3, 71.7, 77.9, 69.4],
    "RAGAS Faithfulness":   [70.6, 75.8, 59.5, 59.6, 75.0, 67.7, 66.9],
    "GPT-3.5-Turbo":        [62.2, 50.7, 60.9, 57.2, 56.7, 62.8, 58.7],
    "Lynx (8B)":            [85.7, 80.0, 72.5, 77.8, 96.3, 85.2, 82.9],
    "Lynx (70B)":           [88.4, 80.2, 81.4, 86.4, 97.5, 90.4, 87.4],
}

LABELS = {"llm_judge": "LLM judge", "llm_claims": "LLM-Claims", "geval": "G-Eval",
          "nli": "Claims + NLI"}

# Decision rules compared on the same saved scores: name -> (method, threshold or None for
# the method's stored decision). The two LLM-Claims rules differ only in the threshold.
RULES = [("judge", "llm_judge", None), ("claims_0.5", "llm_claims", 0.5),
         ("claims_any", "llm_claims", 0.0), ("geval", "geval", None)]
RULE_METRICS = ["f1", "balanced_accuracy", "roc_auc"]


def _load(out_dir: Path):
    samples = json.loads((out_dir / "samples.json").read_text())
    cfg = json.loads((out_dir / "config.json").read_text())
    preds = {}
    for p in sorted((out_dir / "predictions").glob("*.jsonl")):
        recs = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[f"{r['subset']}::{r['id']}"] = r  # later lines (retries) win
        preds[p.stem] = recs
    return samples, cfg, preds


def _ok(r) -> bool:
    return bool(r) and not r.get("error") and r.get("hallucinated") is not None \
        and r.get("score") is not None


def _cost(recs: dict, keys: List[str], cfg: dict) -> dict:
    u = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    n = 0
    for k in keys:
        if k in recs:
            n += 1
            for f in u:
                u[f] += recs[k].get("usage", {}).get(f, 0) or 0
    usd = (u["input_tokens"] * cfg["price_input_per_m"] + u["output_tokens"] * cfg["price_output_per_m"]) / 1e6
    return {**u, "samples": n, "usd": round(usd, 4),
            "usd_per_1k_samples": round(1000 * usd / n, 3) if n else None,
            "api_seconds_per_sample": round(u["seconds"] / n, 3) if n else None}


def _rules(preds: dict, keys: List[str], y: List[bool], n_bootstrap: int) -> dict:
    rules = [r for r in RULES if r[1] in preds]
    decided = {}
    for name, m, thr in rules:
        scores = [preds[m][k]["score"] for k in keys]
        decisions = ([preds[m][k]["hallucinated"] for k in keys] if thr is None
                     else [s > thr for s in scores])
        decided[name] = (decisions, scores)
    return {
        "n": len(keys), "n_hallucinated": int(sum(y)),
        "rules": {n: {"method": m, "threshold": t} for n, m, t in rules},
        "metrics": {n: evaluate(y, *decided[n], n_bootstrap=n_bootstrap) for n in decided},
        "paired": [{"a": a, "b": b, **paired_diff(y, *decided[a], *decided[b], metric=metric,
                                                  n_bootstrap=n_bootstrap)}
                   for a, b in itertools.combinations(decided, 2) for metric in RULE_METRICS],
    }


def build_summary(out_dir: Path, n_bootstrap: int = 1000) -> dict:
    samples, cfg, preds = _load(Path(out_dir))
    methods = [m for m in ALL_METHODS if m in preds]
    subsets = list(samples)
    keys = {sub: [f"{sub}::{s['id']}" for s in rows] for sub, rows in samples.items()}
    label = {f"{sub}::{s['id']}": s["hallucinated"] for sub, rows in samples.items() for s in rows}
    all_keys = [k for ks in keys.values() for k in ks]
    # Compare methods only on samples that every method scored successfully.
    common = {sub: [k for k in ks if all(_ok(preds[m].get(k)) for m in methods)]
              for sub, ks in keys.items()}
    pooled = [k for sub in subsets for k in common[sub]]
    y = [label[k] for k in pooled]

    def run(m, ks):
        return evaluate([label[k] for k in ks], [preds[m][k]["hallucinated"] for k in ks],
                        [preds[m][k]["score"] for k in ks], n_bootstrap=n_bootstrap)

    summary = {"methods": methods, "subsets": subsets, "config": cfg,
               "errors": {m: {"missing": sum(k not in preds[m] for k in all_keys),
                              "failed": sum(bool(preds[m].get(k, {}).get("error")) for k in all_keys)}
                          for m in methods},
               "cost": {m: _cost(preds[m], all_keys, cfg) for m in methods},
               "per_subset": {sub: {m: run(m, common[sub]) for m in methods} for sub in subsets},
               "overall": {}, "paired": {}}

    for m in methods:
        o = run(m, pooled)
        accs = [summary["per_subset"][s][m]["accuracy"]["value"] for s in subsets]
        accs = [a for a in accs if a is not None]
        o["macro_accuracy"] = {"value": float(np.mean(accs)) if accs else None,
                               "ci_low": None, "ci_high": None}
        summary["overall"][m] = o

    if "llm_judge" in methods:
        judge = preds["llm_judge"]
        for m in methods[1:]:
            summary["paired"][f"{m} - llm_judge"] = paired_diff(
                y, [preds[m][k]["hallucinated"] for k in pooled], [preds[m][k]["score"] for k in pooled],
                [judge[k]["hallucinated"] for k in pooled], [judge[k]["score"] for k in pooled],
                metric="accuracy", n_bootstrap=n_bootstrap)
    if "llm_claims" in methods:
        summary["rules"] = _rules(preds, pooled, y, n_bootstrap)
    return summary


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _cell(m: dict, ci: bool = True) -> str:
    if m.get("value") is None:
        return "–"
    if ci and m.get("ci_low") is not None:
        return f"{_pct(m['value'])} [{_pct(m['ci_low'])}, {_pct(m['ci_high'])}]"
    return _pct(m["value"])


def _diff(d: dict) -> List[str]:
    return [_pct(d["diff"]), f"[{_pct(d['ci_low'])}, {_pct(d['ci_high'])}]"]


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def to_markdown(s: dict) -> str:
    methods, subsets, cfg = s["methods"], s["subsets"], s["config"]
    held_out = f" Held out from: {', '.join(cfg['exclude_from'])}." if cfg.get("exclude_from") else ""
    out = ["# Results", "",
           f"Judge model: `{cfg['model']}` · samples per subset: {cfg['n_per_subset'] or 'all'} "
           f"({'balanced' if cfg['balanced'] else 'natural label ratio'}) · seed {cfg['seed']}.{held_out}",
           "",
           f"Decision rules: LLM-Claims FAIL if fraction of FAIL claims > {cfg['claims_threshold']}; "
           f"G-Eval FAIL if 1 - normalised score > {cfg['geval_threshold']}; "
           f"NLI FAIL on any contradiction or neutral fraction > {cfg['nli_neutral_threshold']}.",
           "", "Values in %, 95% bootstrap CIs in brackets; positive class = hallucinated. "
           "Only samples that every method scored successfully are included.", ""]

    n = s["overall"][methods[0]]["n"]["value"] if methods else 0
    out += ["## Overall", "",
            _table(["Method", "Accuracy", "Macro acc.", "Bal. acc.", "F1", "ROC-AUC", "PR-AUC"],
                   [[LABELS[m], _cell(o["accuracy"]), _cell(o["macro_accuracy"], False),
                     _cell(o["balanced_accuracy"]), _cell(o["f1"]), _cell(o["roc_auc"]),
                     _cell(o["pr_auc"])] for m, o in s["overall"].items()]),
            "", f"n = {n} samples.", ""]

    for metric, title in [("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]:
        rows = []
        for sub in subsets:
            ps = s["per_subset"][sub]
            first = ps[methods[0]]
            rows.append([f"{sub} (n={first['n']['value']}, {first['n_hallucinated']['value']} halluc.)"]
                        + [_cell(ps[m][metric]) for m in methods])
        out += [f"## {title} per subset", "", _table(["Subset"] + [LABELS[m] for m in methods], rows), ""]

    if s["paired"]:
        out += ["## Paired accuracy difference vs. the LLM judge", "",
                _table(["Comparison", "Δ accuracy", "95% CI"], [[k] + _diff(v) for k, v in s["paired"].items()]),
                "", "A CI that contains 0 means the difference is not distinguishable from noise.", ""]

    if "rules" in s:
        r = s["rules"]
        out += ["## Decision rules (same scores, different thresholds)", "",
                f"Pooled over this run's samples: n = {r['n']} ({r['n_hallucinated']} hallucinated).", "",
                _table(["Rule", "Precision", "Recall", "F1", "Bal. acc.", "ROC-AUC", "PR-AUC"],
                       [[name] + [_cell(m[k]) for k in ["precision", "recall", "f1", "balanced_accuracy",
                                                         "roc_auc", "pr_auc"]]
                        for name, m in r["metrics"].items()]),
                "", _table(["A", "B", "Metric", "Δ", "95% CI"],
                           [[p["a"], p["b"], p["metric"]] + _diff(p) for p in r["paired"]]), ""]

    out += ["## Cost and latency", "",
            _table(["Method", "API calls", "Input tok.", "Output tok.", "USD", "USD / 1k", "API s / sample",
                    "Errors"],
                   [[LABELS[m], str(c["calls"]), f"{c['input_tokens']:,}", f"{c['output_tokens']:,}",
                     f"{c['usd']:.2f}", str(c["usd_per_1k_samples"]), str(c["api_seconds_per_sample"]),
                     str(s["errors"][m]["failed"])] for m, c in s["cost"].items()]),
            "", f"At ${cfg['price_input_per_m']}/M input and ${cfg['price_output_per_m']}/M output tokens. "
            "Local NLI inference time is not included.", ""]

    rows = [[f"**{LABELS[m]}** (this run)"]
            + [_pct(s["per_subset"][sub][m]["accuracy"]["value"]) if sub in subsets else "–"
               for sub in PAPER_SUBSETS] + [_pct(s["overall"][m]["macro_accuracy"]["value"])]
            for m in methods]
    rows += [[name] + [f"{v:.1f}" for v in vals] for name, vals in PAPER_RESULTS.items()]
    out += ["## Published HaluBench accuracies (context)", "",
            "From Ravi et al. (2024) on the full subsets; this run uses a sample, so the comparison "
            "is indicative only.", "", _table(["Model"] + PAPER_SUBSETS + ["Overall"], rows), ""]
    return "\n".join(out)


def write_report(out_dir, n_bootstrap: int = 1000) -> dict:
    out_dir = Path(out_dir)
    summary = build_summary(out_dir, n_bootstrap=n_bootstrap)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "REPORT.md").write_text(to_markdown(summary))
    return summary
