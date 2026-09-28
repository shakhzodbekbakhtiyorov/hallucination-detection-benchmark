#!/usr/bin/env python
"""Build the report's figures and results tables from the saved run summaries.

    python report/make_figures.py    # from the repository root
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MAIN = json.loads((ROOT / "results/main/summary.json").read_text())
RULES = json.loads((ROOT / "results/ragtruth_holdout/summary.json").read_text())["rules"]

# Colour-blind-safe categorical palette, fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, QUIET, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
METHODS = ["llm_judge", "llm_claims", "geval", "nli"]
NAMES = {"llm_judge": "LLM judge", "llm_claims": "LLM-Claims", "geval": "G-Eval",
         "nli": "Claims + NLI"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": QUIET, "axes.labelcolor": INK, "xtick.color": QUIET,
    "ytick.color": QUIET, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def pct(x):
    return None if x is None else 100 * x


def fmt(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def ci(m):
    return f"{fmt(pct(m['value']))} [{fmt(pct(m['ci_low']))}, {fmt(pct(m['ci_high']))}]"


def signed(x):
    if abs(x) < 0.05:
        return "0.0"
    return f"{'+' if x > 0 else '$-$'}{abs(x):.1f}"


def tex(text):
    return text.replace("%", "\\%").replace(">", "$>$")


# ---------------------------------------------------------------- figure 1
def fig_roc_by_subset():
    subsets = MAIN["subsets"]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    width = 0.19
    for k, m in enumerate(METHODS):
        xs = [i + (k - 1.5) * width for i in range(len(subsets))]
        cells = [MAIN["per_subset"][s][m]["roc_auc"] for s in subsets]
        vals = [pct(c["value"]) for c in cells]
        err = [[v - pct(c["ci_low"]) for v, c in zip(vals, cells)],
               [pct(c["ci_high"]) - v for v, c in zip(vals, cells)]]
        ax.bar(xs, vals, width * 0.9, color=C[k], label=NAMES[m], zorder=2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7,
                    capsize=1.5, zorder=3)
    ax.axhline(50, color=QUIET, lw=0.8, ls=(0, (4, 3)), zorder=1, label="Chance (50)")
    ax.set_xticks(range(len(subsets)), subsets)
    ax.set_ylim(0, 100)
    ax.set_ylabel("ROC-AUC (%)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8.5,
              handlelength=1.4, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/roc_auc_by_subset.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
RULE_LABELS = {"judge": "LLM judge\n(own verdict)",
               "claims_0.5": "LLM-Claims\n(>50% of claims fail)",
               "claims_any": "LLM-Claims\n(any claim fails)",
               "geval": "G-Eval\n(score > 0.5)"}


def fig_holdout_rules():
    rules = list(RULES["metrics"])
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    width = 0.25
    for k, (key, name) in enumerate(metrics):
        xs = [i + (k - 1) * width for i in range(len(rules))]
        cells = [RULES["metrics"][r][key] for r in rules]
        vals = [pct(c["value"]) for c in cells]
        ax.bar(xs, vals, width * 0.9, color=C[k], label=name, zorder=2)
        if key == "f1":
            err = [[v - pct(c["ci_low"]) for v, c in zip(vals, cells)],
                   [pct(c["ci_high"]) - v for v, c in zip(vals, cells)]]
            ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7,
                        capsize=1.5, zorder=3)
            tops = [pct(c["ci_high"]) for c in cells]
        else:
            tops = vals
        for x, v, t in zip(xs, vals, tops):
            ax.text(x, t + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(rules)), [RULE_LABELS[r] for r in rules], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hallucinated answers (%)")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/holdout_rules.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def paired(a, b, metric):
    for p in RULES["paired"]:
        if p["metric"] == metric and {p["a"], p["b"]} == {a, b}:
            sign = 1 if p["a"] == a else -1
            lo, hi = sign * p["ci_low"], sign * p["ci_high"]
            return 100 * sign * p["diff"], 100 * min(lo, hi), 100 * max(lo, hi)
    raise KeyError((a, b, metric))


FOREST = [
    ("H1", "ROC-AUC: LLM-Claims vs. judge", "claims_any", "judge", "roc_auc", True),
    ("H2", "F1: any-claim rule vs. >50% rule", "claims_any", "claims_0.5", "f1", True),
    ("H3", "F1: LLM-Claims (any) vs. judge", "claims_any", "judge", "f1", True),
    ("Expl.", "F1: G-Eval vs. judge", "geval", "judge", "f1", False),
]


def fig_forest():
    fig, ax = plt.subplots(figsize=(6.5, 2.1))
    for i, (tag, label, a, b, metric, pre) in enumerate(FOREST):
        d, lo, hi = paired(a, b, metric)
        y = len(FOREST) - 1 - i
        colour = C[0] if pre else "#a8a79f"
        ax.plot([lo, hi], [y, y], color=colour, lw=2.2, solid_capstyle="round")
        ax.plot(d, y, "o", color=colour, ms=6.5, mec="white", mew=1)
        ax.text(-0.02, y, f"{tag}  {label}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(hi + 1.5, y, f"{signed(d)} [{signed(lo)}, {signed(hi)}]", va="center",
                fontsize=8, color=QUIET,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.axvline(0, color=QUIET, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks([])
    ax.set_xlim(-22, 92)
    ax.set_ylim(-0.6, len(FOREST) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired difference (percentage points), 95% bootstrap CI")
    fig.tight_layout()
    fig.subplots_adjust(left=0.38)
    fig.savefig(OUT / "figures/holdout_forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def table_overall():
    o, cost = MAIN["overall"], MAIN["cost"]
    rows = []
    for m in METHODS:
        diff = MAIN["paired"].get(f"{m} - llm_judge")
        d = "--" if diff is None else (f"{signed(100 * diff['diff'])} "
                                       f"[{signed(100 * diff['ci_low'])}, {signed(100 * diff['ci_high'])}]")
        usd = f"{cost[m]['usd_per_1k_samples']:.2f}" + ("\\textsuperscript{a}" if m == "nli" else "")
        rows.append(f"{NAMES[m]} & {ci(o[m]['accuracy'])} & {fmt(pct(o[m]['balanced_accuracy']['value']))} & "
                    f"{fmt(pct(o[m]['f1']['value']))} & {ci(o[m]['roc_auc'])} & "
                    f"{fmt(pct(o[m]['pr_auc']['value']))} & {d} & {usd} \\\\")
    (OUT / "tables/overall.tex").write_text("\n".join(rows) + "\n")


def table_holdout():
    rows = []
    for r, m in RULES["metrics"].items():
        label = tex(RULE_LABELS[r].replace("\n", " "))
        rows.append(f"{label} & {fmt(pct(m['precision']['value']))} & {fmt(pct(m['recall']['value']))} & "
                    f"{ci(m['f1'])} & {ci(m['balanced_accuracy'])} & {ci(m['roc_auc'])} & "
                    f"{fmt(pct(m['pr_auc']['value']))} \\\\")
    (OUT / "tables/holdout.tex").write_text("\n".join(rows) + "\n")


def table_hypotheses():
    rows = []
    for tag, label, a, b, metric, pre in FOREST:
        d, lo, hi = paired(a, b, metric)
        verdict = ("Supported" if lo > 0 else "Not supported") if pre else "Exploratory"
        rows.append(f"{tag} & {tex(label)} & {signed(d)} & [{signed(lo)}, {signed(hi)}] & {verdict} \\\\")
    (OUT / "tables/hypotheses.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    fig_roc_by_subset()
    fig_holdout_rules()
    fig_forest()
    table_overall()
    table_holdout()
    table_hypotheses()
    print("figures and tables written to", OUT)
