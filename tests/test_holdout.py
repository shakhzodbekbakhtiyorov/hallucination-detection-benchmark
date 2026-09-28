import json

import pytest

from halubench_eval.config import Config
from halubench_eval.data import Sample, drop_excluded, excluded_keys, stratified_sample
from halubench_eval.metrics import paired_diff
from halubench_eval.report import build_summary, to_markdown
from halubench_eval.runner import save_manifest


def rt(n_pos, n_neg):
    return [Sample(f"r{i}", "RAGTruth", "q", "c", "a", i < n_pos) for i in range(n_pos + n_neg)]


def test_holdout_is_disjoint_from_earlier_run(tmp_path):
    pool = {"RAGTruth": rt(160, 740)}
    main = {"RAGTruth": stratified_sample(pool["RAGTruth"], 200, seed=42)}
    save_manifest(tmp_path, Config(subsets=["RAGTruth"]), main)

    held = drop_excluded(pool, excluded_keys([str(tmp_path)]))["RAGTruth"]
    assert len(held) == 700
    assert not {s.id for s in held} & {s.id for s in main["RAGTruth"]}
    assert sum(s.hallucinated for s in held) == 160 - sum(s.hallucinated for s in main["RAGTruth"])


def test_excluded_keys_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        excluded_keys([str(tmp_path / "nope")])


def test_paired_diff_f1():
    y = [1, 1, 0, 0]
    d = paired_diff(y, [1, 1, 0, 0], [.9, .8, .1, .2], [1, 0, 0, 0], [.9, .1, .1, .2],
                    metric="f1", n_bootstrap=300)
    assert d["diff"] == pytest.approx(1 - 2 / 3)


def test_report_compares_decision_rules_on_same_scores(tmp_path):
    save_manifest(tmp_path, Config(subsets=["RAGTruth"], methods=["llm_judge", "llm_claims"]),
                  {"RAGTruth": [Sample(f"r{i}", "RAGTruth", "q", "c", "a", i < 4) for i in range(10)]})
    (tmp_path / "predictions").mkdir()
    # Hallucinated answers have 1 of 3 claims failing: missed at > 0.5, caught at > 0.
    claims = [{"subset": "RAGTruth", "id": f"r{i}", "label": i < 4, "hallucinated": False,
               "score": 1 / 3 if i < 4 else 0.0, "error": None} for i in range(10)]
    judge = [{"subset": "RAGTruth", "id": f"r{i}", "label": i < 4, "hallucinated": i < 2,
              "score": float(i < 2), "error": None} for i in range(10)]
    judge[9]["error"] = "429"  # excluded for every method
    for name, recs in [("llm_claims", claims), ("llm_judge", judge)]:
        (tmp_path / "predictions" / f"{name}.jsonl").write_text("\n".join(map(json.dumps, recs)))

    r = build_summary(tmp_path, n_bootstrap=200)["rules"]
    assert r["n"] == 9 and list(r["metrics"]) == ["judge", "claims_0.5", "claims_any"]
    assert r["metrics"]["claims_0.5"]["recall"]["value"] == 0.0
    assert r["metrics"]["claims_any"]["recall"]["value"] == 1.0
    assert r["metrics"]["judge"]["recall"]["value"] == 0.5
    same_scores = [p for p in r["paired"] if {p["a"], p["b"]} == {"claims_0.5", "claims_any"}
                   and p["metric"] == "roc_auc"]
    assert same_scores[0]["diff"] == pytest.approx(0.0)
    assert "Decision rules" in to_markdown(build_summary(tmp_path, n_bootstrap=50))
