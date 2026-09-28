import json
import random
import zlib

from halubench_eval.config import Config
from halubench_eval.data import Sample
from halubench_eval.llm import Usage
from halubench_eval.methods.base import Method, Prediction
from halubench_eval.report import write_report
from halubench_eval.runner import run_method, save_manifest


class Noisy(Method):
    def __init__(self, name, flip, fail_ids=()):
        self.name, self.flip, self.fail_ids, self.seen = name, flip, set(fail_ids), 0

    def predict(self, s):
        self.seen += 1
        if s.id in self.fail_ids:
            raise RuntimeError("boom")
        r = random.Random(zlib.crc32(f"{self.name}:{s.id}".encode()))
        wrong = r.random() < self.flip
        pred = s.hallucinated != wrong
        return Prediction(pred, (0.75 if pred else 0.25) + r.uniform(-0.2, 0.2),
                          usage=Usage(calls=1, input_tokens=1000, output_tokens=100, seconds=0.5))


def make_samples():
    out = {}
    for sub, pos in [("HaluEval", 20), ("RAGTruth", 6)]:
        out[sub] = [Sample(f"{sub}-{i}", sub, "q", "c", "a", i < pos) for i in range(40)]
    return out


def test_run_resume_report(tmp_path):
    cfg = Config(subsets=["HaluEval", "RAGTruth"], methods=["llm_judge", "llm_claims"],
                 n_per_subset=40, out_dir=str(tmp_path))
    samples = make_samples()
    save_manifest(tmp_path, cfg, samples)

    judge = Noisy("llm_judge", 0.1)
    claims = Noisy("llm_claims", 0.3, fail_ids={"HaluEval-3"})
    run_method(judge, samples, tmp_path, workers=4)
    run_method(claims, samples, tmp_path, workers=4)
    assert judge.seen == 80

    # resume: nothing re-run
    judge2 = Noisy("llm_judge", 0.1)
    run_method(judge2, samples, tmp_path)
    assert judge2.seen == 0

    # retry errors only
    claims2 = Noisy("llm_claims", 0.3)
    run_method(claims2, samples, tmp_path, retry_errors=True)
    assert claims2.seen == 1

    summary = write_report(tmp_path, n_bootstrap=100)
    assert summary["errors"]["llm_claims"]["failed"] == 0      # the retry succeeded
    assert summary["overall"]["llm_judge"]["n"]["value"] == 80
    assert summary["overall"]["llm_judge"]["accuracy"]["value"] > \
        summary["overall"]["llm_claims"]["accuracy"]["value"]
    assert summary["cost"]["llm_judge"]["usd"] > 0
    report = (tmp_path / "REPORT.md").read_text()
    assert "Lynx (70B)" in report and "Paired accuracy difference" in report
    assert json.loads((tmp_path / "summary.json").read_text())["methods"] == ["llm_judge", "llm_claims"]


def test_errors_are_excluded_not_counted_as_faithful(tmp_path):
    cfg = Config(subsets=["HaluEval"], methods=["llm_judge"], n_per_subset=40, out_dir=str(tmp_path))
    samples = {"HaluEval": make_samples()["HaluEval"]}
    save_manifest(tmp_path, cfg, samples)
    run_method(Noisy("llm_judge", 0.0, fail_ids={"HaluEval-0", "HaluEval-1"}), samples, tmp_path)
    s = write_report(tmp_path, n_bootstrap=50)
    assert s["errors"]["llm_judge"]["failed"] == 2
    assert s["overall"]["llm_judge"]["n"]["value"] == 38
    assert s["overall"]["llm_judge"]["accuracy"]["value"] == 1.0
