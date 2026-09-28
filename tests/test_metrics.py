import numpy as np
import pytest

from halubench_eval.metrics import average_precision, evaluate, paired_diff, roc_auc


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)  # many ties
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    expected = ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))
    assert roc_auc(y, s) == pytest.approx(expected)


def test_roc_auc_single_class_is_none():
    assert roc_auc(np.ones(5), np.arange(5.0)) is None


def test_average_precision_perfect_and_known():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0)
    # ranking 1,0,1,0 -> AP = (1/1 + 2/3) / 2
    assert average_precision(np.array([1, 0, 1, 0]), np.array([4, 3, 2, 1.0])) == pytest.approx((1 + 2 / 3) / 2)


def test_point_metrics_and_cis():
    y = [True, True, False, False, True, False]
    p = [True, False, False, True, True, False]
    s = [0.9, 0.4, 0.1, 0.6, 0.8, 0.2]
    m = evaluate(y, p, s, n_bootstrap=200)
    assert m["accuracy"]["value"] == pytest.approx(4 / 6)
    assert m["precision"]["value"] == pytest.approx(2 / 3)
    assert m["recall"]["value"] == pytest.approx(2 / 3)
    assert m["n"]["value"] == 6 and m["n_hallucinated"]["value"] == 3
    assert m["accuracy"]["ci_low"] <= m["accuracy"]["value"] <= m["accuracy"]["ci_high"]


def test_single_class_subset_reports_none_not_zero():
    m = evaluate([False] * 4, [False, True, False, False], [0.1, 0.7, 0.2, 0.1], n_bootstrap=50)
    assert m["f1"]["value"] is None and m["roc_auc"]["value"] is None
    assert m["accuracy"]["value"] == pytest.approx(0.75)


def test_paired_diff():
    y = [1, 0, 1, 0]
    d = paired_diff(y, [1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1],
                    metric="accuracy", n_bootstrap=200)
    assert d["diff"] == pytest.approx(0.5)
