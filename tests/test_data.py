from halubench_eval.data import group_by_subset, stratified_sample


def rows(n_fail, n_pass, subset="DROP"):
    # ordered like HaluBench: all FAIL first, then all PASS
    return ([{"id": f"f{i}", "source_ds": subset, "label": "FAIL", "passage": "c",
              "question": "q", "answer": "a"} for i in range(n_fail)] +
            [{"id": f"p{i}", "source_ds": subset, "label": "PASS", "passage": "c",
              "question": "q", "answer": "a"} for i in range(n_pass)])


def test_first_n_would_be_single_class_but_stratified_is_not():
    samples = group_by_subset(rows(500, 500), ["DROP"])["DROP"]
    assert all(s.hallucinated for s in samples[:100])      # the old bug
    chosen = stratified_sample(samples, 100, seed=42)
    assert len(chosen) == 100
    assert sum(s.hallucinated for s in chosen) == 50


def test_keeps_natural_ratio_by_default():
    samples = group_by_subset(rows(160, 740, "RAGTruth"), ["RAGTruth"])["RAGTruth"]
    chosen = stratified_sample(samples, 200, seed=1)
    assert len(chosen) == 200
    assert sum(s.hallucinated for s in chosen) == round(200 * 160 / 900)


def test_balanced_option():
    samples = group_by_subset(rows(160, 740, "RAGTruth"), ["RAGTruth"])["RAGTruth"]
    chosen = stratified_sample(samples, 200, seed=1, balanced=True)
    assert sum(s.hallucinated for s in chosen) == 100


def test_seeded_and_unique():
    samples = group_by_subset(rows(300, 300), ["DROP"])["DROP"]
    a = [s.id for s in stratified_sample(samples, 50, seed=7)]
    b = [s.id for s in stratified_sample(samples, 50, seed=7)]
    assert a == b and len(set(a)) == 50


def test_subset_matching_is_case_insensitive():
    g = group_by_subset(rows(1, 1, "covidQA"), ["CovidQA"])
    assert len(g["CovidQA"]) == 2
