import json

import pytest

from halubench_eval.data import Sample
from halubench_eval.llm import LLMClient, LLMError, parse_json
from halubench_eval.methods import ClaimsNLI, GEval, LLMClaims, LLMJudge
from halubench_eval.methods.nli import aggregate_claim, normalise_answer, window_starts
from tests.fakes import FakeOpenAI, response, token

S = Sample(id="1", subset="HaluEval", question="Who wrote X?", context="X was written by A.",
           answer="B wrote X.", hallucinated=True)


def llm(replies):
    return LLMClient(FakeOpenAI(replies), model="m", sleep=lambda s: None)


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_retry_then_success_and_usage():
    class RateLimitError(Exception):
        status_code = 429
    client = llm([RateLimitError("slow down"), response('{"SCORE": "PASS"}')])
    from halubench_eval.llm import Usage
    u = Usage()
    client.chat([{"role": "user", "content": "x"}], u)
    assert u.calls == 1 and u.input_tokens == 100


def test_non_retryable_error_raises():
    class BadRequest(Exception):
        status_code = 400
    with pytest.raises(LLMError):
        from halubench_eval.llm import Usage
        llm([BadRequest("bad")]).chat([], Usage())


def test_llm_judge_uses_verdict_logprobs():
    toks = [token('{"', 1.0), token("SCORE", 1.0), token('":"', 1.0),
            token("FAIL", 0.7, alts=[("PASS", 0.3)]), token('"}', 1.0)]
    resp = response(json.dumps({"REASONING": ["B is not A"], "SCORE": "FAIL"}), toks)
    pred = LLMJudge(llm([resp])).predict(S)
    assert pred.hallucinated is True
    assert pred.score == pytest.approx(0.7)


def test_llm_judge_missing_verdict_is_error_not_faithful():
    pred = LLMJudge(llm([response('{"REASONING": "hmm"}', [])])).safe_predict(S)
    assert pred.error and pred.hallucinated is None


def test_llm_claims_fraction():
    body = {"REASONING": [], "CLAIMS": [{"claim": "B wrote X", "verdict": "FAIL"},
                                        {"claim": "X exists", "verdict": "PASS"},
                                        {"claim": "X is a book", "verdict": "FAIL"}]}
    pred = LLMClaims(llm([response(json.dumps(body))]), threshold=0.5).predict(S)
    assert pred.score == pytest.approx(2 / 3) and pred.hallucinated is True
    assert len(pred.detail["claims"]) == 3


def test_llm_claims_no_claims_is_faithful():
    pred = LLMClaims(llm([response('{"REASONING": [], "CLAIMS": []}')])).predict(S)
    assert pred.hallucinated is False and pred.score == 0.0


def test_geval_expected_score_uses_last_digit():
    toks = [token("Claim", 1.0), token(" 3", 1.0), token(" is wrong.\n", 1.0),
            token("2", 0.5, alts=[("1", 0.5)])]
    pred = GEval(llm([response("reasoning ... 2", toks)])).predict(S)
    # E[score] = 1.5 -> faithfulness 0.125 -> hallucination score 0.875
    assert pred.score == pytest.approx(0.875) and pred.hallucinated is True


def test_geval_truncated_reply_is_error():
    toks = [token("The", 1.0), token(" answer", 1.0)]
    pred = GEval(llm([response("The answer", toks)])).safe_predict(S)
    assert pred.error is not None and pred.hallucinated is None


def test_nli_aggregation_and_decision():
    assert aggregate_claim([{"entailment": .1, "neutral": .8, "contradiction": .1},
                            {"entailment": .9, "neutral": .05, "contradiction": .05}]) == "entailment"
    assert aggregate_claim([{"entailment": .1, "neutral": .2, "contradiction": .7}]) == "contradiction"

    def fake_nli(premises, hyp):
        lab = {"B wrote X.": "contradiction", "X is a text.": "entailment"}[hyp]
        return [{"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0, lab: 1.0} for _ in premises]

    m = ClaimsNLI(llm([response('["B wrote X.", "X is a text."]')]), fake_nli,
                  chunker=lambda c: [c], neutral_threshold=0.6)
    pred = m.predict(S)
    assert pred.hallucinated is True and pred.score == pytest.approx(0.5)


def test_window_starts_cover_whole_context():
    assert window_starts(300, 400, 100) == [0]
    for n in (401, 450, 1000, 1234):
        starts = window_starts(n, 400, 100)
        covered = set()
        for s in starts:
            covered.update(range(s, s + 400))
        assert covered >= set(range(n)) and max(starts) + 400 == n


def test_drop_answers_are_normalised():
    assert normalise_answer("['1936', '1937']", "DROP") == "The answer is: 1936, 1937"
    assert normalise_answer("['x']", "HaluEval") == "['x']"
