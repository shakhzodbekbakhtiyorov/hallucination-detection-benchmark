from ..llm import LLMClient, Usage, candidate_probs, content_of, parse_json, token_logprobs
from ..prompts import JUDGE_PROMPT
from .base import Method, Prediction


def _norm(tok: str) -> str:
    return tok.strip().strip('"').strip().upper()


def fail_probability(resp):
    # P(FAIL) normalised against P(PASS), at the first generated PASS/FAIL token.
    for tok in token_logprobs(resp):
        if _norm(tok.token) in ("PASS", "FAIL"):
            p = {"PASS": 0.0, "FAIL": 0.0}
            for text, prob in candidate_probs(tok):
                k = _norm(text)
                if k in p:
                    p[k] = max(p[k], prob)
            total = p["PASS"] + p["FAIL"]
            return p["FAIL"] / total if total > 0 else None
    return None


class LLMJudge(Method):
    name = "llm_judge"

    def __init__(self, llm: LLMClient, top_logprobs: int = 5):
        self.llm = llm
        self.top_logprobs = top_logprobs

    def predict(self, sample):
        usage = Usage()
        prompt = JUDGE_PROMPT.format(question=sample.question, context=sample.context,
                                     answer=sample.answer)
        resp = self.llm.chat([{"role": "user", "content": prompt}], usage,
                             logprobs=True, top_logprobs=self.top_logprobs)
        raw = content_of(resp)
        try:
            parsed = parse_json(raw)
            verdict = str(parsed.get("SCORE", "")).strip().upper()
            reasoning = parsed.get("REASONING", "")
        except ValueError:
            verdict, reasoning = "", raw[:500]
        if verdict not in ("PASS", "FAIL"):
            return Prediction(None, None, detail=raw[:500],
                              error="no PASS/FAIL verdict", usage=usage)
        hallucinated = verdict == "FAIL"
        p_fail = fail_probability(resp)
        score = p_fail if p_fail is not None else float(hallucinated)  # no logprobs -> 0/1
        return Prediction(hallucinated, score, detail=reasoning, usage=usage)
