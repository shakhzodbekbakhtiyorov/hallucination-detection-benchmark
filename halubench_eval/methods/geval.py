from ..llm import LLMClient, Usage, candidate_probs, content_of, token_logprobs
from ..prompts import GEVAL_PROMPT, GEVAL_SYSTEM
from .base import Method, Prediction

SCORE_TOKENS = ("1", "2", "3", "4", "5")


def expected_score(resp):
    # Probability-weighted score at the last generated digit 1-5; None if the reply has none.
    toks = token_logprobs(resp)
    pos = next((t for t in reversed(toks) if t.token.strip() in SCORE_TOKENS), None)
    if pos is None:
        return None
    probs = {k: 0.0 for k in SCORE_TOKENS}
    for text, prob in candidate_probs(pos):
        k = text.strip()
        if k in probs:
            probs[k] = max(probs[k], prob)
    total = sum(probs.values())
    if total == 0:
        return None
    return sum(int(k) * p for k, p in probs.items()) / total


class GEval(Method):
    name = "geval"

    def __init__(self, llm: LLMClient, threshold: float = 0.5, top_logprobs: int = 5,
                 max_tokens: int = 1024):
        self.llm = llm
        self.threshold = threshold
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens

    def predict(self, sample):
        usage = Usage()
        prompt = GEVAL_PROMPT.format(question=sample.question, context=sample.context,
                                     answer=sample.answer)
        resp = self.llm.chat(
            [{"role": "system", "content": GEVAL_SYSTEM}, {"role": "user", "content": prompt}],
            usage, max_tokens=self.max_tokens, logprobs=True, top_logprobs=self.top_logprobs)
        reasoning = content_of(resp)
        e = expected_score(resp)
        if e is None:
            return Prediction(None, None, detail=reasoning[-500:],
                              error="no 1-5 score token in reply", usage=usage)
        score = 1.0 - min(max((e - 1) / 4, 0.0), 1.0)
        return Prediction(score > self.threshold, score,
                          detail={"expected_score": round(e, 3), "reasoning": reasoning},
                          usage=usage)
