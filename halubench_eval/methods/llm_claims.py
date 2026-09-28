from ..llm import LLMClient, Usage, content_of, parse_json
from ..prompts import CLAIMS_PROMPT
from .base import Method, Prediction


class LLMClaims(Method):
    name = "llm_claims"

    def __init__(self, llm: LLMClient, threshold: float = 0.5):
        self.llm = llm
        self.threshold = threshold

    def predict(self, sample):
        usage = Usage()
        prompt = CLAIMS_PROMPT.format(question=sample.question, context=sample.context,
                                      answer=sample.answer)
        resp = self.llm.chat([{"role": "user", "content": prompt}], usage)
        raw = content_of(resp)
        try:
            parsed = parse_json(raw)
        except ValueError as e:
            return Prediction(None, None, detail=raw[:500], error=str(e), usage=usage)

        claims = parsed.get("CLAIMS", []) if isinstance(parsed, dict) else []
        details = []
        for c in claims if isinstance(claims, list) else []:
            if not isinstance(c, dict):
                continue
            verdict = str(c.get("verdict", "")).strip().upper()
            if verdict not in ("PASS", "FAIL"):
                continue
            details.append({"claim": str(c.get("claim", "")).strip(), "verdict": verdict})

        if not details:  # no verifiable claims
            return Prediction(False, 0.0, detail={"claims": []}, usage=usage)

        score = sum(d["verdict"] == "FAIL" for d in details) / len(details)  # fraction of FAIL claims
        return Prediction(score > self.threshold, score,
                          detail={"claims": details}, usage=usage)
