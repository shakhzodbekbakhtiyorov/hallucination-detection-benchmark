# Claim extraction (LLM) + NLI cross-encoder. Long contexts are split into overlapping token
# windows; a claim counts as entailed if any window entails it.

import ast
from typing import Callable, Dict, List, Sequence

from ..llm import LLMClient, Usage, content_of, parse_json
from ..prompts import CLAIM_EXTRACTION_PROMPT
from .base import Method, Prediction

# (premises, hypothesis) -> one {label: probability} dict per premise
NLIScorer = Callable[[Sequence[str], str], List[Dict[str, float]]]


def window_starts(n_tokens: int, size: int, stride: int) -> List[int]:
    if n_tokens <= size:
        return [0]
    step = max(1, size - stride)
    starts = list(range(0, n_tokens - size, step))
    if not starts or starts[-1] != n_tokens - size:
        starts.append(n_tokens - size)
    return starts


class DebertaNLI:
    def __init__(self, model_name: str, device: str = "cpu",
                 chunk_tokens: int = 400, stride: int = 100, batch_size: int = 8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.chunk_tokens, self.stride, self.batch_size = chunk_tokens, stride, batch_size
        self.labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        if set(self.labels.values()) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(f"unexpected NLI labels: {self.labels}")

    def chunks(self, context: str) -> List[str]:
        ids = self.tok(context, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.chunk_tokens:
            return [context]
        return [self.tok.decode(ids[s:s + self.chunk_tokens])
                for s in window_starts(len(ids), self.chunk_tokens, self.stride)]

    def __call__(self, premises: Sequence[str], hypothesis: str) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for i in range(0, len(premises), self.batch_size):
            batch = list(premises[i:i + self.batch_size])
            enc = self.tok(batch, [hypothesis] * len(batch), truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1).cpu().tolist()
            out += [{self.labels[j]: p for j, p in enumerate(row)} for row in probs]
        return out


def aggregate_claim(window_probs: List[Dict[str, float]]) -> str:
    top = [max(p, key=p.get) for p in window_probs]
    if "entailment" in top:
        return "entailment"
    if "contradiction" in top:
        return "contradiction"
    return "neutral"


def normalise_answer(answer: str, subset: str) -> str:
    # DROP answers are stringified Python lists, e.g. "['1936']"
    if subset.lower() == "drop":
        try:
            parsed = ast.literal_eval(answer)
            if isinstance(parsed, (list, tuple)):
                return "The answer is: " + ", ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass
    return answer


class ClaimsNLI(Method):
    name = "nli"

    def __init__(self, llm: LLMClient, nli: NLIScorer, chunker: Callable[[str], List[str]],
                 neutral_threshold: float = 0.6):
        self.llm = llm
        self.nli = nli
        self.chunker = chunker
        self.neutral_threshold = neutral_threshold

    def extract_claims(self, sample, usage: Usage) -> List[str]:
        answer = normalise_answer(sample.answer, sample.subset)
        prompt = CLAIM_EXTRACTION_PROMPT.format(question=sample.question, answer=answer)
        resp = self.llm.chat([{"role": "user", "content": prompt}], usage)
        parsed = parse_json(content_of(resp))
        if not isinstance(parsed, list):
            raise ValueError("claim extractor did not return a JSON list")
        return [c.strip() for c in parsed if isinstance(c, str) and c.strip()]

    def predict(self, sample):
        usage = Usage()
        claims = self.extract_claims(sample, usage)
        if not claims:
            return Prediction(False, 0.0, detail={"claims": []}, usage=usage)

        windows = self.chunker(sample.context)
        labelled = [(c, aggregate_claim(self.nli(windows, c))) for c in claims]
        n = len(labelled)
        n_contra = sum(l == "contradiction" for _, l in labelled)
        n_neutral = sum(l == "neutral" for _, l in labelled)
        score = (n_contra + n_neutral) / n
        hallucinated = n_contra > 0 or (n_neutral / n) > self.neutral_threshold
        return Prediction(hallucinated, score, usage=usage, detail={
            "n_windows": len(windows),
            "claims": [{"claim": c, "label": l} for c, l in labelled],
        })
