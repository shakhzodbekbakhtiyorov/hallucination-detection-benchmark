import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, List

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRYABLE_ERRORS = {"RateLimitError", "APIConnectionError", "APITimeoutError"}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(calls=self.calls, input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens, seconds=round(self.seconds, 3))


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    client: Any  # an openai.OpenAI instance, or anything with chat.completions.create
    model: str
    temperature: float = 0.0
    max_retries: int = 6
    request_delay: float = 0.0
    sleep: Any = field(default=time.sleep, repr=False)

    def chat(self, messages: List[dict], usage: Usage, **kwargs):
        last_err = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, **kwargs)
            except Exception as e:
                last_err = e
                retryable = (getattr(e, "status_code", None) in RETRYABLE_STATUS
                             or type(e).__name__ in RETRYABLE_ERRORS)
                if not retryable:
                    raise LLMError(f"{type(e).__name__}: {e}") from e
                self.sleep(min(60, 2 ** attempt) + random.uniform(0, 1))
                continue
            usage.calls += 1
            usage.seconds += time.perf_counter() - start
            if resp.usage is not None:
                usage.input_tokens += resp.usage.prompt_tokens or 0
                usage.output_tokens += resp.usage.completion_tokens or 0
            if self.request_delay:
                self.sleep(self.request_delay)
            return resp
        raise LLMError(f"gave up after {self.max_retries} retries: {last_err}")


def content_of(resp) -> str:
    return (resp.choices[0].message.content or "").strip()


def parse_json(text: str):
    """Parse JSON from a model reply, tolerating code fences and surrounding text."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"unparseable JSON: {text[:200]!r}")


def token_logprobs(resp):
    lp = getattr(resp.choices[0], "logprobs", None)
    return getattr(lp, "content", None) or []


def candidate_probs(tok) -> List[tuple]:
    """(text, probability) for a generated token and its top alternatives."""
    cands = [(tok.token, tok.logprob)] + [(a.token, a.logprob) for a in (tok.top_logprobs or [])]
    return [(t, math.exp(lp)) for t, lp in cands]
