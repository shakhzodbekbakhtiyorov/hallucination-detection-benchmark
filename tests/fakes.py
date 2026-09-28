import math
from types import SimpleNamespace as NS


def token(text, prob, alts=()):
    return NS(token=text, logprob=math.log(prob),
              top_logprobs=[NS(token=t, logprob=math.log(p)) for t, p in alts])


def response(content, tokens=None, prompt_tokens=100, completion_tokens=20):
    choice = NS(message=NS(content=content),
                logprobs=NS(content=tokens) if tokens is not None else None)
    return NS(choices=[choice], usage=NS(prompt_tokens=prompt_tokens,
                                         completion_tokens=completion_tokens))


class FakeOpenAI:
    """Returns queued responses (or calls a function of the messages)."""

    def __init__(self, replies):
        self.replies = list(replies) if not callable(replies) else replies
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.replies):
            return self.replies(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
