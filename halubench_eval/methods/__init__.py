from ..config import Config
from ..llm import LLMClient
from .base import Method, Prediction
from .geval import GEval
from .llm_claims import LLMClaims
from .llm_judge import LLMJudge
from .nli import ClaimsNLI, DebertaNLI


def build_methods(cfg: Config, llm: LLMClient) -> dict:
    methods = {}
    for name in cfg.methods:
        if name == "llm_judge":
            methods[name] = LLMJudge(llm, top_logprobs=cfg.top_logprobs)
        elif name == "llm_claims":
            methods[name] = LLMClaims(llm, threshold=cfg.claims_threshold)
        elif name == "geval":
            methods[name] = GEval(llm, threshold=cfg.geval_threshold,
                                  top_logprobs=cfg.top_logprobs)
        elif name == "nli":
            nli = DebertaNLI(cfg.nli_model, device=cfg.nli_device,
                             chunk_tokens=cfg.nli_chunk_tokens, stride=cfg.nli_chunk_stride)
            methods[name] = ClaimsNLI(llm, nli, nli.chunks,
                                      neutral_threshold=cfg.nli_neutral_threshold)
        else:
            raise ValueError(f"unknown method: {name}")
    return methods


__all__ = ["Method", "Prediction", "LLMJudge", "LLMClaims", "GEval", "ClaimsNLI",
           "DebertaNLI", "build_methods"]
