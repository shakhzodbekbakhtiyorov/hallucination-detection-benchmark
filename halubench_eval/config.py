from dataclasses import dataclass, field
from typing import List, Optional

ALL_SUBSETS = ["HaluEval", "RAGTruth", "FinanceBench", "DROP", "CovidQA", "PubMedQA"]
ALL_METHODS = ["llm_judge", "llm_claims", "geval", "nli"]


@dataclass
class Config:
    subsets: List[str] = field(default_factory=lambda: list(ALL_SUBSETS))
    n_per_subset: Optional[int] = 200  # None = full subset
    balanced: bool = False             # False keeps each subset's own label ratio
    seed: int = 42
    exclude_from: List[str] = field(default_factory=list)  # run dirs whose samples are held out

    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_logprobs: int = 5

    # Decision thresholds (score > threshold -> hallucinated), fixed in advance.
    claims_threshold: float = 0.5
    geval_threshold: float = 0.5
    nli_neutral_threshold: float = 0.6

    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    nli_device: str = "cpu"
    nli_chunk_tokens: int = 400
    nli_chunk_stride: int = 100

    max_retries: int = 6
    request_delay: float = 0.0
    price_input_per_m: float = 0.15   # USD per 1M tokens, cost column only
    price_output_per_m: float = 0.60

    out_dir: str = "results/run"
    n_bootstrap: int = 1000
