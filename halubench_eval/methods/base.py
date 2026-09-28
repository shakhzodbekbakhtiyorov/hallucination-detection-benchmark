from dataclasses import dataclass, field
from typing import Any, Optional

from ..data import Sample
from ..llm import Usage


@dataclass
class Prediction:
    hallucinated: Optional[bool]  # decision; None if the method failed
    score: Optional[float]        # hallucination score in [0, 1], used for ROC-AUC / PR-AUC
    detail: Any = None
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return dict(hallucinated=self.hallucinated, score=self.score, detail=self.detail,
                    error=self.error, usage=self.usage.to_dict())


class Method:
    name = "base"

    def predict(self, sample: Sample) -> Prediction:
        raise NotImplementedError

    def safe_predict(self, sample: Sample) -> Prediction:
        # A failure is recorded as an error and excluded from metrics, never counted as "faithful".
        try:
            return self.predict(sample)
        except Exception as e:
            return Prediction(None, None, error=f"{type(e).__name__}: {e}")
