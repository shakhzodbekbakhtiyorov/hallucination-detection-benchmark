import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

DATASET_ID = "PatronusAI/HaluBench"


@dataclass(frozen=True)
class Sample:
    id: str
    subset: str
    question: str
    context: str
    answer: str
    hallucinated: bool


def _to_sample(row: dict, subset: str, idx: int) -> Sample:
    return Sample(
        id=str(row.get("id") or f"{subset.lower()}-{idx}"),
        subset=subset,
        question=row.get("question") or "",
        context=row.get("passage") or "",
        answer=str(row.get("answer") or ""),
        hallucinated=str(row.get("label", "")).strip().upper() == "FAIL",
    )


def group_by_subset(rows: Iterable[dict], subsets: List[str]) -> Dict[str, List[Sample]]:
    wanted = {s.lower(): s for s in subsets}
    out = {s: [] for s in subsets}
    for i, row in enumerate(rows):
        name = wanted.get(str(row.get("source_ds", "")).strip().lower())
        if name:
            out[name].append(_to_sample(row, name, i))
    return out


def stratified_sample(samples: List[Sample], n: Optional[int], seed: int,
                      balanced: bool = False) -> List[Sample]:
    # HaluBench rows are ordered by label, so taking the first n rows can give a single class.
    if n is None or n >= len(samples):
        chosen = list(samples)
    else:
        pos = [s for s in samples if s.hallucinated]
        neg = [s for s in samples if not s.hallucinated]
        if balanced:
            n_pos = min(len(pos), n // 2)
            n_neg = min(len(neg), n - n_pos)
        else:
            n_pos = min(len(pos), round(n * len(pos) / len(samples)))
            n_neg = min(len(neg), n - n_pos)
        rng = random.Random(seed)
        chosen = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    random.Random(seed + 1).shuffle(chosen)
    return chosen


def excluded_keys(run_dirs: Iterable[str]) -> Set[str]:
    keys = set()
    for d in run_dirs:
        path = Path(d) / "samples.json"
        if not path.exists():
            raise FileNotFoundError(f"--exclude-from: {path} not found")
        for subset, rows in json.loads(path.read_text()).items():
            keys.update(f"{subset}::{r['id']}" for r in rows)
    return keys


def drop_excluded(grouped: Dict[str, List[Sample]], exclude: Set[str]) -> Dict[str, List[Sample]]:
    return {name: [s for s in rows if f"{s.subset}::{s.id}" not in exclude]
            for name, rows in grouped.items()}


def load_halubench(subsets: List[str], n_per_subset: Optional[int], seed: int,
                   balanced: bool = False, exclude: Optional[Set[str]] = None):
    from datasets import load_dataset

    grouped = group_by_subset(load_dataset(DATASET_ID, split="test"), subsets)
    for name, rows in grouped.items():
        if not rows:
            raise ValueError(f"Subset '{name}' not found in {DATASET_ID}")
    if exclude:
        grouped = drop_excluded(grouped, exclude)
    return {name: stratified_sample(rows, n_per_subset, seed, balanced)
            for name, rows in grouped.items()}
