# Runs a method over samples, appending one JSON line per sample so interrupted runs resume.

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import Config
from .data import Sample
from .methods.base import Method


def _key(subset: str, sid: str) -> str:
    return f"{subset}::{sid}"


def load_predictions(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    recs = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {_key(r["subset"], r["id"]): r for r in recs}


def save_manifest(out_dir: Path, cfg: Config, samples: Dict[str, List[Sample]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    manifest = {name: [{"id": s.id, "hallucinated": s.hallucinated} for s in rows]
                for name, rows in samples.items()}
    path = out_dir / "samples.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise SystemExit(
                f"{path} holds a different sample set. Use a new --out-dir or the same "
                "--n/--seed/--balanced/--subsets as the original run.")
    path.write_text(json.dumps(manifest, indent=2))


def run_method(method: Method, samples: Dict[str, List[Sample]], out_dir: Path,
               workers: int = 8, retry_errors: bool = False) -> None:
    path = out_dir / "predictions" / f"{method.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_predictions(path)

    todo = [s for rows in samples.values() for s in rows
            if _key(s.subset, s.id) not in done
            or (retry_errors and done[_key(s.subset, s.id)].get("error"))]
    if not todo:
        print(f"[{method.name}] all {sum(map(len, samples.values()))} samples already done")
        return

    n_err = 0
    with path.open("a") as f, tqdm(total=len(todo), desc=method.name) as bar, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(method.safe_predict, s): s for s in todo}
        for fut in as_completed(futures):
            s, pred = futures[fut], fut.result()
            rec = {"subset": s.subset, "id": s.id, "label": s.hallucinated, **pred.to_dict()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_err += pred.error is not None
            bar.update(1)
            bar.set_postfix(errors=n_err)
    if n_err:
        print(f"[{method.name}] {n_err} errors (excluded from metrics; rerun with --retry-errors)")
