#!/usr/bin/env python
"""Benchmark hallucination detectors on HaluBench.

  python run_benchmark.py --dry-run                        # show the sample, no API calls
  python run_benchmark.py --out-dir results/main           # 200 per subset, all methods
  python run_benchmark.py --out-dir results/main --report-only   # rebuild the report only

Rerunning the same command resumes an interrupted run.
"""

import argparse
import os
import sys
from pathlib import Path

from halubench_eval.config import ALL_METHODS, ALL_SUBSETS, Config


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", nargs="+", default=d.subsets, choices=ALL_SUBSETS)
    p.add_argument("--methods", nargs="+", default=d.methods, choices=ALL_METHODS)
    p.add_argument("--n", type=int, default=d.n_per_subset,
                   help="samples per subset (0 = full subset)")
    p.add_argument("--balanced", action="store_true", help="draw 50/50 PASS/FAIL per subset")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--exclude-from", nargs="+", default=[], metavar="RUN_DIR",
                   help="skip every sample used by these earlier runs (held-out evaluation)")
    p.add_argument("--claims-threshold", type=float, default=d.claims_threshold,
                   help="llm_claims: hallucinated if fraction of FAIL claims > this (0 = any FAIL claim)")
    p.add_argument("--model", default=d.model)
    p.add_argument("--nli-model", default=d.nli_model)
    p.add_argument("--nli-device", default=d.nli_device)
    p.add_argument("--workers", type=int, default=8, help="parallel API requests per method")
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--bootstrap", type=int, default=d.n_bootstrap)
    p.add_argument("--retry-errors", action="store_true", help="re-run samples that errored")
    p.add_argument("--dry-run", action="store_true", help="show the sample and exit")
    p.add_argument("--report-only", action="store_true", help="rebuild the report and exit")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out_dir)
    from halubench_eval.report import write_report

    if a.report_only:
        write_report(out_dir, n_bootstrap=a.bootstrap)
        print(f"Wrote {out_dir / 'REPORT.md'}")
        return

    cfg = Config(subsets=a.subsets, methods=a.methods, n_per_subset=a.n or None,
                 balanced=a.balanced, seed=a.seed, exclude_from=a.exclude_from,
                 claims_threshold=a.claims_threshold, model=a.model, nli_model=a.nli_model,
                 nli_device=a.nli_device, out_dir=str(out_dir), n_bootstrap=a.bootstrap)

    from halubench_eval.data import excluded_keys, load_halubench
    exclude = excluded_keys(cfg.exclude_from) if cfg.exclude_from else None
    if exclude:
        print(f"Holding out {len(exclude)} samples used by: {', '.join(cfg.exclude_from)}")
    samples = load_halubench(cfg.subsets, cfg.n_per_subset, cfg.seed, cfg.balanced, exclude)
    print(f"{'Subset':<14}{'n':>6}{'halluc.':>9}{'faithful':>10}")
    for name, rows in samples.items():
        pos = sum(s.hallucinated for s in rows)
        print(f"{name:<14}{len(rows):>6}{pos:>9}{len(rows) - pos:>10}")
    if a.dry_run:
        return

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in your environment (see .env.example). Never put it in code.")

    import openai
    from halubench_eval.llm import LLMClient
    from halubench_eval.methods import build_methods
    from halubench_eval.runner import run_method, save_manifest

    save_manifest(out_dir, cfg, samples)
    llm = LLMClient(openai.OpenAI(), model=cfg.model, temperature=cfg.temperature,
                    max_retries=cfg.max_retries, request_delay=cfg.request_delay)
    for name, method in build_methods(cfg, llm).items():
        run_method(method, samples, out_dir, workers=1 if name == "nli" else a.workers,  # local model: sequential
                   retry_errors=a.retry_errors)

    write_report(out_dir, n_bootstrap=cfg.n_bootstrap)
    print(f"\nDone. Report: {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
