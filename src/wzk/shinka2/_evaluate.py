from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ._config import WindowConfig

__all__ = [
    "aggregate_scores",
    "code_hash",
    "jitter_windows",
    "load_evolved_module",
    "run_eval_cli",
    "validate_score",
    "write_metrics",
]


def code_hash(module_path: str) -> int:
    """Derive a deterministic seed from the program source code SHA256."""
    code = Path(module_path).read_text()
    return int(hashlib.sha256(code.encode()).hexdigest(), 16) % (2**31)


def jitter_windows(
    windows: list[WindowConfig],
    rng: np.random.Generator,
    jitter_days: int = 30,
) -> list[WindowConfig]:
    """Shift each window's start/end by a random offset in [-jitter_days, +jitter_days]."""
    import pandas as pd

    jittered = []
    for w in windows:
        s = pd.Timestamp(w.start, tz="UTC") + pd.Timedelta(days=int(rng.integers(-jitter_days, jitter_days + 1)))
        e = pd.Timestamp(w.end, tz="UTC") + pd.Timedelta(days=int(rng.integers(-jitter_days, jitter_days + 1)))
        jittered.append(WindowConfig(label=w.label, start=str(s), end=str(e)))
    return jittered


def write_metrics(
    results_dir: str,
    combined_score: float,
    *,
    error: str | None = None,
    public_metrics: dict[str, Any] | None = None,
    private_metrics: dict[str, Any] | None = None,
    n_seeds: int = 1,
) -> None:
    """Write metrics.json and correct.json for ShinkaEvolve."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    is_correct = error is None and math.isfinite(combined_score)

    # correct.json -- required by ShinkaEvolve
    correct_data = {"correct": is_correct, "error": error}
    (results_path / "correct.json").write_text(json.dumps(correct_data))

    metrics = {
        "combined_score": combined_score if is_correct else -1000.0,
        "correct": is_correct,
        "error": error,
        "execution_time_mean": 0.0,
        "num_valid_runs": n_seeds if is_correct else 0,
        "public_metrics": public_metrics or {},
        "private_metrics": private_metrics or {},
    }
    (results_path / "metrics.json").write_text(json.dumps(metrics, indent=2))


def validate_score(result: dict[str, Any]) -> tuple[bool, str]:
    """Check that the evaluation result has a finite combined_score."""
    if result.get("error"):
        return False, result["error"]
    score = result.get("combined_score", float("-inf"))
    if not math.isfinite(score):
        return False, f"Non-finite combined_score: {score}"
    return True, ""


def aggregate_scores(
    results: list[dict[str, Any]],
    score_key: str = "combined_score",
    window_key: str = "window_scores",
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate multiple evaluation runs into final metrics.

    Parameters
    ----------
    results : list of dicts
        Each dict must have *score_key* (float) and optionally *window_key*
        (list of per-window dicts).
    score_key : str
        Key for the overall score in each result dict.
    window_key : str
        Key for the list of per-window dicts in each result dict.
    metric_keys : list of str, optional
        Which per-window metric keys to summarise. Defaults to
        ``["total_return_pct", "sortino_ratio", "excess_return_pct", "trade_count"]``.

    Returns
    -------
    dict with ``combined_score``, ``public_metrics``, ``private_metrics``.
    """
    if metric_keys is None:
        metric_keys = ["total_return_pct", "sortino_ratio", "excess_return_pct", "trade_count"]

    scores = [r[score_key] for r in results if math.isfinite(r.get(score_key, float("-inf")))]
    if not scores:
        return {
            "combined_score": -1000.0,
            "public_metrics": {"error": "no valid runs"},
            "private_metrics": {},
        }

    combined = float(np.mean(scores))

    # Per-window summary
    all_windows: dict[str, list[dict]] = {}
    for r in results:
        for ws in r.get(window_key, []):
            label = ws.get("label", "unknown")
            all_windows.setdefault(label, []).append(ws)

    window_summary: dict[str, dict[str, float]] = {}
    for label, ws_list in all_windows.items():
        summary: dict[str, float] = {}
        for key in metric_keys:
            vals = [w[key] for w in ws_list if key in w]
            if vals:
                summary[f"mean_{key}"] = float(np.mean(vals))
        window_summary[label] = summary

    # Build public metrics (flat)
    public: dict[str, Any] = {
        "combined_score": round(combined, 4),
        "num_runs": len(scores),
        "score_std": round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0,
    }
    for label, summary in window_summary.items():
        if "mean_sortino_ratio" in summary:
            public[f"{label}_sortino"] = round(summary["mean_sortino_ratio"], 3)
        if "mean_total_return_pct" in summary:
            public[f"{label}_return"] = round(summary["mean_total_return_pct"], 2)

    return {
        "combined_score": combined,
        "public_metrics": public,
        "private_metrics": {
            "all_scores": scores,
            "window_summary": window_summary,
        },
    }


def load_evolved_module(module_path: str) -> ModuleType:
    """Dynamically import a Python module from a file path."""
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("evolved_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_eval_cli(
    run_fn,
    aggregate_fn,
    validate_fn,
    *,
    n_seeds: int = 2,
    default_module: str | None = None,
    write_holdout_fn=None,
) -> None:
    """Generic ``__main__`` entry point for ShinkaEvolve evaluation scripts.

    Parameters
    ----------
    run_fn : callable(module_path) -> (list[dict], int)
        Run multi-seed evaluation, return (results, base_seed).
    aggregate_fn : callable(results) -> dict
        Aggregate results into {combined_score, public_metrics, private_metrics}.
    validate_fn : callable(result) -> (bool, str)
        Validate a single result dict.
    n_seeds : int
        Number of evaluation seeds.
    default_module : str, optional
        Default module path when none specified.
    write_holdout_fn : callable, optional
        If provided, called as write_holdout_fn(results_dir, merged, public, private)
        to add holdout-specific fields to metrics. If None, write_metrics is called
        directly.
    """
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Evaluate evolved program")
    parser.add_argument("--program_path", type=str, default=None, help="Path to evolved module")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory for metrics output")
    parser.add_argument("module_path", nargs="?", default=None, help="Module path (positional, standalone)")
    args = parser.parse_args()

    module_path = args.program_path or args.module_path or default_module

    try:
        results, base_seed = run_fn(module_path)

        # Validate all seed runs
        all_valid = True
        first_err = ""
        for r in results:
            ok, err = validate_fn(r)
            if not ok:
                all_valid = False
                first_err = first_err or err

        if args.results_dir:
            # ShinkaEvolve mode
            agg = aggregate_fn(results) if all_valid else {}
            combined = agg.get("combined_score", -1000.0)

            if write_holdout_fn is not None:
                write_holdout_fn(
                    args.results_dir,
                    results,
                    combined,
                    error=first_err if not all_valid else None,
                    public_metrics=agg.get("public_metrics", {}),
                    private_metrics=agg.get("private_metrics", {}),
                )
            else:
                write_metrics(
                    args.results_dir,
                    combined,
                    error=first_err if not all_valid else None,
                    public_metrics=agg.get("public_metrics", {}),
                    private_metrics=agg.get("private_metrics", {}),
                    n_seeds=n_seeds,
                )
            print(f"Score: {combined:.4f} | seeds={n_seeds} | valid={all_valid}")
        else:
            # Standalone mode
            if not all_valid:
                print(f"INVALID: {first_err}")
                sys.exit(1)

            agg = aggregate_fn(results)
            print(f"\nCombined score (mean of {n_seeds} seeds): {agg['combined_score']:.4f}")
            print(f"Per-seed scores: {[round(r['combined_score'], 4) for r in results]}")
            print(f"Base seed: {base_seed} (from code hash)")

            print("\nPublic metrics:")
            for k, v in agg["public_metrics"].items():
                print(f"  {k}: {v}")

            # Print per-window detail from first seed if available
            if results and "window_scores" in results[0]:
                _print_windows(results[0])

    except Exception as exc:
        if args.results_dir:
            write_metrics(args.results_dir, -1000.0, error=str(exc))
            print(f"Error: {exc}")
        else:
            traceback.print_exc()
            sys.exit(1)


def _print_windows(result: dict[str, Any]) -> None:
    """Print per-window detail for standalone mode."""

    def _fmt(ws: dict) -> None:
        print(
            f"  {ws['label']:25s}"
            f"  ret={ws['total_return_pct']:+7.1f}%"
            f"  sortino={ws['sortino_ratio']:+.2f}"
            f"  dd={ws['max_drawdown_pct']:6.1f}%"
            f"  trades={ws['trade_count']:3d}"
            f"  excess={ws['excess_return_pct']:+7.1f}%"
        )

    if result.get("fitness_scores"):
        print("\nFitness windows (seed 0, in-sample):")
        for ws in result["fitness_scores"]:
            _fmt(ws)

    if result.get("holdout_scores"):
        print("\nHoldout windows (seed 0, out-of-sample):")
        for ws in result["holdout_scores"]:
            _fmt(ws)
