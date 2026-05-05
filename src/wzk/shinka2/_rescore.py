"""Generic rescore pattern for ShinkaEvolve databases."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["rescore_database"]


def rescore_database(
    results_dir: str,
    run_multi_seed_fn: Callable[[str], tuple[list[dict[str, Any]], int]],
    aggregate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    validate_fn: Callable[[dict[str, Any]], tuple[bool, str]],
    compute_combined_fn: Callable[[list[dict[str, Any]]], float] | None = None,
    *,
    score_key: str = "combined_score",
    dry_run: bool = False,
) -> None:
    """Rescore all programs in a ShinkaEvolve SQLite database.

    Extracts each program's code, evaluates it with the provided functions,
    and updates combined_score / public_metrics / private_metrics in-place.
    Also rewrites gen_N/results/metrics.json files.

    Parameters
    ----------
    results_dir : str
        Path to the ShinkaEvolve results directory containing programs.sqlite.
    run_multi_seed_fn : callable(module_path) -> (results, base_seed)
        Evaluate a module file and return list of result dicts + base seed.
    aggregate_fn : callable(results) -> dict
        Aggregate results into {combined_score, public_metrics, private_metrics}.
    validate_fn : callable(result) -> (ok, error_msg)
        Validate a single evaluation result.
    compute_combined_fn : callable(window_scores) -> float, optional
        Compute holdout score from window scores.
    score_key : str
        Key for the combined score in result dicts.
    dry_run : bool
        If True, evaluate but don't write changes.
    """
    from ._evaluate import write_metrics

    results_path = Path(results_dir)
    db_path = results_path / "programs.sqlite"

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute("SELECT id, code, combined_score, generation FROM programs ORDER BY rowid")
    rows = cursor.fetchall()
    print(f"Total programs: {len(rows)}")

    # Deduplicate: evaluate each unique code once
    code_to_rows: dict[str, list[tuple[str, float, int]]] = {}
    for pid, code, old_score, gen in rows:
        code_to_rows.setdefault(code, []).append((pid, old_score, gen))

    unique_codes = list(code_to_rows.keys())
    print(f"Unique codes: {len(unique_codes)}")
    if dry_run:
        print("DRY RUN -- no changes will be made")
    print()

    updated = 0
    errors = 0
    t0 = time.monotonic()

    for i, code in enumerate(unique_codes):
        code_rows = code_to_rows[code]
        old_score = code_rows[0][1]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            results, base_seed = run_multi_seed_fn(tmp_path)
        except Exception as exc:
            print(f"  [{i + 1}/{len(unique_codes)}] ERROR ({len(code_rows)} rows): {exc}")
            errors += len(code_rows)
            Path(tmp_path).unlink(missing_ok=True)
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        all_valid = all(validate_fn(r)[0] for r in results)
        if not all_valid:
            print(f"  [{i + 1}/{len(unique_codes)}] INVALID ({len(code_rows)} rows)")
            errors += len(code_rows)
            continue

        agg = aggregate_fn(results)
        new_score = agg[score_key]
        public_metrics = agg.get("public_metrics", {})
        private_metrics = agg.get("private_metrics", {})

        # Compute holdout score if function provided
        if compute_combined_fn is not None:
            holdout_scores = [ws for r in results for ws in r.get("holdout_scores", [])]
            if holdout_scores:
                holdout_score = compute_combined_fn(holdout_scores)
                if math.isfinite(holdout_score):
                    public_metrics["holdout_score"] = round(holdout_score, 4)

        pub_json = json.dumps(public_metrics)
        priv_json = json.dumps(private_metrics)

        if not dry_run:
            for pid, _, _ in code_rows:
                cursor.execute(
                    "UPDATE programs SET combined_score = ?, public_metrics = ?, private_metrics = ? WHERE id = ?",
                    (new_score, pub_json, priv_json, pid),
                )

            for _, _, gen in code_rows:
                gen_results_dir = results_path / f"gen_{gen}" / "results"
                if gen_results_dir.exists():
                    write_metrics(
                        str(gen_results_dir),
                        new_score,
                        public_metrics=public_metrics,
                        private_metrics=private_metrics,
                    )

        updated += len(code_rows)
        elapsed = time.monotonic() - t0
        rate = (i + 1) / elapsed
        eta = (len(unique_codes) - i - 1) / rate if rate > 0 else 0
        print(
            f"  [{i + 1}/{len(unique_codes)}] {old_score:+.4f} -> {new_score:+.4f}"
            f"  ({len(code_rows)} rows)  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
        )

    if not dry_run:
        conn.commit()

    conn.close()
    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s -- updated {updated}, errors {errors}")
