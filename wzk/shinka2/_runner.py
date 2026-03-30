from __future__ import annotations

import argparse
from typing import Any

from ._config import RunnerConfig

__all__ = ["add_runner_args", "build_runner", "runner_config_from_args"]


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    """Add standard ShinkaEvolve runner arguments to an argparse parser."""
    parser.add_argument("--generations", type=int, default=50, help="Number of evolutionary generations")
    parser.add_argument("--budget", type=float, default=1000.0, help="Max LLM API cost in USD")
    parser.add_argument("--islands", type=int, default=2, help="Number of evolutionary islands")
    parser.add_argument("--archive-size", type=int, default=30, help="Archive size per island")
    parser.add_argument("--eval-workers", type=int, default=2, help="Parallel evaluation workers")
    parser.add_argument("--proposal-workers", type=int, default=2, help="Parallel LLM proposal workers")
    parser.add_argument("--results-dir", type=str, default="shinka_results", help="Results directory (reuse to resume)")


def runner_config_from_args(args: argparse.Namespace, **overrides: Any) -> RunnerConfig:
    """Build a RunnerConfig from parsed argparse args with optional overrides."""
    kwargs: dict[str, Any] = {
        "generations": args.generations,
        "budget": args.budget,
        "islands": args.islands,
        "archive_size": args.archive_size,
        "eval_workers": args.eval_workers,
        "proposal_workers": args.proposal_workers,
        "results_dir": args.results_dir,
    }
    kwargs.update(overrides)
    return RunnerConfig(**kwargs)


def build_runner(config: RunnerConfig) -> Any:
    """Lazy-import shinka-evolve and construct a ShinkaEvolveRunner.

    Returns the runner instance ready to call ``.run()``.
    """
    try:
        from shinka.core import EvolutionConfig, ShinkaEvolveRunner
        from shinka.database import DatabaseConfig
        from shinka.launch import LocalJobConfig
    except ImportError:
        print("ShinkaEvolve not installed. Run: pip install shinka-evolve")
        raise SystemExit(1) from None

    evo_config = EvolutionConfig(
        num_generations=config.generations,
        task_sys_msg=config.task_sys_msg,
        language=config.language,
        patch_types=config.patch_types,
        patch_type_probs=config.patch_type_probs,
        llm_models=config.llm_models,
        llm_kwargs={"max_tokens": config.llm_max_tokens},
        max_api_costs=config.budget,
        meta_rec_interval=config.meta_rec_interval,
        code_embed_sim_threshold=config.code_embed_sim_threshold,
        embedding_model=config.embedding_model,
        init_program_path=config.init_program_path,
        results_dir=config.results_dir,
        evolve_prompts=config.evolve_prompts,
        prompt_evolution_interval=config.prompt_evolution_interval,
        prompt_archive_size=config.prompt_archive_size,
        prompt_llm_models=config.prompt_llm_models,
    )

    db_config = DatabaseConfig(
        num_islands=config.islands,
        archive_size=config.archive_size,
        parent_selection_strategy=config.parent_selection_strategy,
        num_archive_inspirations=config.num_archive_inspirations,
        num_top_k_inspirations=config.num_top_k_inspirations,
        migration_interval=config.migration_interval,
        migration_rate=config.migration_rate,
    )

    job_config = LocalJobConfig(
        eval_program_path=config.eval_program_path,
        time=config.eval_timeout,
    )

    return ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.eval_workers,
        max_proposal_jobs=config.proposal_workers,
    )
