from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowConfig:
    """A named time window for evaluation."""

    label: str
    start: str
    end: str


@dataclass
class RunnerConfig:
    """Configuration for a ShinkaEvolve runner."""

    # Evolution
    generations: int = 50
    budget: float = 1000.0
    islands: int = 2
    archive_size: int = 30
    eval_workers: int = 2
    proposal_workers: int = 2
    results_dir: str = "shinka_results"

    # LLM
    llm_models: list[str] = field(default_factory=lambda: ["claude-sonnet-4-6"])
    llm_max_tokens: int = 16384

    # System prompt (domain-specific)
    task_sys_msg: str = ""

    # Program paths
    init_program_path: str = ""
    eval_program_path: str = ""

    # Evolution tuning
    language: str = "python"
    patch_types: list[str] = field(default_factory=lambda: ["diff", "full", "cross"])
    patch_type_probs: list[float] = field(default_factory=lambda: [0.5, 0.35, 0.15])
    meta_rec_interval: int = 5
    code_embed_sim_threshold: float = 0.98
    embedding_model: str | None = None
    eval_timeout: str = "00:03:00"

    # Prompt evolution
    evolve_prompts: bool = True
    prompt_evolution_interval: int = 5
    prompt_archive_size: int = 10
    prompt_llm_models: list[str] = field(default_factory=lambda: ["claude-sonnet-4-6"])

    # Database tuning
    parent_selection_strategy: str = "weighted"
    num_archive_inspirations: int = 3
    num_top_k_inspirations: int = 2
    migration_interval: int = 8
    migration_rate: float = 0.15
