# Agent Instructions (Compiled From User Requests)

This file consolidates all user instructions given in this thread.

## 1. JAX Porting Scope

1. Port `math2`, `geometry`, and `np2` to JAX.
2. Write the new JAX code in a new directory: `jax2`.
3. Port `random2.py` to `jax2` as well.

## 2. Typing Requirements (JAX Modules)

1. Add typing to JAX modules.
2. Use `jaxtyping`.
3. Create `jax2._types` and define shared aliases/constants for:
   - `float32`
   - `int32`
4. Use those shared types consistently throughout JAX modules.

## 3. Behavior-Parity Testing

1. Add tests that ensure JAX behavior matches NumPy versions.
2. Include parity checks for the ported modules.

## 4. TSP / Ordering Requirement

1. Implement a NumPy version of TSP-based ordering in `tsp` using this `q_anchor` behavior:
   - Use `_order_q_with_tsp(..., anchor_q=...)` behavior equivalent to the provided snippet.
2. Add default behavior:
   - If `n <= 2`, return `np.arange(n)`.

## 5. Logging and Verbosity Cleanup

1. Search through the whole repo for:
   - `verbose` arguments
   - `print` statements
2. Replace them with structured logging.
3. Remove all `verbose: int` function arguments.
4. Use logger levels instead of `verbose` flags.
5. Use the same logger setup pattern as `../kinematix`.

## 6. Codebase Hygiene and Pitfalls

1. Identify big pitfalls in the codebase.
2. Fix all pitfalls by priority.
3. Resolve remaining TODOs after user rewrites.

## 7. Tooling and Dependencies

1. Add dependencies with `uv`:
   - `jax`
   - `pytest`
   - `ruff`
   - `ty`
2. Remove usage of:
   - `unittest`
   - `py_compile`

## 8. Git Ignore Rules

1. Add `uv.lock` to `.gitignore`.
2. Add image/document patterns to `.gitignore`:
   - `*.png`
   - `*.pdf`
   - `*.jpeg`

## 9. Commit Workflow

1. Prepare a commit checkpoint first before continuing major follow-up work.
