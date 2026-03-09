# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**wzk** (WerkZeugKasten, German for "toolbox") is a personal Python utility library used across robotics/ML/scientific computing projects. Not related to the Python `Werkzeug` package.

## Commands

```bash
# Install (editable, via pixi)
pixi install

# Run all tests
pytest wzk/tests/
pytest wzk/mpl2/tests/
pytest wzk/spatial/tests/
pytest wzk/mc2/tests/

# Run a single test file
pytest wzk/tests/test_math2.py

# Run a single test
pytest wzk/tests/test_math2.py::test_function_name

# Lint and format
ruff check .
ruff format .

# Type check
ty check
```

## Architecture

### Import Side Effects
Importing `wzk` sets multiprocessing-safe env vars (`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, etc.) and auto-configures the matplotlib backend. Import `wzk` before `numpy` or multiprocessing.

### Key Modules
- **`np2/`** — NumPy extensions (basics, find, range, reshape, shape, tile, dtypes)
- **`jax2/`** — JAX reimplementations mirroring `np2`, `math2`, `geometry`, `random2`; uses `jaxtyping` for type annotations
- **`math/`** — Math utilities including `geometry` (triangles, convex hulls, rotations)
- **`spatial/`** — 3D transforms (SE3, DCM, quaternions, euler, DH parameters)
- **`mpl2/`** — Matplotlib extensions (backend, figures, axes, colors, draggable patches, 3D)
- **`io/`** — File I/O (msgpack/pickle/json/mat) and SQLite via pandas (`sql2`)
- **`alg/`** — Algorithms (TSP via OR-Tools, ICP, genetic algorithm)
- **`opt/`** — Optimization (gradient descent, weighted least squares)
- **`wandb2/`** — Optional W&B logging (install with `pip install -e ".[wandb]"`)

### Lazy Loading
Top-level `wzk/__init__.py` uses lazy loading — heavy submodules (`jax2`, `wandb2`, `files`, `sql2`, etc.) are only imported on first attribute access. `wandb2` and `jax2` raise `AttributeError` with install hints if their dependencies are missing.

### JAX Parity
`wzk/jax2/` mirrors NumPy implementations. Parity is enforced by `test_jax2_*_parity.py` tests that verify JAX outputs match NumPy originals.

## Conventions

- **Logging**: Use `from wzk.logger import setup_logger, log_print` with `logger = setup_logger(__name__)`. Use logger levels instead of `verbose` flags or `print()`.
- **Formatting**: ruff with double quotes, 120 char line length, Python 3.13 target.
- **`__init__.py`**: Star imports (`F403`) and unused imports (`F401`) are intentionally allowed.
- **Unused imports/variables**: `F401` and `F841` are marked unfixable in ruff — do not auto-remove them.
- **Tests**: Use plain pytest functions (not `unittest.TestCase`).
- **Type annotations**: Use `from __future__ import annotations` in every file. Add full type signatures (all params + return types). Use PEP 695 `type` keyword for type aliases (not `TypeAlias`). Each module with custom types has a `_types.py` file defining `ArrayLike`, `ShapeLike`, `AxisLike`, `DTypeLike`. JAX modules additionally use `jaxtyping`.
- **Ruff rules**: `I` (isort), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `PERF`, `RUF` are enforced. Use `strict=True` on `zip()` calls. Use `itertools.pairwise()` instead of `zip(x[:-1], x[1:])`.
- **SQL indices**: `sql2.py` uses 1-based indexing (adds +1 offset).
- **Tools**: Run via `.venv/bin/` prefix (e.g. `.venv/bin/ruff`, `.venv/bin/ty`, `.venv/bin/python -m pytest`).
