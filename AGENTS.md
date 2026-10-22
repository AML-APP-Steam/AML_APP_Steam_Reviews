# AGENTS.md

## Core Principles

- **Concise & Direct**: Write minimal, clean, and self-documenting code. Never add unnecessary bloat, placeholder code, or speculative features.
- **Strict Adherence**: Follow instructions precisely. Do only what is requested—no more, no less.
- **No Comments**: Code must be clear enough to explain itself. Do not include comments, inline commentary.
- **Docstrings**: Docstrings are required for each file and function with description attributes and return type, and exceptions if applicable

---

## Code Quality & Tooling

### Python Standards
- Fully type-annotate every function signature (arguments and return types) and non-obvious variable assignments.
- Pass strict `mypy` type checking without warnings or errors.
- Pass strict `ruff` linting and formatting rules without errors.

### Package & Defensive Design
- Raise appropriate standard Python exceptions (`ValueError`, `TypeError`, `KeyError`, etc.) or project-specific custom exceptions with clear error messages when invalid data or states are encountered.
- Keep defensive checks lightweight and focused; avoid over-engineering validation logic.

### Testing
- Write fast, deterministic unit tests using `pytest`.
- Aim for high test coverage on all added or modified functionality.
- Place tests in the standard `tests/` directory mirroring module structure.

---

## Agent Output Rules

1. **No Unrequested Changes**: Do not refactor untouched code unless specifically directed.
2. **No Over-Explaining**: Avoid long conversational responses, preamble, or post-implementation chatter. State what was done concisely.
3. **Run Validation**: Ensure all modified files pass `mypy`, `ruff`, and `pytest` before finalizing tasks.
