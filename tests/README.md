# Tests

Install the project and its development dependencies with `uv sync`, then
run the suite from the repository root:

```sh
uv run python -m pytest -q
```

Pytest resolves the `src/` package through `pyproject.toml` and uses importlib
collection to support repeated module basenames. Plot tests use the Agg
backend, so a display server is unnecessary.

The runnable tests in `unit/` use synthetic data and temporary files. They
cover player lookup and pipeline timing, event and tracking transforms,
possession joins and tempo, model summaries, and figure reproduction. No
provider downloads or local match data are required.

Historical tests for unavailable APIs are retained in `archive/`. See its
README before reactivating them. New regression tests belong in `unit/`;
keep generated data, caches, and output artifacts out of Git.
