#!/usr/bin/env python3
"""Create the local directories required by the pipelines and paper scripts.

Run this script after cloning the repository:

    python initiate_repository.py

Existing directories and their contents are left unchanged.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORIES = (
    Path("data/analysis"),
    Path("data/curated/gradient_sports/metadata_lookup"),
    Path("data/curated/gradient_sports/possession_lookup"),
    Path("data/integrated/gradient_sports"),
    Path("data/investigate/join_issues"),
    Path("data/processed/gradient_sports/events"),
    Path("data/processed/gradient_sports/tracking"),
    Path("data/raw/gradient_sports/events"),
    Path("data/raw/gradient_sports/metadata"),
    Path("data/raw/gradient_sports/roster"),
    Path("data/raw/gradient_sports/tracking"),
    Path("data/staged/gradient_sports/tracking"),
    Path("paper/figures"),
    Path("paper/results"),
    Path("paper/tables"),
)


def initialize_repository(
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[Path, ...]:
    """Create required repository directories beneath a repository root.

    Args:
        repository_root: Root directory of the repository to initialize.

    Returns:
        Absolute paths to all required repository directories.

    """
    root_path = Path(repository_root).resolve()
    created_paths = tuple(root_path / path for path in DATA_DIRECTORIES)

    for directory_path in created_paths:
        directory_path.mkdir(parents=True, exist_ok=True)

    return created_paths


def main() -> None:
    """Create and display the required repository directories."""
    created_paths = initialize_repository()
    print("Repository directories are ready:")
    for directory_path in created_paths:
        print(f"  - {directory_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
