"""Artifact and memory helpers for the sequential BST teaching notebooks.

The modeling and sampling steps stay visible in the notebooks. These helpers
only handle bookkeeping, snapshot integrity, and allocation estimates.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from tempoctrl.research.bst_models import (
    FAMILIES,
    DISPERSION_PREDICTORS,
    PreparedData,
    design_matrix,
    file_sha256,
    repository_root,
    specification_hash,
)
from tempoctrl.research.bst_workflow import RunConfig, memory_estimate, run_family
from tempoctrl.research.bst_evaluation import markdown_table
from tempoctrl.reproduce_tables import CATEGORICAL_PREDICTORS, CONTINUOUS_PREDICTORS


def interactive_model_source_hash(path: str | Path) -> str:
    """Hash model code independently of editable execution settings.

    Prior names, data paths and sampler counts are recorded in the manifest.
    Editing those settings must not prevent comparisons between intended prior
    sensitivity runs. Changes to distributions, priors or evaluation code do.
    Full unmodified source bytes are saved separately for provenance.
    """
    tree = ast.parse(Path(path).read_text())
    settings = {
        "PRIOR_NAME", "ROOT", "DATA_PATH", "SAVED_RUN_DIRECTORY",
        "RUN_PRIOR_PREDICTIVE", "RUN_MCMC", "COMPARISON_RUNS", "CONFIG",
    }
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in settings for target in node.targets
        ))
    ]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def start_interactive_run(data: PreparedData, config: RunConfig) -> Path:
    """Freeze the inspected cohort without fitting; preserve teaching sources.

    Reuse preparation-only artifact creation from the existing workflow. Check
    that the input has not changed since the user inspected it. Hash the whole
    set of teaching scripts together so different families can be compared
    while changes to their implementation invalidate old comparisons.
    """
    if config.prior == "all":
        raise ValueError("Choose one prior configuration for a sequential walkthrough.")
    directory = run_family(replace(config, mode="prepare"))[0]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest["data"][key] != data.audit[key] for key in ("source_sha256", "cohort_sha256")):
        manifest.update(status="failed", error="Input changed after inspection; restart the data cells.")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise ValueError(manifest["error"])
    source_directory = directory / "interactive_sources"
    source_directory.mkdir()
    root = repository_root()
    sources = [Path(__file__)] + [
        root / "notebooks/research_investigations" / f"{family}_model_comparison_interactive.py"
        for family in FAMILIES
    ]
    hashes = {path.name: file_sha256(path) for path in sources}
    model_hashes = {
        path.name: interactive_model_source_hash(path) if path.name.endswith("_interactive.py") else hashes[path.name]
        for path in sources
    }
    for path in sources:
        shutil.copyfile(path, source_directory / path.name)
    manifest.update(
        workflow="sequential_interactive",
        interactive_sources=hashes,
        interactive_model_sources=model_hashes,
        implementation_sha256=specification_hash({
            "shared_workflow": manifest["implementation_sha256"],
            "interactive_model_sources": model_hashes,
        }),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return directory


def load_interactive_snapshot(directory: str | Path, family: str) -> tuple[PreparedData, RunConfig]:
    """Load the frozen design and its settings, independent of current inputs."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["family"] != family:
        raise ValueError(f"Saved family is {manifest['family']}, expected {family}.")
    snapshot = directory / "observations.parquet"
    if file_sha256(snapshot) != manifest["snapshot_sha256"]:
        raise ValueError("Saved observation snapshot changed; use the original artifact.")
    # Pandas restores the category metadata it wrote, avoiding an Arrow enum
    # incompatibility in the Polars reader. Reuse the integrity-checked saved
    # z score and its original scaler; do not learn preprocessing again.
    frame = pd.read_parquet(snapshot)
    cohort_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    if cohort_hash != manifest["data"]["cohort_sha256"]:
        raise ValueError("Saved cohort differs from its recorded fingerprint.")
    mean_design = design_matrix(frame, (*CONTINUOUS_PREDICTORS, *CATEGORICAL_PREDICTORS))
    dispersion_design = design_matrix(frame, DISPERSION_PREDICTORS)
    if (list(mean_design) != manifest["data"]["mean_columns"]
            or list(dispersion_design) != manifest["data"]["dispersion_columns"]):
        raise ValueError("Reconstructed contrasts differ from the saved design.")
    players = sorted(frame["player_id"].unique())
    opponents = sorted(frame["opponent_id"].unique())
    data = PreparedData(
        frame=frame,
        exclusions=pd.read_csv(directory / "exclusions.csv"),
        mean_design=mean_design,
        dispersion_design=dispersion_design,
        player_index=pd.Categorical(frame["player_id"], categories=players).codes,
        opponent_index=pd.Categorical(frame["opponent_id"], categories=opponents).codes,
        players=players,
        opponents=opponents,
        audit=manifest["data"],
    )
    return data, RunConfig(**manifest["settings"])


def check_interactive_memory(data: PreparedData, config: RunConfig, *, stage: str) -> dict:
    """Check the same allocation estimates as the batch workflow, stage by stage."""
    estimate = memory_estimate(len(data.frame), config)
    if stage == "prior":
        required = 3 * estimate["prior_predictive_gib"]
    elif stage == "posterior":
        required = estimate["approximate_working_gib"]
    else:
        raise ValueError("stage must be 'prior' or 'posterior'.")
    if required > config.max_working_gib:
        raise MemoryError(
            f"Estimated working arrays need {required:.2f} GiB; budget is "
            f"{config.max_working_gib:.2f}. Reduce draws or adjust max_working_gib."
        )
    return estimate


def update_interactive_status(directory: str | Path, status: str) -> None:
    """Record progress between visible notebook cells without changing run identity."""
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_interactive_results(directory, sampling, loo, moments, predictive_checks) -> Path:
    """Save conclusions from results already computed in the visible cells.

    This does not rerun MCMC, posterior prediction or PSIS-LOO.
    """
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    evaluation = {**sampling, **loo}
    healthy = sampling["sampling_diagnostics_pass"] and loo["loo_diagnostics_pass"]
    overall = predictive_checks.query("variable == 'overall'")
    misses = overall[
        (overall.observed < overall.predictive_q025)
        | (overall.observed > overall.predictive_q975)
    ]
    text = [
        "# Interactive BST results", "",
        f"Family: {manifest['family']}; prior: {manifest['prior']['name']}; "
        f"observations: {manifest['data']['included_rows']}.", "",
        "Sampling and PSIS diagnostic gates passed." if healthy else
        "A sampling or PSIS diagnostic gate failed. Resolve it before interpreting or ranking fits.",
        "", markdown_table(pd.DataFrame([evaluation])), "",
        "## Reference conditional mean, SD and CV", "", markdown_table(moments), "",
        "These population reference-context quantities set group effects to zero. "
        "Their equal-tail intervals describe posterior uncertainty; SD and CV describe outcome variability.",
        "", "## Overall posterior predictive checks", "", markdown_table(overall), "",
        ("Observed summaries outside their 95% predictive replicate intervals: "
         + ", ".join(misses.statistic) + ". Inspect these discrepancies.") if len(misses) else
        "No listed pooled summary lies outside its 95% predictive replicate interval. "
        "Inspect subgroup, player and tail checks before assessing adequacy.", "",
        "## Inferences and remaining comparisons", "",
        "One run cannot establish a preferred distribution or prior robustness. "
        "Compare aligned saved fits, keeping within-family prior sensitivity separate "
        "from family sensitivity within a prior. Judge paired ELPD differences against "
        "their uncertainty, and retain predictive deficiencies in the interpretation. "
        "LOO targets additional possessions in represented groups; new-match or future-match "
        "claims require dedicated refits. Common-context player/archetype inference remains "
        "a later stage. Proposed priors require substantive prior predictive review.", "",
        "See methods.md and the saved interactive sources for the formulation and evaluation rationale.",
    ]
    report = directory / "results.md"
    report.write_text("\n".join(text) + "\n")
    # Nonfinite diagnostics signal failure; JSON records them as null.
    evaluation = {
        key: value if not isinstance(value, float) or math.isfinite(value) else None
        for key, value in evaluation.items()
    }
    manifest.update(
        status="complete", evaluation=evaluation,
        evaluation_implementation_sha256=specification_hash({
            name: file_sha256(Path(__file__).parent / name)
            for name in ("bst_interactive.py", "bst_evaluation.py")
        }),
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
