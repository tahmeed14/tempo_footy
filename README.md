# "Rise to the Rhythm"

Open source repository for research project on tempo in football (soccer)

## Prerequisites

Before running the project, make sure the following are available:

- At least 20 GB of free disk space for the complete 64-match workflow.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for managing
  Python and the project dependencies.
- Python 3.12 or newer. Older versions will not run end to end. `uv` can
  download a compatible Python version when one is not already installed.
- Your own Jupyter-compatible notebook application for fitting the model
  interactively. Examples include VS Code with the Jupyter extension,
  JupyterLab, or Jupyter Notebook.

The project installs an IPython kernel, but it does not install a graphical
Jupyter interface. When opening `src/tempoctrl/cmsac_model.ipynb`, select the
Python interpreter from this project's `.venv` as the notebook kernel.

## Install uv and create the project environment

On macOS or Linux, install `uv` with its official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run the official installer from PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Other installation methods are listed in
the [`uv` installation documentation](https://docs.astral.sh/uv/getting-started/installation/).

After cloning the repository, change into its root directory and create the
locked project environment:

```bash
uv sync --frozen
```

This creates `.venv`, installs the Python version and dependencies specified by
the project, and preserves the package versions recorded in `uv.lock`.

### Set up directories

Before downloading any data, run the repository initializer from the
repository root:

```bash
uv run python initiate_repository.py
```

This creates all required directories but does not download any data. After it
finishes, copy the downloaded files into the appropriate destinations.

## Download the public data

This project uses the 64-match FIFA 2022 World Cup datasets released publicly
by Gradient Sports (formerly PFF FC). Gradient Sports does not provide an API for this
dataset, so the files must be downloaded and placed in the repository **manually**.

Download the datasets from the
[Gradient Sports Google Drive folder](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa).

The download contains four data types. Copy the files for each type into the
matching repository directory:

| Downloaded data type | Required destination |
|---|---|
| `events` | `data/raw/gradient_sports/events/` |
| `tracking` | `data/raw/gradient_sports/tracking/` |
| `metadata` | `data/raw/gradient_sports/metadata/` |
| `roster` | `data/raw/gradient_sports/roster/` |

The directory names and downloaded filenames must match exactly. Do not rename
the data-type folders or source files. In particular, use lowercase directory
names as shown above and preserve these file patterns:

- Events: `<match_id>.json`
- Tracking: `<match_id>.jsonl.bz2`
- Metadata: `<match_id>.json`
- Rosters: `<match_id>.json`

For example, the event file for match `10517` must be located at
`data/raw/gradient_sports/events/10517.json`.

### Disk-space requirements

The complete raw Gradient Sports download uses approximately **3.7 GB**. A
full run retains raw, staged, processed, integrated, and analysis datasets; in
the current project these occupy approximately **12 GB** in total. The fitted
posterior files under `paper/results/` require roughly another **734 MB**.

Plan for at least **15 GB** of storage for all retained data and model files.
Having **20 GB of free disk space** before running the complete pipeline is
recommended to provide room for temporary files and filesystem overhead.
Exact usage may vary slightly by platform and file version. This repo attempts to follow good data engineering practices in development. As a result, we require staging and stage by stage snapshots of the data. The final version of this open source repo will significantly reduce the disk space requirements.

The full 64-match dataset is not required for development or testing. To reduce
storage and processing time, download the files for only the matches of
interest and use the selected-match pipeline described below. Each selected
match must still have matching event, tracking, metadata, and roster files in
the required raw-data directories.

## Reproducible pipeline

### 1. Run the full development pipeline

Run the full pipeline from the repository root:

```bash
bash scripts/dev_full_pipeline.sh
```

The script runs the player lookup, event, tracking, integration, possession and
tempo-metric, and model-data pipelines in order. Pipeline outputs are written
to the standard directories under `data/curated`, `data/staged`,
`data/processed`, `data/integrated`, and `data/analysis`.

The run time is dependent on the number of matches being processed.

### 2. Reproduce the model dataframe

After the development pipeline finishes, create the final dataframe used to
fit the model:

```bash
bash scripts/reproduce_model_df.sh
```

This reads `data/analysis/modeldata_v0.parquet` and writes the curated model
dataframe to `data/analysis/model_data_vFINAL.parquet`.

### 3. Fit the model

Open `src/tempoctrl/cmsac_model.ipynb` in Jupyter or VS Code and run the entire
notebook from top to bottom. The notebook reads the final model dataframe, fits
the Bayesian gamma model, and writes the fitted posterior to
`paper/results/tempo_gamma_posterior.nc`.

The notebook must finish successfully before the paper tables and figures can
be reproduced.

### 4. Reproduce the paper figures and tables

After the model notebook finishes, run:

```bash
bash scripts/reproduce_paper_figs_and_tables.sh
```

This regenerates the model-summary, fixed-effect, random-effect, and player-
archetype tables under `paper/tables/`. It also regenerates the possession,
tempo-distribution, player-effect, and player-archetype figures under
`paper/figures/`. Both PDF and PNG versions of those files are generated for 
downstream Quarto paper writing.

### OPTIONAL: Run the development pipeline for selected matches

To avoid running the whole set of matches, you can pass one or more numeric match IDs to
the selected-match development script:

```bash
# One match
bash scripts/dev_one_match_pipeline.sh 3859

# Multiple matches processed as one combined investigation subset
bash scripts/dev_one_match_pipeline.sh 3859 3860 3861
```

This option is useful for **only** for development and testing when running the complete
64-match workflow would require too much storage or processing time. Selected-
match outputs are written under `data/investigate/` and do not replace the
full-sample model inputs required to reproduce the paper results. 

## Development and testing environments

## macOS

Specifications:

- MacBook Air M4 
- SSD: 512 GB
- RAM: 16 GB
- Apple Silicon Chip

### End-to-end reproduction time

The complete workflow was timed on the macOS development environment described
above using a clean clone, the full 64-match raw dataset, a new `uv`
environment, and a newly fitted model. No staged, processed, analysis, or
posterior files were carried into the run.

- `bash scripts/dev_full_pipeline.sh` took 31 minutes, 34 seconds.
- `bash scripts/reproduce_model_df.sh` took 9 seconds.
- Running `src/tempoctrl/cmsac_model.ipynb` from top to bottom took 12 minutes,
  57 seconds.
- `bash scripts/reproduce_paper_figs_and_tables.sh` took 17 seconds.

The complete four-stage workflow took approximately 44 minutes, 58 seconds.
Runtime will vary with hardware, operating system, and available processor
cores.

## Linux Mint xFCE

To be added soon...
