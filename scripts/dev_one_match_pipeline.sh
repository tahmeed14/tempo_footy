#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <match_id> [match_id ...]" >&2
    exit 2
fi

match_ids=("$@")
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_directory}/.." && pwd)
cd "${repository_root}"

metadata_directory="data/raw/gradient_sports/metadata"
possession_lookup_path="data/curated/gradient_sports/possession_lookup/match_possession_lookup.parquet"
divider="========================================================================"
match_ids_seen=""
processed_event_paths=()
integrated_paths=()

for match_id in "${match_ids[@]}"; do
    if [[ ! ${match_id} =~ ^[0-9]+$ ]]; then
        echo "Match IDs must be numeric: ${match_id}" >&2
        exit 2
    fi

    if [[ " ${match_ids_seen} " == *" ${match_id} "* ]]; then
        echo "Duplicate match ID: ${match_id}" >&2
        exit 2
    fi
    match_ids_seen+=" ${match_id}"

    raw_event_path="data/raw/gradient_sports/events/${match_id}.json"
    raw_tracking_path="data/raw/gradient_sports/tracking/${match_id}.jsonl.bz2"
    if [[ ! -f ${raw_event_path} ]]; then
        echo "Raw event file not found: ${raw_event_path}" >&2
        exit 1
    fi
    if [[ ! -f ${raw_tracking_path} ]]; then
        echo "Raw tracking file not found: ${raw_tracking_path}" >&2
        exit 1
    fi

    processed_event_paths+=(
        "data/processed/gradient_sports/events/${match_id}.parquet"
    )
    integrated_paths+=(
        "data/integrated/gradient_sports/${match_id}.parquet"
    )
done

match_label=$(IFS=_; echo "${match_ids[*]}")
match_description=$(IFS=,; echo "${match_ids[*]}")
if [[ ${#match_ids[@]} -eq 1 ]]; then
    investigation_directory="data/investigate/one_match/${match_label}"
else
    investigation_directory="data/investigate/selected_matches/${match_label}"
fi
player_possessions_path="${investigation_directory}/player_possessions.parquet"
modeldata_path="${investigation_directory}/modeldata_v0.parquet"

mkdir -p "${investigation_directory}"

run_step() {
    local step_name=$1
    shift
    local started_at=$SECONDS

    echo "${divider}"
    echo "${step_name}"
    echo "Selected matches: ${match_description}"
    echo "${divider}"
    "$@"
    echo "${step_name} completed in $((SECONDS - started_at)) seconds"
    echo "${divider}"
    echo
}

run_step "PLAYER LOOKUP PIPELINE" \
    uv run python pipelines/pipeline_players.py

run_step "EVENT PIPELINE — SELECTED MATCHES" \
    uv run python -c \
    'import sys; from pipelines.pipeline_events import configure_logging, run_pipeline; configure_logging(); run_pipeline([int(value) for value in sys.argv[1:]])' \
    "${match_ids[@]}"

for match_id in "${match_ids[@]}"; do
    run_step "TRACKING PIPELINE — MATCH ${match_id}" \
        uv run python -c \
        'import sys; from pipelines.pipeline_tracking import configure_logging, run_pipeline; configure_logging(); run_pipeline(int(sys.argv[1]))' \
        "${match_id}"

    run_step "INTEGRATION PIPELINE — MATCH ${match_id}" \
        uv run python -c \
        'import sys; from pipelines.pipeline_integrate import configure_logging, run_pipeline; configure_logging(); run_pipeline(int(sys.argv[1]))' \
        "${match_id}"
done

run_step "POSSESSION & TEMPO METRICS PIPELINE" \
    uv run python -c \
    'import sys; from pipelines.pipeline_possessions import configure_logging, run_pipeline; configure_logging(); run_pipeline(df_path=sys.argv[3:], metadata_dir=sys.argv[1], output_dir=sys.argv[2])' \
    "${metadata_directory}" \
    "${investigation_directory}" \
    "${integrated_paths[@]}"

run_step "MODEL DATA PIPELINE" \
    uv run python -c \
    'import sys; from pipelines.pipeline_modeldata import configure_logging, run_pipeline; configure_logging(); run_pipeline(events_path=sys.argv[4:], player_possessions_path=sys.argv[1], possession_lookup_path=sys.argv[2], output_path=sys.argv[3])' \
    "${player_possessions_path}" \
    "${possession_lookup_path}" \
    "${modeldata_path}" \
    "${processed_event_paths[@]}"

echo "Selected-match investigation outputs: ${investigation_directory}"
