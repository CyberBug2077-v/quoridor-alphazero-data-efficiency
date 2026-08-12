#!/bin/bash

accel_submit_job() {
    local worker="$1"
    local results_root="$2"
    local exp_name="$3"
    local job_name="$4"
    shift 4

    mkdir -p "${results_root}/${exp_name}"

    local export_vars=(
        "RESULTS_ROOT=${results_root}"
        "EXP_NAME=${exp_name}"
        "$@"
    )
    local export_csv
    export_csv=$(IFS=,; echo "ALL,${export_vars[*]}")

    local sbatch_args=()
    if [ -n "${SBATCH_NODELIST:-}" ]; then
        sbatch_args+=(--nodelist="${SBATCH_NODELIST}")
    fi
    if [ -n "${SBATCH_CONSTRAINT:-}" ]; then
        sbatch_args+=(--constraint="${SBATCH_CONSTRAINT}")
    fi
    if [ -n "${SBATCH_GRES:-}" ]; then
        sbatch_args+=(--gres="${SBATCH_GRES}")
    fi

    sbatch \
        "${sbatch_args[@]}" \
        --job-name="${job_name}" \
        --output="${results_root}/${exp_name}/slurm-%j.out" \
        --export="${export_csv}" \
        "${worker}"
}
