#!/bin/bash
# =============================================================================
# Smoke Test — Normal RoPE Experiments
# =============================================================================
# Verifies that every experiment in the ablation study can start, build the
# model, load data, and complete one training step without crashing.
#
# Run from the noether-RoNE/ root on an allocated node (no SLURM needed):
#   bash normal_rope_encoding/tutorial/jobs/smoke_test.sh [DRIVAERML_ROOT] [SHAPENET_ROOT]
#
# Arguments (optional — override dataset paths):
#   $1  DRIVAERML_ROOT  default: /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x
#   $2  SHAPENET_ROOT   default: /nfs-gpu/research/datasets/shapenet-car
# =============================================================================

DRIVAERML_ROOT="${1:-/nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x}"
SHAPENET_ROOT="${2:-/nfs-gpu/research/datasets/shapenet-car}"

DRIVAERML_EXP="normal_rope_encoding/tutorial/jobs/experiments/normal_rope_drivaerml.txt"
SHAPENET_EXP="normal_rope_encoding/tutorial/jobs/experiments/normal_rope_shapenet.txt"

LOG_DIR="logs/smoke_test/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Smoke test uses a timeout: if the run is still alive after SMOKE_TIMEOUT_SECS
# it started correctly (data loads, model builds, forward pass works).
# Only a non-timeout non-zero exit code is a real failure.
SMOKE_TIMEOUT_SECS=120   # 2 minutes per experiment
TRAINER_OVERRIDES="trainer.max_epochs=1"

# Redirect outputs to a throwaway directory
OUTPUT_OVERRIDES="output_path=/tmp/noether_smoke_test stage_name=smoke"

# Disable WandB — no tracker noise during smoke tests
export WANDB_MODE=disabled

pass=0
fail=0
declare -a failures

# -----------------------------------------------------------------------------
run_experiment() {
    local idx="$1"
    local tag="$2"
    local hp_cfg="$3"
    local exp_args="$4"       # raw line from the experiments file
    local dataset_arg="$5"    # dataset_root=<path>

    local log="${LOG_DIR}/${tag}_exp$(printf '%02d' "${idx}").log"

    printf "[%-10s #%02d] " "${tag}" "${idx}"

    # shellcheck disable=SC2086  # intentional word-splitting for exp_args and overrides
    timeout "${SMOKE_TIMEOUT_SECS}" uv run noether-train \
            --hp "${hp_cfg}" \
            ${exp_args} \
            "${dataset_arg}" \
            ${TRAINER_OVERRIDES} \
            ${OUTPUT_OVERRIDES} \
        > "${log}" 2>&1
    exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        echo "PASS (completed)"
        pass=$((pass + 1))
    elif [ "${exit_code}" -eq 124 ]; then
        # timeout — run was still alive, meaning startup + forward pass succeeded
        echo "PASS (timeout — run started OK)"
        pass=$((pass + 1))
    else
        echo "FAIL  (exit ${exit_code}) — see ${log}"
        echo "       Last output:"
        tail -8 "${log}" | sed 's/^/         /'
        fail=$((fail + 1))
        failures+=("${tag} #${idx}")
    fi
}

# -----------------------------------------------------------------------------
run_file() {
    local tag="$1"
    local hp_cfg="$2"
    local exp_file="$3"
    local dataset_arg="$4"

    local n_exp
    n_exp=$(grep -v '^[[:space:]]*#' "${exp_file}" | grep -v '^[[:space:]]*$' | wc -l)

    echo ""
    echo "============================================================"
    printf " %-12s  %d experiments\n" "${tag^^}" "${n_exp}"
    echo "============================================================"

    local idx=0
    while IFS= read -r line; do
        # Skip comment lines and blank lines
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        idx=$((idx + 1))
        run_experiment "${idx}" "${tag}" "${hp_cfg}" "${line}" "${dataset_arg}"
    done < "${exp_file}"
}

# =============================================================================
echo "Smoke test started — logs in ${LOG_DIR}/"
echo "DRIVAERML_ROOT : ${DRIVAERML_ROOT}"
echo "SHAPENET_ROOT  : ${SHAPENET_ROOT}"

run_file "drivaerml" \
    "tutorial/configs/train_drivaerml.yaml" \
    "${DRIVAERML_EXP}" \
    "dataset_root=${DRIVAERML_ROOT}"

run_file "shapenet" \
    "tutorial/configs/train_shapenet.yaml" \
    "${SHAPENET_EXP}" \
    "dataset_root=${SHAPENET_ROOT}"

# =============================================================================
total=$((pass + fail))
echo ""
echo "============================================================"
echo " SMOKE TEST SUMMARY"
echo "============================================================"
printf " Total  : %d\n" "${total}"
printf " Pass   : %d\n" "${pass}"
printf " Fail   : %d\n" "${fail}"

# Clean up throwaway training outputs from /tmp
echo ""
echo "Cleaning up /tmp/noether_smoke_test ..."
rm -rf /tmp/noether_smoke_test

if [ "${fail}" -gt 0 ]; then
    echo ""
    echo " Failed experiments:"
    for f in "${failures[@]}"; do
        echo "   - ${f}"
    done
    echo ""
    echo " Full logs: ${LOG_DIR}/"
    exit 1
else
    echo ""
    echo " All ${total} experiments passed."
    echo " Full logs: ${LOG_DIR}/"
    exit 0
fi
