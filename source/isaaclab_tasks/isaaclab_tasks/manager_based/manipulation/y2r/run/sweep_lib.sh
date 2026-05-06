# ==============================================================================
# Shared sweep helpers — sourced by ablations_teacher.sh and ablations_student.sh
# ==============================================================================
# Provides:
#   sweep_init <log_subdir>          — set LOG_DIR + STATUS_FILE, mkdir, init header
#   sweep_set_status <name> <status> <exit_code>
#   sweep_is_done <name>             — exit 0 if marked done, else nonzero
#   sweep_print_header <title>       — banner before the run loop starts
#   sweep_print_final                — final "Sweep finished" summary table
#
# Status TSV schema:
#   run_name | status | start_ts | end_ts | exit_code | log_file
#   status ∈ {running, done, failed}
# Each script defines its own run_one() because the underlying command
# (train.sh vs distill.sh) and per-condition arg shape differ.
# ==============================================================================

# All callers should: set -uo pipefail (NOT -e — one bad run shouldn't abort the sweep).

sweep_init() {
    # sweep_init <log_subdir>  — e.g. "ablations" or "ablations_student"
    local subdir="$1"
    LOG_DIR="$Y2R_DATA_ROOT/IsaacLab/logs/$subdir"
    STATUS_FILE="$LOG_DIR/STATUS.tsv"
    mkdir -p "$LOG_DIR"
    if [ ! -f "$STATUS_FILE" ]; then
        printf "run_name\tstatus\tstart_ts\tend_ts\texit_code\tlog_file\n" > "$STATUS_FILE"
    fi
}

sweep_set_status() {
    # sweep_set_status <run_name> <status> <exit_code>
    # Strips any prior rows for this run, then appends the new one. Preserves
    # start_ts when transitioning running→done/failed so wall time is correct.
    local run_name="$1" status="$2" exit_code="$3"
    local ts="$(date -Iseconds)"
    local tmp="$(mktemp)"
    awk -F'\t' -v rn="$run_name" 'NR==1 || $1 != rn' "$STATUS_FILE" > "$tmp"
    if [ "$status" = "running" ]; then
        printf "%s\t%s\t%s\t\t\t%s\n" "$run_name" "$status" "$ts" "$LOG_DIR/$run_name.log" >> "$tmp"
    else
        local start_ts="$(awk -F'\t' -v rn="$run_name" '$1==rn && $2=="running" {print $3}' "$STATUS_FILE" | tail -1)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$status" "${start_ts:-$ts}" "$ts" "$exit_code" "$LOG_DIR/$run_name.log" >> "$tmp"
    fi
    mv "$tmp" "$STATUS_FILE"
}

sweep_is_done() {
    local run_name="$1"
    awk -F'\t' -v rn="$run_name" '$1==rn && $2=="done" {found=1} END {exit !found}' "$STATUS_FILE"
}

sweep_print_header() {
    # sweep_print_header <title>  — relies on caller having CONDITIONS, SEEDS, MAX_ITERATIONS in scope
    local title="$1"
    echo "=================================================================="
    echo "  $title"
    echo "  Conditions: ${CONDITIONS[*]}"
    echo "  Seeds:      ${SEEDS[*]}"
    echo "  Iterations: $MAX_ITERATIONS"
    echo "  Status:     $STATUS_FILE"
    echo "=================================================================="
}

sweep_print_final() {
    echo ""
    echo "=================================================================="
    echo "  Sweep finished. Final status:"
    echo "=================================================================="
    column -t -s $'\t' "$STATUS_FILE"
}
