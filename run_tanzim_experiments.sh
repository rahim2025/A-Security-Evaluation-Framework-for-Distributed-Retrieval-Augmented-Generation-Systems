#!/usr/bin/env bash
# run_tanzim_experiments.sh
#
# One script, run from a clean checkout, that:
#   1. Starts the full Docker stack (Hardhat node, 3 data sources, LLM service)
#      and waits for every container to report healthy.
#   2. Runs the Membership Inference Attack (MIA), Selective Forwarding
#      Attack (SFA), and DoS attack -- each attack-only, then each attack's
#      defense evaluation -- against the live stack.
#   3. Copies each run's freshly-created log file into a personal, easy-to-
#      browse folder tree under attack_logs/tanzim/<attack>/{attack,defense}/,
#      without touching or duplicating the project's own default log
#      locations (attack_logs/mia/, defense_logs/, attack_logs/
#      selective_forward_sim/, defense_logs/sfa_sim_defense/,
#      attack_logs/ddos_sim/, defense_logs/ddos_sim_defense/ -- all untouched).
#
# Usage
# -----
#   ./run_tanzim_experiments.sh                # seed 42, all three attacks
#   SEED=0 ./run_tanzim_experiments.sh          # a different seed
#   SKIP_DOCKER=1 ./run_tanzim_experiments.sh   # stack already up, skip step 1
#
# See command.txt for the same steps written out individually, in case you
# want to run (or re-run) just one attack/defense pair by hand instead of
# the whole sequence.
#
# Nothing here modifies any attack/defense source file -- this only
# orchestrates the existing entry-point scripts and organizes their output.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
# Thesis rule (.claude/CLAUDE.md): never report single-seed results. Default runs
# seeds 0/42/123. SEED=<n> still works for a quick single-seed smoke test.
SEEDS="${SEEDS:-${SEED:-0 42 123}}"
MIA_DATASET="${MIA_DATASET:-pubmedqa}"   # pubmedqa | healthcaremagic (see data/build_healthcaremagic_corpus.py)
MODE="${MODE:-live}"                 # SFA/DoS support mock|live; MIA is always live
SKIP_DOCKER="${SKIP_DOCKER:-0}"
TANZIM_LOG_ROOT="attack_logs/tanzim"

log()  { printf '\n==> %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }

# ---------------------------------------------------------------------------
# Copy only the files a command just created (not everything already sitting
# in the source directory) into the requested destination -- snapshot the
# source dir before running, diff after, copy the new files only.
# ---------------------------------------------------------------------------
run_step() {
    local desc="$1" src_dir="$2" dest_dir="$3"
    shift 3

    log "$desc"
    mkdir -p "$dest_dir"
    mkdir -p "$src_dir"  # some sources only get created by the command itself

    local before after
    before="$(mktemp)"
    after="$(mktemp)"
    find "$src_dir" -type f 2>/dev/null | sort > "$before"

    "$@"

    find "$src_dir" -type f 2>/dev/null | sort > "$after"
    local copied=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        cp "$f" "$dest_dir/"
        copied=$((copied + 1))
    done < <(comm -13 "$before" "$after")
    rm -f "$before" "$after"

    if [ "$copied" -eq 0 ]; then
        note "no new log file detected in $src_dir -- check the command's output above for an error"
    else
        note "copied $copied new log file(s) -> $dest_dir/"
    fi
}

# ---------------------------------------------------------------------------
# Step 1 -- bring up the stack
# ---------------------------------------------------------------------------
if [ "$SKIP_DOCKER" != "1" ]; then
    log "Starting Docker stack (hardhat-node, 3x data-source, llm-service)"
    docker compose up -d

    note "waiting for every container's healthcheck to pass (up to 3 minutes)..."
    deadline=$((SECONDS + 180))
    while true; do
        unhealthy="$(docker compose ps --format '{{.Name}} {{.Health}}' 2>/dev/null \
            | awk '$2 != "" && $2 != "healthy" { print $1 " -> " $2 }')"
        if [ -z "$unhealthy" ]; then
            note "all containers healthy"
            break
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "  [!] Timed out waiting for containers to become healthy:"
            echo "$unhealthy" | sed 's/^/      /'
            echo "      Check with: docker compose ps ; docker compose logs -f <service>"
            exit 1
        fi
        sleep 5
    done
else
    log "SKIP_DOCKER=1 -- assuming the stack is already up"
fi

echo
echo "  seeds=$SEEDS  mode=$MODE  mia_dataset=$MIA_DATASET  python=$PYTHON"
echo "  git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  logs will be mirrored under: $TANZIM_LOG_ROOT/{mia,sfa,dos}/{attack,defense}/"

for SEED in $SEEDS; do
log "===== SEED $SEED ====="
    # ---------------------------------------------------------------------------
    # Step 2 -- Membership Inference Attack (MIA). Always live -- this module
    # has no mock mode, it queries the real LLM service directly.
    # ---------------------------------------------------------------------------
    run_step "MIA -- attack" \
        "attack_logs/mia" "$TANZIM_LOG_ROOT/mia/attack" \
        "$PYTHON" attack/Mia_attack/run_attack.py --dataset "$MIA_DATASET" --seed "$SEED"

    run_step "MIA -- defense" \
        "defense_logs" "$TANZIM_LOG_ROOT/mia/defense" \
        "$PYTHON" defense/mia_defense/run_defense.py --seed "$SEED"

    # ---------------------------------------------------------------------------
    # Step 3 -- Selective Forwarding Attack (SFA)
    # ---------------------------------------------------------------------------
    run_step "SFA -- attack" \
        "attack_logs/selective_forward_sim" "$TANZIM_LOG_ROOT/sfa/attack" \
        "$PYTHON" attack/selective_forward_sim/run_attack.py --mode "$MODE" --seed "$SEED"

    run_step "SFA -- defense (also runs attack_only + attack_plus_defense internally)" \
        "defense_logs/sfa_sim_defense" "$TANZIM_LOG_ROOT/sfa/defense" \
        "$PYTHON" defense/sfa_sim_defense/run_defense.py --mode "$MODE" --seed "$SEED"

    # ---------------------------------------------------------------------------
    # Step 4 -- DoS attack
    # ---------------------------------------------------------------------------
    run_step "DoS -- attack" \
        "attack_logs/ddos_sim" "$TANZIM_LOG_ROOT/dos/attack" \
        "$PYTHON" attack/ddos_sim/run_attack.py --mode "$MODE" --seed "$SEED"

    run_step "DoS -- defense (also runs attack_only + attack_plus_defense internally)" \
        "defense_logs/ddos_sim_defense" "$TANZIM_LOG_ROOT/dos/defense" \
        "$PYTHON" defense/ddos_sim_defense/run_defense.py --mode "$MODE" --seed "$SEED"


done

log "Done."
note "Logs collected under: $TANZIM_LOG_ROOT/"
note "  mia/attack   mia/defense"
note "  sfa/attack   sfa/defense"
note "  dos/attack   dos/defense"
note "This folder is gitignored -- your run output never gets committed by accident."
