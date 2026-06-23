#!/usr/bin/env bash
# =============================================================================
# update_leaderboard.sh – Full leaderboard update pipeline
#
# For each team listed in teams.yml:
#   1. Clone or pull the team's repository
#   2. Build a Docker image
#   3. Run search.py for all supported (dataset, task) pairs
#   4. Regenerate summary.parquet from all team results
#   5. Re-render the Quarto website
#
# Designed to be run manually (milestone evaluation) or on a cron / CI schedule.
#
# Usage:
#   ./update_leaderboard.sh                          # evaluate all teams
#   ./update_leaderboard.sh --team sisap26-python-baseline   # one team only
#   ./update_leaderboard.sh --skip-eval              # only regenerate parquet + website
#   ./update_leaderboard.sh --dry-run                # print what would run
#
# Options:
#   --team  NAME   Evaluate only this team (matches 'dir' field in teams.yml)
#   --skip-eval    Skip Docker evaluation; re-export and re-render only
#   --dry-run      Print commands without executing
#   --data  DIR    Data directory (default: ./data)
#   --cpus  N      CPUs per container (default: 8)
#   --memory G     Memory in GiB per container (default: 16)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAMS_FILE="$SCRIPT_DIR/teams.yml"
RESULTS_ROOT="$SCRIPT_DIR/team-results"
DATA_DIR="$SCRIPT_DIR/data"
ONLY_TEAM=""
SKIP_EVAL=0
DRY_RUN=0
CPUS=8
MEMORY_GB=16

PASS="[PASS]"
FAIL="[FAIL]"
STEP="[----]"
INFO="[INFO]"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --team)      ONLY_TEAM="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=1;    shift ;;
        --dry-run)   DRY_RUN=1;      shift ;;
        --data)      DATA_DIR="$2";  shift 2 ;;
        --cpus)      CPUS="$2";      shift 2 ;;
        --memory)    MEMORY_GB="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

step()  { echo ""; echo "$STEP $*"; }
ok()    { echo "$PASS $*"; }
fail()  { echo "$FAIL $*"; FAILURES=$((FAILURES + 1)); }
info()  { echo "$INFO $*"; }
run()   { if [[ $DRY_RUN -eq 1 ]]; then echo "  [DRY] $*"; else "$@"; fi; }

FAILURES=0
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "============================================================"
echo "  SISAP 2026 Leaderboard Update"
echo "  $TIMESTAMP"
echo "============================================================"
echo "  Teams file : $TEAMS_FILE"
echo "  Results    : $RESULTS_ROOT"
echo "  Data       : $DATA_DIR"
echo "  Skip eval  : $SKIP_EVAL"
echo "  Dry run    : $DRY_RUN"
[[ -n "$ONLY_TEAM" ]] && echo "  Only team  : $ONLY_TEAM"
echo "============================================================"

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
step "Checking dependencies"

for cmd in python3 docker yq; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd found"
    else
        if [[ "$cmd" == "yq" ]]; then
            # yq is optional — we can parse teams.yml with Python
            info "yq not found; will use Python to parse teams.yml"
        else
            fail "$cmd not found — please install it"
        fi
    fi
done

if [[ ! -f "$TEAMS_FILE" ]]; then
    fail "teams.yml not found: $TEAMS_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 – Evaluate each team
# ---------------------------------------------------------------------------
if [[ $SKIP_EVAL -eq 0 ]]; then
    step "Evaluating teams"

    # Iterate over teams using Python to avoid jq/yq dependency
    while IFS=$'\t' read -r dir repo; do
        [[ -z "$dir" ]] && continue
        [[ -n "$ONLY_TEAM" && "$dir" != "$ONLY_TEAM" ]] && continue

        echo ""
        echo "--- Team: $dir ---"

        EVAL_ARGS=(
            --team    "$dir"
            --repo    "$repo"
            --data    "$DATA_DIR"
            --results "$RESULTS_ROOT"
            --cpus    "$CPUS"
            --memory  "$MEMORY_GB"
        )
        [[ $DRY_RUN -eq 1 ]] && EVAL_ARGS+=(--dry-run)

        if run "$SCRIPT_DIR/evaluate_team.sh" "${EVAL_ARGS[@]}"; then
            ok "Team $dir evaluated successfully"
        else
            fail "Team $dir evaluation failed (exit $?)"
            # Continue with remaining teams; don't abort the whole run
        fi

    done < <(TEAMS_FILE="$TEAMS_FILE" python3 - <<'EOF'
import yaml, os
with open(os.environ["TEAMS_FILE"]) as f:
    data = yaml.safe_load(f)
for t in data.get("teams", []):
    print(t.get("dir","") + "\t" + t.get("repo",""))
EOF
)

else
    info "Skipping evaluation (--skip-eval)"
fi

# ---------------------------------------------------------------------------
# Step 2 – Regenerate summary.parquet
# ---------------------------------------------------------------------------
step "Regenerating summary.parquet"

PARQUET_ARGS=(
    --results-root "$RESULTS_ROOT"
    --output       "$SCRIPT_DIR/website/results/summary.parquet"
)

if run python3 "$SCRIPT_DIR/export_results.py" "${PARQUET_ARGS[@]}"; then
    ok "summary.parquet updated"
else
    fail "export_results.py failed"
    # Don't proceed to website render if parquet is broken
    echo ""
    echo "============================================================"
    echo "$FAIL $FAILURES error(s). Aborting before website render."
    echo "============================================================"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 – Re-render the Quarto website
# ---------------------------------------------------------------------------
step "Re-rendering website"

if run "$SCRIPT_DIR/render_website.sh"; then
    ok "Website rendered successfully"
    echo ""
    echo "  Output: $SCRIPT_DIR/website/_site/"
else
    fail "render_website.sh failed"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
FINISH=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [[ $FAILURES -eq 0 ]]; then
    echo "$PASS Leaderboard update complete. ($FINISH)"
else
    echo "$FAIL $FAILURES error(s) encountered. ($FINISH)"
fi
echo "============================================================"
exit $FAILURES
