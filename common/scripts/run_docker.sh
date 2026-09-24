#!/bin/bash
exec </dev/null

if [ -z "$DOMAIN" ]; then
    echo "ERROR: DOMAIN is not set. Usage: make run domain=vm-management or make run domain=job-placement"
    exit 1
fi
HOST_DOMAIN_DIR="domain/$DOMAIN"
CONFIG_FILE="$HOST_DOMAIN_DIR/config.yml"

# Export UID and GID for docker build args
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# base_log_dir is the single source of truth for where results land. It drives the
# container's bind mount (docker-compose.yml) AND the host path preflight inspects,
# so the two can never disagree. Queue-level: read once from common:.
BASE_LOG_DIR=$(python3 - "$CONFIG_FILE" <<'PYEOF' || echo logs
import sys, yaml
yaml.add_multi_constructor('!', lambda l, s, n: None, Loader=yaml.Loader)
cfg = yaml.load(open(sys.argv[1]), Loader=yaml.Loader) or {}
print((cfg.get('common') or {}).get('base_log_dir', 'logs'))
PYEOF
)
case "$BASE_LOG_DIR" in
    ""|/*|*..*)
        echo "ERROR: base_log_dir must be a non-empty relative path without '..', got '$BASE_LOG_DIR'"
        exit 1
        ;;
esac
export BASE_LOG_DIR
echo "base_log_dir=$BASE_LOG_DIR (host: common/$BASE_LOG_DIR, container: /mgr/$BASE_LOG_DIR)"

# Read a value from the merged (common + experiment[i-1]) params.
# Booleans are printed as lowercase true/false for bash consumption.
get_experiment_value() {
    local idx=$(( $1 - 1 ))
    local key="$2"
    local default="$3"
    python3 - <<PYEOF
import yaml
yaml.add_constructor('!include', lambda l, n: {}, Loader=yaml.Loader)
cfg = yaml.load(open('$CONFIG_FILE'), Loader=yaml.Loader)
merged = {**cfg.get('common', {}), **cfg.get('experiments', [])[$idx]}
val = merged.get('$key', $default)
print(str(val).lower() if isinstance(val, bool) else val)
PYEOF
}

# Count experiments from the YAML list
NUM_EXPERIMENTS=$(python3 -c "
import yaml
yaml.add_constructor('!include', lambda l, n: {}, Loader=yaml.Loader)
cfg = yaml.load(open('$CONFIG_FILE'), Loader=yaml.Loader)
print(len(cfg.get('experiments', [])))
")

# Host-side preflight: refuses only what provably cannot run, and fails open.
python3 common/scripts/preflight.py "$CONFIG_FILE" "common/$BASE_LOG_DIR" || {
    echo "preflight failed - nothing started."
    exit 1
}

cleanup_experiment() {
    docker compose -f common/docker-compose.yml $PROFILE_OPTION down --remove-orphans
    if [ $? -ne 0 ]; then
        echo "Error stopping containers. Retrying..."
        docker compose -f common/docker-compose.yml $PROFILE_OPTION down --remove-orphans
    fi
    sleep 5
    echo "Cleanup completed for experiment containers."
}

COMPLETED=0
SKIPPED=0
FAILED=0

# Build the image ONCE for the whole queue. Nothing in it changes between experiments
# (all Python is volume-mounted), but `up --build` re-exports the ~6GB image every
# iteration, which cost ~2 minutes per experiment. Both profiles share one image tag.
if [ $NUM_EXPERIMENTS -gt 0 ]; then
    echo "Building manager image once for this queue..."
    COMPOSE_BAKE=true DOMAIN="$DOMAIN" docker compose -f common/docker-compose.yml \
        --profile cpu --profile cuda build || {
        echo "image build failed - nothing started."
        exit 1
    }
fi

if [ $NUM_EXPERIMENTS -gt 0 ]; then
    for i in $(seq 1 $NUM_EXPERIMENTS); do
        GPU=$(get_experiment_value $i gpu False)
        ATTACHED=$(get_experiment_value $i attached False)

        if [ "$GPU" = true ]; then
            PROFILE_OPTION="--profile cuda"
            MANAGER_SERVICE="manager-cuda"
        else
            PROFILE_OPTION="--profile cpu"
            MANAGER_SERVICE="manager"
        fi

        # Start all containers
        COMPOSE_BAKE=true EXPERIMENT_ID="$i" NUM_EXPERIMENTS="$NUM_EXPERIMENTS" DOMAIN="$DOMAIN" \
            docker compose -f common/docker-compose.yml $PROFILE_OPTION up --remove-orphans -d

        # Get the container ID for the manager service
        MANAGER_CONTAINER_ID=$(docker compose -f common/docker-compose.yml $PROFILE_OPTION ps -q "$MANAGER_SERVICE")

        if [ -z "$MANAGER_CONTAINER_ID" ]; then
            echo "Error: No running manager container found for experiment $i."
            exit 1
        fi

        if [ "$ATTACHED" = true ]; then
            echo "Attaching to container logs for experiment $i..."
            docker compose -f common/docker-compose.yml $PROFILE_OPTION logs -f
        else
            echo "Waiting for container $MANAGER_CONTAINER_ID to finish for experiment $i..."
        fi
        EXIT_CODE=$(docker wait "$MANAGER_CONTAINER_ID")

        # cleanup_experiment runs `docker compose down`, which destroys these logs,
        # so anything worth reading has to be pulled out here.
        case "$EXIT_CODE" in
            0)
                echo ">>> experiment $i completed"
                COMPLETED=$((COMPLETED + 1))
                ;;
            3)
                echo ">>> experiment $i SKIPPED (run dir already holds a run) - continuing"
                docker logs --tail 10 "$MANAGER_CONTAINER_ID" 2>&1 | tail -6 || true
                SKIPPED=$((SKIPPED + 1))
                ;;
            *)
                echo ">>> experiment $i FAILED (exit $EXIT_CODE) - last 40 log lines:"
                docker logs --tail 40 "$MANAGER_CONTAINER_ID" 2>&1 || true
                FAILED=$((FAILED + 1))
                ;;
        esac

        cleanup_experiment
    done
    echo ""
    echo "-- queue summary: $COMPLETED completed, $SKIPPED skipped, $FAILED failed --"
else
    echo "No experiments found in the YAML file."
fi
