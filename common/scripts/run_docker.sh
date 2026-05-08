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

cleanup_experiment() {
    docker compose -f common/docker-compose.yml $PROFILE_OPTION down --remove-orphans
    if [ $? -ne 0 ]; then
        echo "Error stopping containers. Retrying..."
        docker compose -f common/docker-compose.yml $PROFILE_OPTION down --remove-orphans
    fi
    sleep 5
    echo "Cleanup completed for experiment containers."
}

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
            docker compose -f common/docker-compose.yml $PROFILE_OPTION up --build --remove-orphans -d

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
            docker wait "$MANAGER_CONTAINER_ID"
        fi

        cleanup_experiment
    done
else
    echo "No experiments found in the YAML file."
fi
