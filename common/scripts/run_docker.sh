#!/bin/bash
exec </dev/null

CONFIG_FILE="config.yml"

# Export UID and GID for docker build args
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Detect the correct grep flag based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    GREP_FLAG="-E" # macOS
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GREP_FLAG="-P" # Linux
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Read a value from the globals section of config.yml
get_yaml_value() {
    grep -A 20 '^globals:' "$CONFIG_FILE" | grep -m 1 "^ *$1:" | sed 's/^ *//' | sed 's/.*: //'
}

# Detect the number of replicas
NUM_EXPERIMENTS=$(grep -o "$GREP_FLAG" '^experiment_\d+' "$CONFIG_FILE" | wc -l)

# Set ATTACHED and GPU flags with default values if not provided
ATTACHED=${ATTACHED:-false}
GPU=${GPU:-false}

# Read java log destination from config.yml (not overridden by ATTACHED)
JAVA_LOG_DEST=$(get_yaml_value "java_log_destination")
JAVA_LOG_LEVEL=$(get_yaml_value "java_log_level")

cleanup_experiment() {
    # Stop the experiment containers
    docker compose down --remove-orphans
    if [ $? -ne 0 ]; then
        echo "Error stopping containers. Retrying..."
        docker compose down --remove-orphans
    fi

    # Remove volumes and networks
    # docker volume prune -f
    # docker network prune -f
    # docker system prune --volumes -f

    # Sleep for a short duration to ensure resources are released
    sleep 5

    echo "Cleanup completed for experiment containers."
}

# Check if there are replicas
if [ $NUM_EXPERIMENTS -gt 0 ]; then
    # Determine the Docker command based on the GPU flag
    if [ "$GPU" = true ]; then
        PROFILE_OPTION="--profile cuda"
        MANAGER_SERVICE="manager-cuda"
    else
        PROFILE_OPTION=""
        MANAGER_SERVICE="manager"
    fi

    for i in $(seq 1 $NUM_EXPERIMENTS); do
        # Start all containers
        EXPERIMENT_ID="$i" NUM_EXPERIMENTS="$NUM_EXPERIMENTS" JAVA_LOG_DESTINATION="$JAVA_LOG_DEST" JAVA_LOG_LEVEL="$JAVA_LOG_LEVEL" \
            docker compose $PROFILE_OPTION up --build --remove-orphans -d

        # Get the container ID for the manager service
        MANAGER_CONTAINER_ID=$(docker compose ps -q "$MANAGER_SERVICE")

        if [ -z "$MANAGER_CONTAINER_ID" ]; then
            echo "Error: No running manager container found for experiment $i."
            exit 1
        fi

        if [ "$ATTACHED" = true ]; then
            if [ "$NUM_EXPERIMENTS" -gt 1 ]; then
                # Attach only to the manager container logs
                echo "Attaching to logs of manager container for experiment $i..."
                docker compose logs -f "$MANAGER_SERVICE"
            else
                # Attach to all container logs
                echo "Attaching to all container logs for experiment $i..."
                docker compose logs -f
            fi
        else
            # Wait for the manager container to finish (using docker wait with container ID)
            echo "Waiting for container $MANAGER_CONTAINER_ID to finish for experiment $i..."
            docker wait "$MANAGER_CONTAINER_ID"
        fi

        # Cleanup after the experiment
        cleanup_experiment
    done

else
    echo "No replicas found in the YAML file."
fi
