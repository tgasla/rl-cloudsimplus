#!/bin/bash

CONFIG_FILE="config.yml"

# Detect the correct grep flag based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    GREP_FLAG="-E" # macOS
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GREP_FLAG="-P" # Linux
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Detect the number of replicas
NUM_EXPERIMENTS=$(grep -o "$GREP_FLAG" '^experiment_\d+' "$CONFIG_FILE" | wc -l)

# Set ATTACHED and GPU flags with default values if not provided
ATTACHED=${ATTACHED:-false}
GPU=${GPU:-false}
RUN_MODE=${RUN_MODE:-serial}

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
        SCALE_OPTION="manager-cuda=$NUM_EXPERIMENTS"
        PROFILE_OPTION="--profile cuda"
    else
        SCALE_OPTION="manager=$NUM_EXPERIMENTS"
        PROFILE_OPTION=""
    fi

    # Run the docker compose command based on ATTACHED flag
    if [ "$RUN_MODE" = "batch" ]; then
        RUN_MODE="batch" NUM_EXPERIMENTS="$NUM_EXPERIMENTS" docker compose $PROFILE_OPTION up --scale $SCALE_OPTION --build --remove-orphans -d
        if [ "$ATTACHED" = true ]; then
            echo "Attaching to logs of all containers for batch mode..."
            docker-compose logs -f
        fi

    elif [ "$RUN_MODE" = "serial" ]; then
        for i in $(seq 1 $NUM_EXPERIMENTS); do
            # Start all containers
            RUN_MODE="serial" EXPERIMENT_ID="$i" NUM_EXPERIMENTS="$NUM_EXPERIMENTS" docker compose $PROFILE_OPTION up --build --remove-orphans -d

            # Get the container ID for the manager container
            MANAGER_CONTAINER_ID=$(docker ps --filter "name=manager" --filter "status=running" -q)

            if [ -z "$MANAGER_CONTAINER_ID" ]; then
                echo "Error: No running manager container found for experiment $i."
                exit 1
            fi

            if [ "$ATTACHED" = true ]; then
                # Attach only to the manager container logs
                echo "Attaching to logs of manager container (ID: $MANAGER_CONTAINER_ID) for experiment $i..."
                if [ "$NUM_EXPERIMENTS" -gt 1 ]; then
                    docker logs -f "$MANAGER_CONTAINER_ID" # attach only to rl manager logs
                else
                    docker compose logs -f # attach to all logs
                fi
            fi

            # Wait for the manager container to finish
            echo "Waiting for manager container (ID: $MANAGER_CONTAINER_ID) to finish for experiment $i..."
            # use docker wait if you have done docker ps
            # use docker compose wait if you have done
            # docker compose ps --services | grep manager
            # MANAGER_CONTAINERID=$(docker compose ps --services | grep manager) 
            # docker compose wait "$MANAGER_CONTAINERID"
            docker wait "$MANAGER_CONTAINER_ID"

            # Cleanup after the experiment
            cleanup_experiment
        done
    else
        echo "Invalid RUN_MODE: $RUN_MODE"
    fi

else
    echo "No replicas found in the YAML file."
fi
