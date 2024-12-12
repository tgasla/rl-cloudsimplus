#!/bin/bash

CONFIG_FILE="config.yml"

# Detect the number of replicas
NUM_EXPERIMENTS=$(grep -o -P '^experiment_\d+' "$CONFIG_FILE" | wc -l)

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
    docker volume prune -f
    docker network prune -f

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
        if [ "$ATTACHED" = true ]; then
            DETACHED_OPTION=""
        else
            DETACHED_OPTION="-d"
        fi
            docker compose $PROFILE_OPTION up --scale $SCALE_OPTION $DETACHED_OPTION --build --remove-orphans

    elif [ "$RUN_MODE" = "serial" ]; then
        for i in $(seq 1 $NUM_EXPERIMENTS); do
            if [ "$ATTACHED" = true ]; then
                DETACHED_OPTION=""
            else
                DETACHED_OPTION="-d"
            fi
            RUN_MODE="serial" EXPERIMENT_ID="$i" docker compose $PROFILE_OPTION up --build --remove-orphans
            cleanup_experiment
        done
    else
        echo "Invalid RUN_MODE: $RUN_MODE"
    fi

else
    echo "No replicas found in the YAML file."
fi
