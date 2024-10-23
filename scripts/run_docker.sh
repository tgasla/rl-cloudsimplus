#!/bin/bash

CONFIG_FILE="config.yml"

# Detect the number of replicas
NUM_REPLICAS=$(grep -o -P '^experiment_\d+' "$CONFIG_FILE" | wc -l)

# Set ATTACHED and GPU flags with default values if not provided
ATTACHED=${ATTACHED:-false}
GPU=${GPU:-false}

# Check if there are replicas
if [ $NUM_REPLICAS -gt 0 ]; then
    # Determine the Docker command based on the GPU flag
    if [ "$GPU" = true ]; then
        SCALE_OPTION="manager-cuda=$NUM_REPLICAS"
        PROFILE_OPTION="--profile cuda"
    else
        SCALE_OPTION="manager=$NUM_REPLICAS"
        PROFILE_OPTION=""
    fi
    
    # Run the docker compose command based on ATTACHED flag
    if [ "$ATTACHED" = true ]; then
        DETACHED_OPTION=""
    else
        DETACHED_OPTION="-d"
    fi
        docker compose $PROFILE_OPTION up --scale $SCALE_OPTION $DETACHED_OPTION --build --remove-orphans
else
    echo "No replicas found in the YAML file."
fi
