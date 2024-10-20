#!/bin/bash

CONFIG_FILE="config.yml"

NUM_REPLICAS=$(grep -o -P '^experiment_\d+' "$CONFIG_FILE" | wc -l); \
ATTACHED=${ATTACHED:-false}

if [ $NUM_REPLICAS -gt 0 ]; then
    if [ "$ATTACHED" = true ]; then
        docker compose up --scale manager=$NUM_REPLICAS --build --remove-orphans
    else
        docker compose up --scale manager=$NUM_REPLICAS -d --build --remove-orphans
    fi
else
    echo "No replicas found in the YAML file."
fi