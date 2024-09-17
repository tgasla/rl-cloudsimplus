#!/bin/bash

CONFIG_FILE="rl-manager/mnt/config.yml"

NUM_REPLICAS=$(grep -o -P '^experiment_\d+' "$CONFIG_FILE" | wc -l); \
ATTACHED=${ATTACHED:-false}

if [ $NUM_REPLICAS -gt 0 ]; then
    if [ "$ATTACHED" = true ]; then
        docker compose up --scale manager=$NUM_REPLICAS --build
    else
        docker compose up --scale manager=$NUM_REPLICAS -d --build
    fi
else
    echo "No replicas found in the YAML file."
fi