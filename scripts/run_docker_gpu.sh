#!/bin/bash

CONFIG_FILE="rl-manager/mnt/config.yml"

NUM_REPLICAS=$(grep -o -P '^exp_\d+' "$CONFIG_FILE" | wc -l); \
ATTACHED=${ATTACHED:-false}

if [ $NUM_REPLICAS -gt 0 ]; then
    if [ "$ATTACHED" = true ]; then
        docker compose --profile cuda up --scale manager-cuda=$NUM_REPLICAS
    else
        docker compose --profile cuda up --scale manager-cuda=$NUM_REPLICAS -d
    fi
else
    echo "No replicas found in the YAML file."
fi