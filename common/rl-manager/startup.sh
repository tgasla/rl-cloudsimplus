#!/bin/bash
# Re-install gym_cloudsimplus from the volume-mounted source at runtime
# This ensures any local changes to the source are picked up immediately without rebuild

GYM_DIR="/mgr/gym_cloudsimplus"
if [ -d "$GYM_DIR" ]; then
    echo "Installing gym_cloudsimplus from volume-mounted source: $GYM_DIR"
    (cd "$GYM_DIR" && pip install --no-deps . 2>&1 | tail -5)
    echo "gym_cloudsimplus installed/updated."
fi

export PYTHONPATH="/mgr/gym_cloudsimplus/gym_cloudsimplus:$PYTHONPATH"
exec python entrypoint.py
