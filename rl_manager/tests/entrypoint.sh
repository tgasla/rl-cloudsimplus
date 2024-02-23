#!/bin/bash

python3 tests/a2c.py &
tensorboard --logdir ./a2c_log_cloudsimplus/ --host 0.0.0.0 --port 6006