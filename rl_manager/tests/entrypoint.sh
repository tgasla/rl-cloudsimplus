#!/bin/bash

python3 tests/a2c.py &
tensorboard --logdir ./tb_logs/ --host 0.0.0.0 --port 6006
