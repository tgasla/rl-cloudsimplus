#!/bin/bash
#SBATCH --job-name=rl-cloudsimplus-experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

#SBATCH --array=1

#SBATCH --output=experiments/euromlsys_extension/experiment_%a/logs/slurm.out
#SBATCH --error=experiments/euromlsys_extension/experiment_%a/logs/slurm.err

module load apptainer

##################################
# Paths
##################################
PROJECT_DIR=/home/people/$USER/rl-cloudsimplus
SCRATCH_DIR=/scratch/$USER

EXP_DIR=$SCRATCH_DIR/experiments/experiment_${SLURM_ARRAY_TASK_ID}

CONFIG_FILE=$EXP_DIR/config.yml
LOG_DIR=$EXP_DIR/logs

GATEWAY_IMAGE=$SCRATCH_DIR/containers/gateway.sif
MANAGER_IMAGE=$SCRATCH_DIR/containers/manager.sif

mkdir -p "$LOG_DIR"

##################################
# Environment Variables
##################################
export EXPERIMENT_ID=1
export RUN_MODE=serial
export NUM_EXPERIMENTS=1

echo "Running experiment $SLURM_ARRAY_TASK_ID on node $SLURMD_NODENAME"

##################################
# Start gateway
##################################
apptainer exec \
  --bind "$LOG_DIR:/app/logs" \
  "$GATEWAY_IMAGE" \
  java -Xms1G -Xmx8G -cp /app/resources:/app/classes:/app/libs/* daislab.cspg.Main \
  > "$LOG_DIR/gateway.out" 2> "$LOG_DIR/gateway.err" &

GATEWAY_PID=$!
sleep 5 # Give the gateway some time to start

##################################
# Start manager (blocking)
##################################
apptainer exec \
  --nv \
  --bind "$CONFIG_FILE:/mgr/config.yml" \
  --bind "$PROJECT_DIR/datacenters_a.yml:/mgr/datacenters_a.yml" \
  --bind "$PROJECT_DIR/datacenters_b.yml:/mgr/datacenters_b.yml" \
  --bind "$PROJECT_DIR/datacenters_c.yml:/mgr/datacenters_c.yml" \
  --bind "$PROJECT_DIR/rl-manager/mnt:/mgr/mnt" \
  --bind "$PROJECT_DIR/rl-manager/gym_cloudsimplus:/mgr/gym_cloudsimplus" \
  --bind "$LOG_DIR:/mgr/logs" \
  --pwd /mgr \
  "$MANAGER_IMAGE" \
  python /mgr/mnt/entrypoint.py \
  > "$LOG_DIR/manager.out" 2>"$LOG_DIR/manager.err"

MANAGER_RC=$?

##################################
# Cleanup
##################################
echo "Experiment $EXPERIMENT_ID finished, stopping gateway"

kill "$GATEWAY_PID"
wait "$GATEWAY_PID" 2>/dev/null

exit "$MANAGER_RC"
