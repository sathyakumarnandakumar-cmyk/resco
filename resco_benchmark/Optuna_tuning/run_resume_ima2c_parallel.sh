#!/bin/bash

# Configuration
N_WORKERS=4
TRIALS_PER_WORKER=10
STUDY_NAME="ima2c_parallel_study"
STORAGE="sqlite:///optuna_resco_ima2c.db"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=========================================================="
echo "Starting Parallel Optuna Tuning for IMA2C"
echo "  Workers: $N_WORKERS"
echo "  Trials per worker: $TRIALS_PER_WORKER"
echo "  Total expected trials: $((N_WORKERS * TRIALS_PER_WORKER))"
echo "  Study Name: $STUDY_NAME"
echo "=========================================================="

# Ensure the DB storage is initialized by letting the first worker handle it or just launch.
# Optuna's load_if_exists=True handles this.

pids=""

for i in $(seq 1 $N_WORKERS); do
    echo "[Worker $i] Starting..."
    
    # Stagger starts by 15 seconds to avoid SUMO/TraCI port collision race conditions
    python3 main-o.py \
        --agent IMA2C \
        --net gru \
        --activation relu \
        --n_trials $TRIALS_PER_WORKER \
        --n_jobs 1 \
        --study_name "$STUDY_NAME" \
        --storage "$STORAGE" \
        --pruner hyperband &
    
    pid=$!
    pids="$pids $pid"
    echo "[Worker $i] Launched with PID $pid. Sleeping 15s before next launch..."
    sleep 15
done

echo "All workers launched. Waiting for completion..."
wait $pids
echo "All trials complete."

# Run one final time with 0 trials just to trigger the plotting and summary
echo "Generating final plots..."
python3 main-o.py \
    --agent IMA2C \
    --net gru \
    --activation relu \
    --n_trials 0 \
    --study_name "$STUDY_NAME" \
    --storage "$STORAGE"

echo "Done!"
