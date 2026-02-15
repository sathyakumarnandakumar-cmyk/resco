#!/bin/bash

# Configuration
N_WORKERS=4
TRIALS_PER_WORKER=5
STUDY_NAME="idqn_parallel_study"
STORAGE="sqlite:///optuna_resco.db"

echo "=========================================================="
echo "Starting Parallel Optuna Tuning"
echo "  Workers: $N_WORKERS"
echo "  Trials per worker: $TRIALS_PER_WORKER"
echo "  Total expected trials: $((N_WORKERS * TRIALS_PER_WORKER))"
echo "  Study Name: $STUDY_NAME"
echo "=========================================================="

# Create the study first (to ensure DB exists) by running a quick check or just letting the first worker create it
# We'll just launch them. loading_if_exists=True handles creation.

pids=""

for i in $(seq 1 $N_WORKERS); do
    echo "[Worker $i] Starting..."
    
    # We use & to run in background
    # We stagger starts by 10 seconds to avoid SUMO/TraCI port collision race conditions
    python main-o.py \
        --agent IDQN \
        --net mlp \
        --activation relu \
        --n_trials $TRIALS_PER_WORKER \
        --n_jobs 1 \
        --study_name "$STUDY_NAME" \
        --storage "$STORAGE" \
        --pruner hyperband &
    
    pid=$!
    pids="$pids $pid"
    echo "[Worker $i] Launched with PID $pid. Sleeping 10s before next launch..."
    sleep 10
done

echo "All workers launched. Waiting for completion..."
wait $pids
echo "All trials complete."

# Run one final time with 0 trials just to trigger the plotting and summary
echo "Generating final plots..."
python main-o.py \
    --agent IDQN \
    --net mlp \
    --activation relu \
    --n_trials 0 \
    --study_name "$STUDY_NAME" \
    --storage "$STORAGE"

echo "Done!"
