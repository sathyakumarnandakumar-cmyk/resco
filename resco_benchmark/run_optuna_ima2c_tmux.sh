#!/bin/bash

# Configuration
N_WORKERS=4
TRIALS_PER_WORKER=10
STUDY_NAME="ima2c_parallel_study"
STORAGE="sqlite:///optuna_resco_ima2c.db"
SESSION_NAME="optuna_ima2c"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed."
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

echo "Starting Tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME"

for i in $(seq 1 $N_WORKERS); do
    if [ "$i" -eq 1 ]; then
        # Use the first window's first pane
        tmux rename-window -t "$SESSION_NAME:0" "Worker-1"
        tmux send-keys -t "$SESSION_NAME:0.0" "export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main-o.py --agent IMA2C --net gru --activation relu --n_trials $TRIALS_PER_WORKER --n_jobs 1 --study_name \"$STUDY_NAME\" --storage \"$STORAGE\" --pruner hyperband" C-m
    else
        # Create new window for each worker
        tmux new-window -t "$SESSION_NAME" -n "Worker-$i"
        # Stagger starts
        DELAY=$(( (i-1) * 15 ))
        tmux send-keys -t "$SESSION_NAME:Worker-$i" "sleep $DELAY && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main-o.py --agent IMA2C --net gru --activation relu --n_trials $TRIALS_PER_WORKER --n_jobs 1 --study_name \"$STUDY_NAME\" --storage \"$STORAGE\" --pruner hyperband" C-m
    fi
done

# Select first window
tmux select-window -t "$SESSION_NAME:0"

echo "=========================================================="
echo "Parallel Optuna Tuning for IMA2C launched in Tmux"
echo "  Session Name: $SESSION_NAME"
echo "  Workers: $N_WORKERS"
echo "  Trials per worker: $TRIALS_PER_WORKER"
echo "=========================================================="
echo "Run 'tmux attach -t $SESSION_NAME' to monitor progress."
