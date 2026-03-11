#!/bin/bash

# Configuration
AGENT="IDQN"
MAP="BB5B"
NETS=("default" "double_conv" "mlp" "gnn")
ACTIVATION="leaky_relu"
SEED=1
EPS_VAL=5  # Total episodes = EPS_VAL * validation_interval (11) = 55

echo "Starting Manual Batch 2 (Leaky ReLU, Seed 1)"

for net in "${NETS[@]}"; do
    session_name="batch_${net}_${ACTIVATION}_s${SEED}"
    
    # Kill existing session if it exists (cleanup)
    tmux kill-session -t "$session_name" 2>/dev/null
    
    # Create new session
    tmux new-session -d -s "$session_name"
    
    # Set GNN specific config if needed
    if [ "$net" == "gnn" ]; then
        command="export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --agent $AGENT --map $MAP --net $net --activation $ACTIVATION --seed $SEED --eps_val $EPS_VAL"
    else
        command="python3 main.py --agent $AGENT --map $MAP --net $net --activation $ACTIVATION --seed $SEED --eps_val $EPS_VAL"
    fi
    
    tmux send-keys -t "$session_name".0 "$command" ENTER
    echo "  > Started session: $session_name"
done

echo "Batch 2 Launched!"
