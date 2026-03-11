#!/bin/bash

# PPO Batch Experiment Script
# Runs 4 nets in parallel (one per CPU core), then next activation batch
# Total episodes per experiment: EPS_VAL * validation_interval = 5 * 11 = 55

AGENT="IPPO"
MAP="BB5B"
NETS=("default" "double_conv" "mlp" "gnn")
ACTIVATIONS=("relu" "leaky_relu")
SEED=1
EPS_VAL=5  # Total episodes = 5 * 11 = 55
GROUP_TAG="PPO_Batch_55eps_GPU"

echo "=========================================="
echo "PPO Batch Experiment: $AGENT on $MAP"
echo "Nets: ${NETS[*]}"
echo "Activations: ${ACTIVATIONS[*]}"
echo "Seed: $SEED | Episodes: $EPS_VAL * 11 = 55"
echo "Group Tag: $GROUP_TAG"
echo "=========================================="

for activation in "${ACTIVATIONS[@]}"; do

    echo ""
    echo "----------------------------------------------------"
    echo "Launching Batch: Activation=$activation, Seed=$SEED"
    echo "----------------------------------------------------"

    # Launch the 4 nets in separate tmux sessions (CPU only to avoid CuBLAS errors)
    for net in "${NETS[@]}"; do
        session_name="ppo_${net}_${activation}_s${SEED}"

        # Kill existing session if it exists (cleanup)
        tmux kill-session -t "$session_name" 2>/dev/null

        # Create new session
        tmux new-session -d -s "$session_name"

        # Enable GPU with deterministic algorithms fix
        # CUBLAS_WORKSPACE_CONFIG is required when torch.use_deterministic_algorithms(True) is set
        command="export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --agent $AGENT --map $MAP --net $net --activation $activation --seed $SEED --eps_val $EPS_VAL --group_tag $GROUP_TAG"

        tmux send-keys -t "$session_name".0 "$command" ENTER
        echo "  > Started: $session_name"
        sleep 10  # Stagger launches to allow SUMO port/file initialization
    done

    # Wait for batch to initialize (SUMO can be slow to start)
    echo "Waiting for batch to initialize..."
    sleep 60  # Give sessions time to stabilize

    while true; do
        active_sessions=0
        for net in "${NETS[@]}"; do
            session_name="ppo_${net}_${activation}_s${SEED}"
            if tmux has-session -t "$session_name" 2>/dev/null; then
                # Check if session is alive
                ((active_sessions++))
            else
                 # Session finished
                 :
            fi
        done

        if [ "$active_sessions" -eq 0 ]; then
            echo "Batch complete!"
            break
        fi

        echo "  Active sessions remaining: $active_sessions"
        sleep 60
    done

done

echo ""
echo "=========================================="
echo "All PPO experiments complete!"
echo "=========================================="
