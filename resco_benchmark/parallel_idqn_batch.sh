#!/bin/bash

# Configuration
AGENT="IDQN"
MAP="BB5B"
NETS=("default" "double_conv" "mlp" "gnn")
ACTIVATIONS=("relu" "leaky_relu")
SEEDS=(1 2)
EPS_VAL=5 # Total episodes = EPS_VAL * validation_interval (approx 55)

echo "Starting Batch Parallel Experiment for $AGENT on $MAP"
echo "Each batch will run 4 nets in parallel (one per CPU core)."

for seed in "${SEEDS[@]}"; do
    for activation in "${ACTIVATIONS[@]}"; do
        
        echo "----------------------------------------------------"
        echo "Launching Batch: Activation=$activation, Seed=$seed"
        echo "Nets: ${NETS[*]}"
        echo "----------------------------------------------------"
        
        # Launch the 4 nets in separate tmux sessions
        for net in "${NETS[@]}"; do
            session_name="batch_${net}_${activation}_s${seed}"
            
            # Kill existing session if it exists (cleanup)
            tmux kill-session -t "$session_name" 2>/dev/null
            
            # Create new session
            tmux new-session -d -s "$session_name"
            
            # Set GNN specific config if needed
            if [ "$net" == "gnn" ]; then
                command="export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --agent $AGENT --map $MAP --net $net --activation $activation --seed $seed --eps_val $EPS_VAL"
            else
                command="python3 main.py --agent $AGENT --map $MAP --net $net --activation $activation --seed $seed --eps_val $EPS_VAL"
            fi
            
            tmux send-keys -t "$session_name".0 "$command" ENTER
            echo "  > Started session: $session_name"
        done

        # Monitor Batch Completion
        echo "Waiting for batch to complete..."
        sleep 10 # Give sessions time to initialize
        while true; do
            active_sessions=0
            for net in "${NETS[@]}"; do
                session_name="batch_${net}_${activation}_s${seed}"
                if tmux has-session -t "$session_name" 2>/dev/null; then
                     ((active_sessions++))
                fi
            done
            
            if [ $active_sessions -eq 0 ]; then
                echo "Batch complete!"
                break
            fi
            
            # Show progress
            echo -ne "  Active sessions remaining: $active_sessions \r"
            sleep 60 # Check every minute
        done
    done
done

echo "All experiments complete!"
