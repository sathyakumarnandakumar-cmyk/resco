#!/bin/bash
# MLP and GNN Experiment (55 episodes: eps_val=5 * validation_interval=11)
agents=("IDQN")
activations=("relu")
nets=("mlp" "gnn")
seeds=(1)

# Set Deterministic Algorithm Config for GNN
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Loop through each combination of parameters
for agent in "${agents[@]}"; do
    for net in "${nets[@]}"; do
        for activation in "${activations[@]}"; do
            for seed in "${seeds[@]}"; do
                # Define session name
                session_name="${agent}_${net}_${activation}_${seed}_55eps"

                # Start new tmux session
                tmux new-session -d -s $session_name

                # Run command with --eps_val 5 (Total 55 episodes)
                # Removed --trials 50 as it's confusing/ignored
                tmux send-keys -t "$session_name".0 'python3 main.py --agent '$agent' --map BB5B --net '$net' --activation '$activation' --seed '$seed' --eps_val 5' ENTER

                echo "Started session $session_name (Net: $net, Eps: 55)"
            done
        done
    done
done
