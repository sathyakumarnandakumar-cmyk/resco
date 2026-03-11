#!/bin/bash
# MLP Experiment with 50 trials
agents=("IDQN")
activations=("relu")
nets=("mlp")
seeds=(1)

# Loop through each combination of parameters
for agent in "${agents[@]}"; do
    for net in "${nets[@]}"; do
        for activation in "${activations[@]}"; do
            for seed in "${seeds[@]}"; do
                # Define session name
                session_name="${agent}_${net}_${activation}_${seed}_50trials"

                # Start new tmux session
                tmux new-session -d -s $session_name

                # Run command with --trials 50
                tmux send-keys -t "$session_name".0 'python3 main.py --agent '$agent' --map BB5B --net '$net' --activation '$activation' --seed '$seed' --trials 50' ENTER

                echo "Started session $session_name for 50 trials"
            done
        done
    done
done
