#!/bin/bash
# Loop through parameters
agents=("IDQN")
activations=("relu" "leaky_relu")
nets=("mlp" "gnn")
seeds=(1 2)

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Loop through each combination of parameters
for agent in "${agents[@]}"; do
    if [ "$agent" == "IDQN" ]; then
        for net in "${nets[@]}"; do
            for activation in "${activations[@]}"; do
                for seed in "${seeds[@]}"; do
                    # Define session name based on parameters
                    session_name="${agent}_${net}_${activation}_${seed}"

                    # Start new tmux session in detached mode
                    tmux new-session -d -s $session_name

                    # Send the python command to the tmux session
                    # Adding "ENTER" at the end simulates pressing the Enter key to run the command
                    tmux send-keys -t "$session_name".0 'python3 main.py --agent '$agent' --map BB5B --net '$net' --activation '$activation' --seed '$seed'' ENTER

                    echo "Started session $session_name with agent $agent, net $net, activation $activation, seed $seed"
                done
            done
        done
    elif [ "$agent" == "IPPO" ]; then
         # Similar loop for IPPO or just run single config
         tmux new-session -d -s IPPO_test
         tmux send-keys -t IPPO_test.0 'python3 main.py --agent IPPO --map BB5B' ENTER
         echo "Started IPPO session"
    fi
done
