#!/bin/bash

agents=("IDQN" "IPPO")
activations=("relu" "tanh" "leaky_relu" "swish")
nets=("default" "double_conv")
seeds=(1 2)
for agent in ${agents[@]}; do
    if [ $agent == "IDQN" ]
    then
        for activation in ${activations[@]}; do
            for net in ${nets[@]}; do
                for seed in ${seeds[@]}; do
                    session_name="$agent""-""$activation""-""$net""-""$seed"
                    tmux new-session -d -s $session_name
                    tmux send-keys -t "$session_name".0 'python3 main.py --agent '$agent' --map BB5B --net '$net' --activation '$activation' --seed '$seed'' ENTER
                                done
                        done
                done
    else
        session_name="$agent"
        tmux new-session -d -s $session_name
        tmux send-keys -t "$session_name".0 "python3 main.py --agent "$agent" --map BB5B" ENTER
    fi
done


