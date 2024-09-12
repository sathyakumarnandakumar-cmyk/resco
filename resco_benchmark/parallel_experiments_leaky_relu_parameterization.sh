#!/bin/bash

command -v tmux >/dev/null 2>&1 || { echo >&2 "'tmux' is not available. Install and try again."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo >&2 "'python3' is not available. Install and try again."; exit 1; }

agents=("IDQN" "IPPO")
activations=("relu" "leaky_relu" "tanh" "swish")
nets=("default" "double_conv")
eps_val=10
leaky_relu_negative_slopes=()
seeds=()

# Generate negative slopes for leaky_relu activation function
negative_slope_start=0.005
negative_slope_end=0.03
negative_slope_step=0.005
while (( $(bc <<< "$negative_slope_start <= $negative_slope_end") )); do
    leaky_relu_negative_slopes+=("$negative_slope_start")
    negative_slope_start=$(bc <<< "$negative_slope_start + $negative_slope_step")
done

# Generate seed values
seed_start=1
seed_end=10
seed_step=1
for ((seed=seed_start; seed<=seed_end; seed+=seed_step)); do
    seeds+=("$seed")
done

for agent in "${agents[@]}"; do
    for seed in "${seeds[@]}"; do
        for net in "${nets[@]}"; do
            for activation in "${activations[@]}"; do
                if [ "$activation" = "leaky_relu" ]; then
                    for negative_slope in "${leaky_relu_negative_slopes[@]}"; do
                        negative_slope=$(echo "$negative_slope" | sed 's/\.//g')
                        session_name="${agent}-${seed}-${net}-${activation}-${negative_slope}"
                        tmux new-session -d -s "${session_name}"
                        tmux send-keys -t "${session_name}".0 "python3 main.py --agent ${agent} --net ${net} --activation ${activation} --negative_slope ${negative_slope} --map BB5B --seed ${seed} --eps_val ${eps_val}" ENTER
                    done
                else
                    session_name="${agent}-${seed}-${net}-${activation}"
                    tmux new-session -d -s "${session_name}"
                    tmux send-keys -t "${session_name}".0 "python3 main.py --agent ${agent} --net ${net} --activation ${activation} --map BB5B --seed ${seed} --eps_val ${eps_val}" ENTER
                fi
            done
        done
    done
done
