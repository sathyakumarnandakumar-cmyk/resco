#!/bin/bash

# IMA2C Batch Experiment Script
# Runs 4 architectures in parallel (one per CPU core), all on GPU
# Total episodes per experiment: EPS_VAL * validation_interval = 5 * 11 = 55

AGENT="IMA2C"
MAP="BB5B"
NETS=("mlp" "gru" "lstm" "transformer")
ACTIVATION="relu"
SEED=42
EPS_VAL=5  # Total episodes = 5 * 11 = 55
REWARD="queue_maxwait_neighborhood"
GROUP_TAG="IMA2C_Batch_55eps_GPU"

echo "=========================================="
echo "IMA2C Batch Experiment: $AGENT on $MAP"
echo "Nets: ${NETS[*]}"
echo "Activation: $ACTIVATION"
echo "Seed: $SEED | Episodes: $EPS_VAL * 11 = 55"
echo "Reward: $REWARD"
echo "Group Tag: $GROUP_TAG"
echo "=========================================="

echo ""
echo "----------------------------------------------------"
echo "Launching all 4 architectures in parallel"
echo "----------------------------------------------------"

cd /home/sathya/resco-for-malaysia/resco_benchmark

for net in "${NETS[@]}"; do
    session_name="ima2c_${net}_s${SEED}"

    # Kill existing session if it exists (cleanup)
    tmux kill-session -t "$session_name" 2>/dev/null

    # Create new session
    tmux new-session -d -s "$session_name"

    # GPU-enabled command with deterministic algorithms fix
    command="cd /home/sathya/resco-for-malaysia/resco_benchmark && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --agent $AGENT --map $MAP --net $net --activation $ACTIVATION --seed $SEED --eps_val $EPS_VAL --reward-type $REWARD --group_tag $GROUP_TAG"

    tmux send-keys -t "$session_name".0 "$command" ENTER
    echo "  > Started: $session_name"
    sleep 10  # Stagger launches to allow SUMO port/file initialization
done

echo ""
echo "All 4 IMA2C experiments launched!"
echo ""
echo "Monitor with:"
echo "  tmux ls                           # List active sessions"
echo "  tmux attach -t ima2c_mlp_s42      # Attach to a session"
echo "  tmux kill-session -t <name>       # Kill a session"
echo ""
echo "Sessions:"
for net in "${NETS[@]}"; do
    echo "  - ima2c_${net}_s${SEED}"
done
