#!/bin/bash
# Wait for all 4 IMA2C sessions to finish, then launch Optuna

SESSIONS=("ima2c_mlp_s42" "ima2c_gru_s42" "ima2c_lstm_s42" "ima2c_transformer_s42")

echo "Monitoring IMA2C sessions... $(date)"

while true; do
    active=0
    for s in "${SESSIONS[@]}"; do
        if tmux has-session -t "$s" 2>/dev/null; then
            # Check if session still has a running process (not just a shell prompt)
            pane_pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
            if [ -n "$pane_pid" ]; then
                children=$(pgrep -P "$pane_pid" 2>/dev/null | wc -l)
                if [ "$children" -gt 0 ]; then
                    ((active++))
                else
                    echo "  Session $s finished (idle shell). Killing... $(date)"
                    tmux kill-session -t "$s" 2>/dev/null
                fi
            fi
        fi
    done

    if [ "$active" -eq 0 ]; then
        echo ""
        echo "All IMA2C sessions complete! $(date)"
        break
    fi

    echo "  Active: $active / ${#SESSIONS[@]} — $(date)"
    sleep 120
done

echo ""
echo "Launching Optuna IDQN tuning... $(date)"
cd /home/sathya/resco-for-malaysia/resco_benchmark
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 main-o.py --agent IDQN --net mlp --activation relu \
    --n_trials 20 --study_name idqn_mlp_relu_optuna_tuning
echo ""
echo "Optuna tuning complete! $(date)"
