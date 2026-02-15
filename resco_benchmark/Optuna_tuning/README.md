# Optuna Hyperparameter Tuning for RESCO

This directory contains scripts for automated hyperparameter tuning using **Optuna** integrated with the RESCO traffic simulation benchmark.

## Core Components

### 1. `main-o.py`
The primary hyperparameter tuning script. It wraps the RESCO training loop and adds support for:
- **Optimization**: Minimizes average vehicle delay (TTI).
- **Search Space**: tunable learning rate, activation functions, neural network backbones (MLP, GNN, Gru, LSTM, Transformer), and reward functions.
- **Pruning**: Uses `Hyperband` to stop poorly performing trials early, saving significant computational time.
- **Persistence**: Results are stored in a local SQLite database (`.db`) allowing interrupted studies to be resumed.
- **Reporting**: Automatically logs trial metrics (Throughput, Delay) to **Neptune.ai** and generates summary plots (`optimization_history.png`, `param_importances.png`) after the study completes.

### 2. Batch Scripts (Parallel Execution)
To speed up tuning, we use bash scripts to launch multiple Optuna workers in parallel. Each worker pulls a trial configuration from the shared SQLite database.

- **`run_optuna_parallel.sh`**:
  - Targets **IDQN** and **IPPO**.
  - Launches 4 parallel workers (configurable).
  - Staggers starts (10s delay) to avoid port collisions in SUMO/TraCI.
- **`run_optuna_ima2c_parallel.sh`**:
  - Specifically configured for the **IMA2C** (PyTorch) agent.
  - Optimized for the BB5B (Malaysia) arterial map.

## Usage

### Single Trial (Manual)
```bash
python main-o.py --agent IDQN --net mlp --n_trials 5 --study_name "my_study"
```

### Parallel Tuning (Recommended)
```bash
chmod +x run_optuna_parallel.sh
./run_optuna_parallel.sh
```

## Key Hyperparameters Tuned
- **Learning Rate**: Log-uniform search [1e-5, 1e-2].
- **Activation**: `relu`, `leaky_relu`, `tanh`.
- **Architectural Backbone**: Toggle between standard MLP and Graph Neural Networks (GNN).
- **Reward Modeling**: Options for `wait`, `pressure`, or `queue_maxwait`.
