# MLflow Experiment Tracking Guide — RESCO BB5B

This document describes the MLflow logging setup for the BB5B traffic signal control experiments using the SUMO simulator.

> **Code location:** All MLflow logging logic lives in [`utils/mlflow_logger.py`](utils/mlflow_logger.py). It is imported by `main.py`, which handles the training loop.

---

## Quick Start

```bash
# Run an experiment
python3 main.py --map BB5B --agent IDQN --eps_val 5 --validation_interval 3 \
  --description "Testing IDQN with default hyperparams"

# Launch the MLflow dashboard (use absolute path to the DB)
mlflow server \
  --backend-store-uri sqlite:////home/sathya/resco-for-malaysia/resco_benchmark/mlflow.db \
  --port 5020

# Then open http://localhost:5020
```

---

## Architecture

```
main.py
  └── from utils.mlflow_logger import init_mlflow_run, end_mlflow_run, log_metrics, log_model_artifact
```

| Function | Purpose |
|---|---|
| `init_mlflow_run(args, env, main_dir)` | Sets up tracking URI, experiment, tags, params, descriptive run name |
| `log_metrics(buf_infos, done, mode)` | Logs episode-end metrics and per-route data |
| `log_model_artifact(path)` | Uploads the best model `.zip` |
| `end_mlflow_run()` | Ends the current MLflow run |

### Backend Storage

All tracking data is stored in a **SQLite database** (`mlflow.db`) in the `resco_benchmark/` directory. An absolute path is used so the DB location is consistent regardless of where the script is launched:

```python
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(main_dir, 'mlflow.db')}")
```

### Experiment & Run Naming

- **Experiment name:** `{agent}-sumo-{map}` (e.g., `IDQN-sumo-BB5B`)
  - Append a custom suffix: `--experiment_suffix "lr-tuning"` → `IDQN-sumo-BB5B_lr-tuning`
- **Run name:** Descriptive, includes key hyperparameters:
  ```
  IDQN-BB5B_net-default_act-relu_rw-queue_maxwait_seed-42_09_03_21_00
  ```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--agent` | str | `STOCHASTIC` | Agent algorithm (IDQN, IPPO, IMA2C, STOCHASTIC, etc.) |
| `--map` | str | `BB5B` | Map/scenario to run |
| `--eps_val` | int | `10` | Number of validation episodes per cycle |
| `--validation_interval` | int | `11` | Train N-1 episodes, validate on Nth |
| `--net` | str | `default` | Network architecture variant |
| `--activation` | str | `relu` | Activation function (relu, leaky_relu) |
| `--negative_slope` | float | `0.01` | Leaky ReLU slope (only if activation=leaky_relu) |
| `--reward-type` | str | `queue_maxwait` | Reward function (wait, pressure, queue_maxwait, etc.) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--validation_day` | str | `26NovFull` | Validation day directory |
| `--validation_period` | str | `7-8am` | Time period for validation |
| `--description` | str | auto | Custom description for the MLflow run |
| `--group_tag` | str | None | Optional tag to group related experiments |
| `--experiment_suffix` | str | None | String appended to experiment name (e.g., `lr-tuning`) |

---

## What Gets Logged

### 1. Parameters

Logged once at the start of each run via `mlflow.log_params`:

| Parameter | Example Value | Description |
|---|---|---|
| `action_frequency` | `10` | Simulation step length in seconds |
| `algorithm` | `IDQN` | Agent algorithm name |
| `number_of_training_episodes` | `20` | Total training episodes |
| `number_of_validation_episodes` | `10` | Total validation episodes |
| `map` | `BB5B` | Scenario map |
| `net` | `default` | Network architecture |
| `reward` | `queue_maxwait` | Reward function used |
| `activation` | `relu` | Neural network activation function |
| `validation_day` | `26NovFull` | Day used for validation routes |
| `validation_period` | `7-8am` | Time period simulated |
| `seed` | `42` | Random seed |
| `phases` | `{'PBB_Junc': 4, ...}` | Number of signal phases per junction |
| `negative_slope` | `0.01` | *(only if activation=leaky_relu)* |

### 2. Tags

Tags allow filtering runs in the MLflow UI:

| Tag Key | Example Value |
|---|---|
| `environment` | `sumo-v0` |
| `agent` | `IDQN` |
| `net` | `default` |
| `activation` | `relu` |
| `framework` | `stable-baselines3` |
| `phase_config` | `4 phases for PBB_Junc and SIRIM_Junc, 3 phases for INFMain_Junc - Full` |
| `traffic_note` | `no new vehicles after 1 hour` |
| `validation_period` | `7-8am` |
| `group_tag` | *(user-specified, optional)* |

### 3. Metrics

Metrics are organized by **mode** (`training` or `validation`). Each includes a step counter for time-series plotting.

> **Note:** Per-step metrics (actions, rewards, vehicle counts at every simulation step) are **commented out** by default to keep the DB lean. They can be re-enabled in `utils/mlflow_logger.py`.

#### Active Episode-End Metrics (logged once per episode)

Key metrics for model selection are marked with ⭐:

| Metric Key | Description |
|---|---|
| `metrics/{mode}/count_of_all_vehicles_in_simulation` | Total vehicles that appeared during the episode |
| ⭐ `metrics/{mode}/count_of_vehicles_completing_journey` | Vehicles that completed their route (higher = better) |
| ⭐ `metrics/{mode}/total_average_delays_of_all_vehicles_from_all_routes` | Average delay index across all completed vehicles (lower = better) |
| `metrics/{mode}/total_sum_delays_of_all_vehicles_from_all_routes` | Sum of all delay indices |
| `metrics/{mode}/average_time_of_journey` | Average travel time per completing vehicle |
| `metrics/{mode}/total_average_delays_with_weights` | Route-weighted average delay index |
| `metrics/{mode}/total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey` | Delay index including in-progress vehicles |

#### Active Per-Route Metrics (logged once at episode end)

For each of the 14 monitored routes, only 2 key metrics are active:

**Monitored Routes:** `Infout-HLin`, `PBBN-FMin`, `PBBN-SirimS`, `PBBN-SirimW`, `PBBN-SKE`, `PBBW-FMin`, `PBBW-SKE`, `SirimE-HLin`, `SirimS-HLin`, `SirimS-PBBN`, `SirimW-HLin`, `SirimW-SirimE`, `SKE-HLin`, `SKE-PBBN`

| Metric Key | Description |
|---|---|
| `.../routes/{route_id}/throughput_of_the_route-ThruPut_Idx` | Throughput index (actual/scheduled) |
| `.../routes/{route_id}/total_average_delays_of_all_vehicles-Delay_Idx_Average` | Average delay index |

#### Commented-Out Metrics (available in `utils/mlflow_logger.py`)

These can be re-enabled by uncommenting in `mlflow_logger.py`:

<details>
<summary>Per-step metrics (actions, rewards, vehicle counts at every step)</summary>

- `action/INFMain_Junc`, `action/PBB_Junc`, `action/SIRIM_Junc`
- `reward/INFMain_Junc`, `reward/PBB_Junc`, `reward/SIRIM_Junc`
- `current_number_of_vehicles`, `number_of_halting_vehicles_...`
- `waiting_time_all_vehicles_...`, `calculate_average_delta_of_delays_after_action`
- `number_of_vehicles_that_passed_through_...`, `current_average_delays_...`
</details>

<details>
<summary>Additional episode-end metrics</summary>

- `total_time_of_journey`, `total_waiting_time_on_the_incoming_lanes_in_episode`
- `total_average_delays_real_times_by_ideal_times`
</details>

<details>
<summary>Additional per-route metrics</summary>

- `length`, `ThruPut_Scheduled`, `ThruPut_Actual`
- `total_travel_time_of_all_vehicles` (sum), `total_average_travel_time`
- `total_delays_of_all_vehicles` (sum), `Delay_Idx_StDev`, weighted delays
</details>

### 4. Text Artifacts (Per-Vehicle Lists)

For each monitored route, **raw per-vehicle lists** are saved as text files:

```
routes/{route_id}/{mode}_total_travel_time_list_step_{step}.txt
routes/{route_id}/{mode}_total_delays_list_step_{step}.txt
```

These contain individual travel times and delay indices, e.g.:
```
[40.0, 49.0, 60.0, 51.0, 47.0, ...]
```

### 5. Model Artifacts

For agents `IDQN`, `IPPO`, `IMA2C`, and `STOCHASTIC`, the **best validation model** is automatically uploaded as a `.zip` file. Find it under the **Artifacts** tab in the MLflow UI.

---

## Understanding the Key Metrics

### Delay Index
The core performance metric — how much slower a vehicle travels vs free-flow:

```
delay_index = (max_speed × actual_travel_time) / route_length
```

- **1.0** = free-flow speed (perfect) · **2.0** = twice as slow · **Lower is better**

### Throughput Index
Fraction of scheduled vehicles that completed their route:

```
throughput_index = vehicles_completing / vehicles_scheduled
```

- **1.0** = all completed · **< 1.0** = some still in network at episode end

### Best Model Selection
1. **Maximize** `count_of_vehicles_completing_journey`
2. **Minimize** `total_average_delays_of_all_vehicles_from_all_routes` (tiebreaker)

---

## Training/Validation Loop

```
validation_interval = N, eps_val = V
Total episodes = V × N
Training: episodes where (i % N != 0)
Validation: episodes where (i % N == 0)
```

Example with `--eps_val 5 --validation_interval 3`: 15 total episodes, pattern: train, train, **validate**, repeat.

---

## Viewing Results

### MLflow Server
```bash
mlflow server \
  --backend-store-uri sqlite:////home/sathya/resco-for-malaysia/resco_benchmark/mlflow.db \
  --port 5020
```

In the UI: compare runs, plot metric trends, filter by tags, download artifacts.

### Programmatic Access
```python
import mlflow

mlflow.set_tracking_uri("sqlite:////home/sathya/resco-for-malaysia/resco_benchmark/mlflow.db")
client = mlflow.tracking.MlflowClient()

# Search runs sorted by delay index
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.`metrics/validation/total_average_delays_of_all_vehicles_from_all_routes` ASC"],
)

# Get metric history for a run
history = client.get_metric_history(run_id="<run_id>", key="metrics/validation/count_of_vehicles_completing_journey")
```
