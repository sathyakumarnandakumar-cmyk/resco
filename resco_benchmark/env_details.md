# BB5B Environment — Complete Technical Reference

This document covers the BB5B traffic signal control environment: its physical layout, RL formulation (states, actions, rewards), data flow, file responsibilities, and logging outputs.

---

## 1. Physical Environment

BB5B is a real-world intersection network in Malaysia with **3 signalized junctions**:

| Junction | Phases | Incoming Road Groups |
|---|---|---|
| **INFMain_Junc** | 3 green phases | N1 (FMS), N2 (SHS), E (SKW), S1 (INFN), S2 (TCN) |
| **PBB_Junc** | 4 green phases | N (TMS), E (SHW), S1 (SHN/HLout), S2 (FMN), W (HLE) |
| **SIRIM_Junc** | 4 green phases | N1 (TCS/TXout), N2 (INFS/SKWS), E (SIRIMW), S (SIRIMN), W (TCE) |

### Simulation Parameters (from `config/map_config.py`)

| Parameter | Value |
|---|---|
| Network file | `environments/BB5B/BB5B.sumocfg` |
| Step length | **10 seconds** (each RL action step) |
| Yellow phase | **3 seconds** |
| Red phase | **2 seconds** |
| Default time window | 7:00 AM – 8:00 AM (25200s – 28800s) |
| Extra time after end | **+240 seconds** (4 min for vehicles to finish routes) |
| Steps per episode | ~**384** steps (3840s / 10s) |
| Teleporting | Disabled (`--time-to-teleport -1`) |

### 14 Monitored Routes

`Infout-HLin`, `PBBN-FMin`, `PBBN-SirimS`, `PBBN-SirimW`, `PBBN-SKE`, `PBBW-FMin`, `PBBW-SKE`, `SirimE-HLin`, `SirimS-HLin`, `SirimS-PBBN`, `SirimW-HLin`, `SirimW-SirimE`, `SKE-HLin`, `SKE-PBBN`

Each route tracks: vehicles scheduled, vehicles completing, travel times, delay indices, throughput.

---

## 2. RL Formulation

### Actions

**Binary per junction** — each junction gets action ∈ `{0, 1}`:
- **0** = Keep current green phase
- **1** = Switch to next green phase (triggers yellow → red → new green cycle)

Action is **ignored** if `time_since_last_phase_change < min_green + yellow + red` (5 + 3 + 2 = **10s** minimum green).

The phase cycle within one action step (10s total):
```
[yellow: 3s] → [red: 2s] → [new green: 5s]
```

Defined in: [`traffic_signal.py`](traffic_signal.py) → `Signal.prep_phase(action)`

### States (Observations)

Each agent (junction) receives its own observation. The state function is set per agent in `config/agent_config.py`:

#### `drq_norm` (used by IDQN, IPPO) — defined in [`states.py`](states.py)

2D array of shape `(1, num_lanes, 5)` per junction. For each incoming lane:

| Feature | Normalization | Description |
|---|---|---|
| `phase_indicator` | 0 or 1 | Is this lane's phase currently green? |
| `approach` | / 28 | Number of approaching (non-stopped) vehicles |
| `total_wait` | / 28 | Sum of waiting times of stopped vehicles |
| `queue` | / 28 | Number of queued (stopped) vehicles |
| `total_speed` | / 20 / 28 | Sum of vehicle speeds |

#### `ma2c` (used by IMA2C) — defined in [`states.py`](states.py)

1D vector per junction combining:
- **Wave** = `queue + approach` per lane, normalized by `norm_wave`, clipped to `clip_wave`
- **Neighbor waves** = downstream junction waves × `coop_gamma`
- **Max waits** per lane, normalized by `norm_wait`, clipped to `clip_wait`

#### Other state functions (not commonly used for BB5B)
- `wave` — simple queue+approach per direction (MAXWAVE)
- `mplight` — pressure-based (MPLight)
- `drq` — unnormalized version of drq_norm

### Rewards

The reward function is set per agent. Each returns a `dict` mapping `junction_id → reward_value`:

#### `wait_norm` (default for IDQN, IPPO)
```python
reward = clip(-total_wait / 224, -4, 4)
```
Sum of waiting times across all incoming lanes, normalized and clipped. **Lower wait = higher reward.**

#### `queue_maxwait` (commonly used, configurable via `--reward-type`)
```python
reward = -(queue_count + max_wait × 0.4)
```
Penalizes both the queue length and the maximum waiting time per lane.

#### `queue_maxwait_neighborhood` (used by IMA2C)
Same as `queue_maxwait` but adds **0.9 × neighbor's reward** for each downstream junction.

#### Other reward functions
| Function | Formula | Used by |
|---|---|---|
| `wait` | `-total_wait` (raw) | MAXWAVE, MAXPRESSURE, STOCHASTIC |
| `pressure` | `-Σ(inbound_queue - outbound_queue)` | MPLight |
| `fma2c` | fringe arrivals + liquidity + neighborhood | FMA2C |

---

## 3. Deep Dive: How Data Is Extracted from SUMO

This section explains the exact code path for extracting actions, states, rewards, and metrics from the SUMO simulator.

### 3.1 Raw Data Collection via TraCI

All data from SUMO is pulled through **TraCI** (Traffic Control Interface). On every `env.reset()`, a context subscription is set up in `multi_signal.py`:

```python
# multi_signal.py → setup_traci_subscriptions()
traci.junction.subscribeContext(
    junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 1000000,
    [tc.VAR_SPEED, tc.VAR_MAXSPEED, tc.VAR_ROUTE_ID, tc.VAR_LANE_ID,
     tc.VAR_TYPE, tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_DISTANCE,
     tc.VAR_WAITING_TIME]
)
```

This subscribes to **every vehicle** within a 1,000,000m radius (effectively all vehicles) and retrieves these variables each simulation step:

| TraCI Variable | What it provides |
|---|---|
| `VAR_SPEED` | Current vehicle speed (m/s) |
| `VAR_MAXSPEED` | Vehicle's maximum allowed speed |
| `VAR_ROUTE_ID` | Which of the 14 routes this vehicle is on |
| `VAR_LANE_ID` | Current lane the vehicle is on |
| `VAR_TYPE` | Vehicle type (car, truck, etc.) |
| `VAR_ACCUMULATED_WAITING_TIME` | Total time spent waiting (speed < 0.1 m/s) |
| `VAR_DISTANCE` | Total distance traveled |
| `VAR_WAITING_TIME` | Waiting time in the current step |

This subscription is read via `get_traci_subscription()` and cached per step to avoid redundant calls.

### 3.2 Per-Lane Observation Building (`Signal.observe()`)

At the end of each action step, `traffic_signal.py → Signal.observe()` scans each incoming lane and builds a `full_observation` dict:

```python
# traffic_signal.py → Signal.observe()
for lane in self.lanes:
    lane_vehicles = self.get_vehicles(lane, distance)  # Only within max_distance (200m)
    for vehicle in lane_vehicles:
        # Classify each vehicle:
        if waiting_time > 0:
            lane_measures['queue'] += 1           # Stopped vehicle → adds to queue count
            lane_measures['total_wait'] += wait   # Accumulates total wait
            if wait > lane_measures['max_wait']:
                lane_measures['max_wait'] = wait  # Tracks maximum individual wait
        else:
            lane_measures['approach'] += 1        # Moving vehicle → approaching
```

The resulting `full_observation` per lane contains:

| Key | Type | Description |
|---|---|---|
| `queue` | int | Count of stopped vehicles (wait > 0) |
| `approach` | int | Count of moving vehicles |
| `total_wait` | float | Sum of waiting times of all stopped vehicles |
| `max_wait` | float | Maximum waiting time of any single vehicle |
| `vehicles` | list[dict] | Per-vehicle details: `id`, `wait`, `speed`, `acceleration`, `position`, `type` |

Additionally, the observation tracks **arrivals** and **departures** (vehicles entering/leaving the detection zone) by comparing with the previous step.

> **Important:** Only vehicles within `max_distance` (200m for RL agents, 50m for MAXWAVE) from the traffic light are detected. This simulates real-world detector range limitations.

### 3.3 How Actions Are Applied

When `main.py` calls `env.step(act)`, the action dict `{junction_id: 0 or 1}` flows through this pipeline:

```
multi_signal.py → step(act)
│
├─ 1. signal.prep_phase(action)           # traffic_signal.py
│     if action == 0 or min_green not met:
│         → do nothing (keep current phase)
│     if action == 1:
│         → set next_phase = current + 3   # Skip yellow+red to next green
│         → traci.trafficlight.setPhase(id, phase + 1)  # Immediately go yellow
│
├─ 2. step_sim() × 5 (yellow=3s + red=2s)
│     └─ signal.update()                   # traffic_signal.py
│          - After 3s yellow → transitions to red
│          - After 2s red → transitions to new green (next_phase)
│          - Also handles max_green=90s timeout (forced phase change)
│
├─ 3. signal.set_phase()                   # Apply the resolved green phase
│
├─ 4. step_sim() × 5 (remaining green time: 10s - 3s - 2s = 5s)
│
└─ 5. signal.observe()                    # Collect new state after action takes effect
```

**Phase structure in SUMO (per junction):** Phases are stored in triplets: `[green, yellow, red, green, yellow, red, ...]`
- INFMain_Junc: 3 green phases → 9 total SUMO phases (indices 0,3,6 are green)
- PBB_Junc: 4 green phases → 12 total SUMO phases (indices 0,3,6,9 are green)
- SIRIM_Junc: 4 green phases → 12 total SUMO phases

### 3.4 How States Are Constructed

After `signal.observe()` collects raw lane data, `states.py` transforms it into agent-ready tensors.

**Example: `drq_norm` (used by IDQN):**

```python
# states.py → drq_norm()
for signal_id in signals:
    obs = []
    for i, lane in enumerate(signal.lanes):
        lane_obs = [
            1 if i == signal.phase else 0,                    # Current phase indicator
            signal.full_observation[lane]['approach'] / 28,    # Normalized approaching vehicles
            signal.full_observation[lane]['total_wait'] / 28,  # Normalized total waiting time
            signal.full_observation[lane]['queue'] / 28,       # Normalized queue count
            sum(v['speed'] for v in vehicles) / 20 / 28,      # Normalized total speed
        ]
        obs.append(lane_obs)
    observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    # Shape: (1, num_lanes, 5)
```

**The normalization constant 28** comes from the expected max number of vehicles per lane (~28 vehicles fit in a 200m detection zone at typical car spacing).

**Example: `ma2c` (used by IMA2C):**

```python
# states.py → ma2c()
# Step 1: Compute wave (queue + approach) per lane, normalized
waves = [(queue + approach) for each lane]
waves = clip(waves / norm_wave, 0, clip_wave)

# Step 2: Append neighbor junction waves (cooperative)
for each downstream neighbor:
    waves = concat(waves, coop_gamma * neighbor_waves)

# Step 3: Compute max waits per lane, normalized
waits = [max_wait for each lane]
waits = clip(waits / norm_wait, 0, clip_wait)

# Final observation = concat(waves, waits)
# Shape: 1D vector, size varies by junction connectivity
```

### 3.5 How Rewards Are Computed

Rewards are computed from the same `full_observation` data, but use a different aggregation:

**Example: `wait_norm` (IDQN default):**

```python
# rewards.py → wait_norm()
for signal_id in signals:
    total_wait = 0
    for lane in signal.lanes:
        total_wait += signal.full_observation[lane]['total_wait']
    rewards[signal_id] = clip(-total_wait / 224, -4, 4)
```

**The normalization constant 224** = 28 vehicles × 8 lanes (approx. max total wait).

The reward is always **negative** (penalty-based): the agent is never rewarded positively, it just tries to minimize the penalty. A reward of 0 means no waiting at all.

**Example: `queue_maxwait` (commonly used with `--reward-type queue_maxwait`):**

```python
# rewards.py → queue_maxwait()
for signal_id in signals:
    reward = 0
    for lane in signal.lanes:
        reward += signal.full_observation[lane]['queue']        # Queue penalty
        reward += signal.full_observation[lane]['max_wait'] * 0.4  # Max wait penalty (weighted)
    rewards[signal_id] = -reward
```

This penalizes both **how many** vehicles are waiting AND **how long** the worst-off vehicle has been waiting, weighted at 0.4.

### 3.6 How Episode-End Metrics Are Collected

When `done=True`, `multi_signal.py` performs a final computation pass:

#### Vehicle Lifecycle Tracking (runs every simulation step)

Throughout the episode, `multi_signal.py` maintains `vehicles_on_simulation` — a dict tracking every vehicle that ever appeared:

```python
# multi_signal.py → update_waiting_time_all_vehicles_in_simulation()
# Called every sim step via step_sim()
vehicles_on_simulation[veh_id] = {
    "lanes": {lane_id: accumulated_waiting_time},
    "time": {
        "time_of_appearance": sim_step,          # When the vehicle entered
        "time_of_disappearance": sim_step,       # When it left (set later)
        "time_of_total_journey": disappear - appear,  # Total travel time
    },
    "routeID": route_id,                         # Which of the 14 routes
    "type": vehicle_type,                        # Car, truck, etc.
    "max_speed": max_speed,                      # Vehicle's speed limit
    "ideal_travel_time": route_length / max_speed,  # Free-flow travel time
    "total_distance": distance_traveled,
}
```

Each step also calls:
- `check_if_vehicle_has_not_disappeared_from_environment()` — detects if a vehicle left the simulation and records `time_of_disappearance`
- `update_waiting_time_vehicles_on_incoming_lanes()` — tracks per-lane waiting on incoming lanes
- `check_if_vehicle_pass_the_junctions()` — records when vehicles appear on outbound lanes

#### Final Calculations (`calculate_travel_time_and_delays()`)

At episode end, for each **completed** vehicle (has `time_of_total_journey`):

```python
# 1. Record actual travel time per route
routes_info[route_id]["total_travel_time_of_all_vehicles"].append(time_of_total_journey)

# 2. Record ideal travel time per route
routes_info[route_id]["total_ideal_travel_time_of_all_vehicles"].append(ideal_travel_time)

# 3. Compute per-vehicle delay index
delay = actual_travel_time / ideal_travel_time_for_vehicle_type
routes_info[route_id]["total_delays_of_all_vehicles"].append(delay)
```

Then per-route aggregates are computed:

```python
# Throughput
ThruPut_Idx = vehicles_completing / vehicles_scheduled

# Average delay
Delay_Idx_Average = mean(total_delays_of_all_vehicles)

# Weighted average delay (route's share of traffic × its avg delay)
weight = (vehicles_on_route / total_vehicles_completing) * Delay_Idx_Average
```

All of this lands in the `info` dict returned by `env.step()`, which `log_metrics()` then writes to MLflow.

---

## 4. Agent Configurations

Defined in [`config/agent_config.py`](config/agent_config.py):

| Agent | State fn | Reward fn | Key Hyperparameters |
|---|---|---|---|
| **IDQN** | `drq_norm` | `wait_norm` | batch=256, γ=0.99, ε: 1.0→0.0, target_update=500 |
| **IPPO** | `drq_norm` | `wait_norm` | (uses pfrl defaults) |
| **IMA2C** | `ma2c` | `queue_maxwait_neighborhood` | GRU=64, hidden=128, batch=120, γ=0.96, lr=2.5e-4 |
| **STOCHASTIC** | `mplight` | `wait` | Random phase selection (baseline) |
| **MAXWAVE** | `wave` | `wait` | Greedy: pick phase with max queue+approach |
| **MAXPRESSURE** | `mplight` | `wait` | Greedy: pick phase minimizing pressure |

All RL agents use `max_distance=200` (meters) — vehicles beyond this distance are not observed.

---

## 5. File Responsibilities

### Core Training Loop

| File | Role |
|---|---|
| [`main.py`](main.py) | **Orchestrator** — parses args, runs episodes, calls agent/env, handles model selection |
| [`utils/mlflow_logger.py`](utils/mlflow_logger.py) | **MLflow logging** — init, metrics, artifacts, run management |

### Environment

| File | Role |
|---|---|
| [`multi_signal.py`](multi_signal.py) | **Gym environment** — manages SUMO, computes step/reset, builds `info` dict with all metrics, saves `metrics.csv` |
| [`traffic_signal.py`](traffic_signal.py) | **Per-junction signal controller** — phase transitions (green→yellow→red→green), lane observations, vehicle detection, waiting time tracking |
| [`rewards.py`](rewards.py) | **Reward computation** — 6 functions mapping signal observations → per-junction scalar rewards |
| [`states.py`](states.py) | **State construction** — 7 functions mapping signal observations → per-junction feature vectors |

### Configuration

| File | Role |
|---|---|
| [`config/map_config.py`](config/map_config.py) | Map parameters: net file, timing, step length |
| [`config/agent_config.py`](config/agent_config.py) | Agent → state/reward/hyperparameter mapping |
| [`config/mdp_config.py`](config/mdp_config.py) | MDP-specific params (norm values, cooperation gamma, management hierarchy) |
| [`config/signal_config.py`](config/signal_config.py) | Lane-to-junction mappings, downstream connections |

### Agents

| File | Role |
|---|---|
| [`agents/pfrl_dqn.py`](agents/pfrl_dqn.py) | **IDQN** — Independent DQN per junction (pfrl library) |
| [`agents/pfrl_ppo.py`](agents/pfrl_ppo.py) | **IPPO** — Independent PPO per junction |
| [`agents/pfrl_ma2c.py`](agents/pfrl_ma2c.py) | **IMA2C** — Multi-Agent Actor-Critic with GRU |
| [`agents/stochastic.py`](agents/stochastic.py) | **STOCHASTIC** — Random baseline |
| [`agents/maxwave.py`](agents/maxwave.py) | **MAXWAVE** — Greedy rule-based baseline |
| [`agents/maxpressure.py`](agents/maxpressure.py) | **MAXPRESSURE** — Pressure-minimization baseline |

### Utilities

| File | Role |
|---|---|
| [`utils/BB5B_sumo_methods.py`](utils/BB5B_sumo_methods.py) | `get_the_routes_info()` — initializes the 14-route tracking structure |
| [`utils/time_utils.py`](utils/time_utils.py) | Converts time period strings (e.g., `7-8am`) to seconds |

---

## 6. Data Flow — One RL Step

```
main.py                        multi_signal.py                    traffic_signal.py
───────                        ───────────────                    ─────────────────
act = agent.act(obs)  ──────►  step(act):
                                 ├─ signal.prep_phase(act)  ──►  Sets yellow if action=1
                                 ├─ step_sim() × 5 steps         Updates waiting times
                                 │    └─ signal.update()    ──►  yellow→red→green cycle
                                 ├─ signal.set_phase()      ──►  Applies new green phase
                                 ├─ step_sim() × 5 steps         Run remaining sim steps
                                 ├─ signal.observe()        ──►  Collects lane data:
                                 │                                queue, approach, wait,
                                 │                                speed, max_wait per lane
                                 │
                                 ├─ state_fn(signals)  ─────────► states.py: builds obs
                                 ├─ reward_fn(signals) ─────────► rewards.py: computes reward
                                 ├─ calc_metrics()                Saves to metrics.csv
                                 │
                                 └─ if done:
obs, rew, done, _, info  ◄─────    calculate_travel_time_and_delays()
                                    Build full info dict with
                                    all episode-end metrics
agent.observe(obs, rew, ...)
```

---

## 7. What Gets Logged and Where

### metrics.csv (legacy, per-step)
- **Created by:** `multi_signal.py` → `save_metrics()` on `env.close()`
- **Location:** `{log_dir}/{connection_name}/metrics_{run}.csv`
- **Contents:** `step, rewards, max_queues, queue_lengths` per simulation step
- **Purpose:** Fine-grained step-by-step CSV data

### tripinfo.xml (SUMO native)
- **Created by:** SUMO automatically via `--tripinfo-output`
- **Location:** `{log_dir}/{connection_name}/tripinfo_{run}.xml`
- **Contents:** Per-vehicle trip data (depart time, arrival time, route, waiting time, etc.)

### MLflow (experiment tracking)
- **Created by:** `utils/mlflow_logger.py`
- **Storage:** `resco_benchmark/mlflow.db` (SQLite)
- **Contents:** See [MLFLOW_TRACKING_GUIDE.md](MLFLOW_TRACKING_GUIDE.md) for full details
  - Parameters (agent, map, hyperparams)
  - Episode-end metrics (delay index, throughput, vehicle counts)
  - Per-route metrics (throughput, average delay)
  - Per-vehicle lists as text artifacts
  - Best model as .zip artifact

### Model checkpoints
- **Created by:** `main.py` → saves after each validation episode
- **Location:** `{models_dir}/eps_{N}/` per validation episode
- **Best model:** Auto-selected, zipped, uploaded to MLflow

---

## 8. Key Concepts

### Delay Index
```
delay_index = (max_speed × travel_time) / route_length
```
- **1.0** = free-flow (ideal) · **>1.0** = delayed · **Lower is better**

### Throughput Index
```
throughput = vehicles_completing / vehicles_scheduled
```
- **1.0** = all vehicles completed their route

### Training vs Validation Routes
- **Training:** Random routes generated each episode via `RoutesGenerator`
- **Validation:** Fixed routes from real-world data (`{validation_day}/{validation_period}`)

### Best Model Selection (in `main.py`)
1. **Maximize** vehicles completing journey
2. **Minimize** average delay index (tiebreaker)
