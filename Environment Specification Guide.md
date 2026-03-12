# RESCO Benchmark: Environment Specification Guide

This document outlines the Multi-Agent Reinforcement Learning (MARL) abstractions, transition dynamics, and formal Markov Decision Process (MDP) configurations bridging the OpenAI Gymnasium interface with the SUMO microscopic traffic simulator.

## 1. Architectural Abstractions

The environment relies on a two-tier abstraction model mapping the global simulation state to localized actor-critic networks.

* **Global Environment (`MultiSignal`):** Extends `gym.Env` to orchestrate the global simulation loop and handle multi-agent action distribution. It advances the underlying TraCI simulation at a fixed macro-step control frequency, $\Delta t$. It optimizes I/O latency by leveraging large-radius context subscriptions (`traci.junction.subscribeContext`) to extract global kinematic data in a single API call per step.
* **Localized POMDP Node (`Signal`):** Represents an isolated intersection proxy. It acts as a deterministic Finite State Machine (FSM) that intercepts agent actions, applying hard-coded safety transition logic (yellow clearance and all-red intervals) to shield the actor network from learning safety-critical collision avoidance.

## 2. Markov Decision Process (MDP) Formulation

### State Space ($\mathcal{S}$)
The environment constructs localized, continuous observation tensors by aggregating TraCI loop-detector metrics. A standard formulation used in shared-parameter algorithms (like IDQN/IPPO) is `drq_norm`:
* **Dimensionality:** $1 \times N \times 5$, where $N$ is the number of incoming lanes.
* **Feature Vector:**
    1.  One-hot phase indicator variable $\{0, 1\}$.
    2.  Approaching vehicle count (normalized by $28$).
    3.  Cumulative waiting time (normalized by $28$).
    4.  Halting queue length (normalized by $28$).
    5.  Aggregate spatial velocity of vehicles on the lane (normalized by $20 \times 28$).

### Action Space ($\mathcal{A}$)
The discrete action space dictates phase transitions. The interpretation relies on the configuration:
* **Direct Phase Selection (Acyclic):** The actor outputs logits across a categorical distribution corresponding to the total number of predefined green phases.
* **Binary Selection (Cyclic):** In specific configurations (e.g., `BB5B`), the agent operates within $\mathcal{A} \in \{0, 1\}$, where $0$ extends the current active green phase and $1$ triggers the FSM to cycle to the next sequential predefined phase.

### Reward Functions ($\mathcal{R}$)
Rewards yield scalar localized credit assignments decoupled from the global simulation state.
* **`queue_maxwait`:** $r_t = -\sum_{l} (Q_l + 0.4 \times W_{max, l})$, establishing a penalty gradient based on absolute queue accumulation and the maximum starvation delay of any single vehicle on lane $l$.
* **`pressure`:** $r_t = -(\sum Q_{in} - \sum Q_{out})$, driving max-pressure routing to optimize global throughput via local spatial density differentials.

## 3. Topologies and Valid Phase Spaces

To facilitate weight sharing across heterogeneous intersections (e.g., MPLight), native action indices are mapped to a global, standardized dictionary of phase movement pairs.

**Standard Semantic Phase Movement Indices:**
* **0:** S-W | **1:** S-S | **2:** S-E
* **3:** W-N | **4:** W-W | **5:** W-S
* **6:** N-E | **7:** N-N | **8:** N-W
* **9:** E-S | **10:** E-E | **11:** E-N

**Environment Phase Dimensionality:**
The action space cardinality is defined strictly by the valid legal phases configured per topology.

* **Synthetic Grids:**
    * `grid4x4` (8 phases): `[1, 7]`, `[2, 8]`, `[1, 2]`, `[7, 8]`, `[4, 10]`, `[5, 11]`, `[10, 11]`, `[4, 5]`.
    * `arterial4x4` & `arterial5x5` (5 phases): `[1, 7]`, `[5, 11]`, `[4, 10]`, `[4, 5]`, `[10, 11]`.
* **Real-World Networks:**
    * `ingolstadt1` (3 phases): `[1, 7]`, `[7, 8]`, `[9, 11]`.
    * `cologne1` (4 phases): `[1, 7]`, `[2, 8]`, `[4, 10]`, `[5, 11]`.
    * `cologne3` (9 phases): `[1, 7]`, `[2, 8]`, `[1, 2]`, `[7, 8]`, `[4, 10]`, `[5, 11]`, `[10, 11]`, `[4, 5]`, `[9, 11]`.
    * `ingolstadt7` (11 phases): Contains the 9 cologne phases, plus `[3, 5]` and `[7, 9]`.
    * `ingolstadt21` & `cologne8` (13 phases): Contains the 11 ingolstadt7 phases, plus `[0, 2]` and `[6, 8]`.
    * `turin5` (14 phases): Contains the 13 standard phases, plus `[0, 1]`.
* **Custom Topology (`BB5B`):**
    * (7 phases): `[0, 1]`, `[6, 7]`, `[4, 5]`, `[9, 10]`, `[1, 7]`, `[2, 3]`, `[3, 4]`.

## 4. Localized Action Space Masking (`valid_acts`)

In Multi-Agent Reinforcement Learning (MARL) topologies utilizing shared-parameter algorithms (such as MPLight or FMA2C), the actor network $\pi_\theta$ must output logits across a standardized categorical distribution. 

The framework employs the `valid_acts` dictionary as an invariant projection mask to create a bijective mapping from the standard global phase movements to the specific, discrete NEMA phases defined in a local intersection's `.net.xml` logic.

### The `BB5B` Topology Mapping Matrix

For the custom `BB5B` environment, the global tensor of non-conflicting traffic movements (`phase_pairs`) is defined as: `[[0, 1], [6, 7], [4, 5], [9, 10], [1, 7], [2, 3], [3, 4]]`.

**1. `PBB_Junc` (4-Phase Action Space)**
* **Mapping:** `{1: 0, 7: 1, 4: 2, 10: 3}`
* **Logic:** The local action logits $a_t \in \{0, 1, 2, 3\}$ correspond to the shared network's output dimensions mapped to global keys 1, 7, 4, and 10.

**2. `INFMain_Junc` (3-Phase Action Space)**
* **Mapping:** `{1: 0, 7: 1, 3: 2}`
* **Logic:** The local action logits $a_t \in \{0, 1, 2\}$ correspond to the shared network's output dimensions mapped to global keys 1, 7, and 3.

**3. `SIRIM_Junc` (4-Phase Action Space)**
* **Mapping:** `{1: 0, 4: 1, 7: 2, 10: 3}`
* **Logic:** The local action logits $a_t \in \{0, 1, 2, 3\}$ correspond to the shared network's output dimensions mapped to global keys 1, 4, 7, and 10.

### Architectural Anomaly Note for `BB5B`
In standard benchmark configurations, the keys within the `valid_acts` dictionary strictly correlate to the $0$-indexed positions of the `phase_pairs` array. However, in the `BB5B` configuration, the specific mapping keys (e.g., 7, 10) exceed the length of its 7-element `phase_pairs` array. This indicates that the `BB5B` implementation maps its local actions directly to the absolute traffic movement index primitives (where $1$ maps to S-S, $4$ maps to W-W, $7$ maps to N-N, and $10$ maps to E-E) rather than utilizing the `phase_pairs` array as the intermediary lookup vector.
