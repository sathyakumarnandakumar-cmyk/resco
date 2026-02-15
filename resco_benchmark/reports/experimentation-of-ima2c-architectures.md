
# Experimentation: IMA2C Model Architectures

**Date**: Feb 15, 2026
**Agent**: IMA2C (Independent Multi-Agent Advantage Actor-Critic)
**Environment**: BB5B (Malaysia Map)
**Experiment Group**: IMA2C Batch (55 Episodes)

## 1. Introduction

This report analyzes the performance of the newly implemented **PyTorch IMA2C** agent across four neural network architectures. We compare these results against the previously benchmarked **IDQN** and **IPPO** agents.

*Note: Metrics in this report are averaged over the entire duration of the validation episodes (3600s), providing a more accurate representation of overall traffic flow than instantaneous snapshots.*

---

## 2. Model Architectures & Implementation Details

The **IMA2C (Independent Multi-Agent Advantage Actor-Critic)** agent was implemented in **PyTorch**, porting the logic from the original TensorFlow MA2C codebase but adapting it to a modern, modular architecture. 

### Implementation Specifics
1.  **Decentralized Execution, Centralized Configuration**:
    *   The `IMA2C` class manages a dictionary of independent `MA2CAgent` instances, one for each intersection.
    *   While agents act independently, they share a common configuration and are updated synchronously.

2.  **Shared Feature Encoder**:
    *   Raw observations are not fed directly into the network. Instead, a custom **Feature Encoder** slices the input vector into three semantic components:
        *   **Wave Features**: Queue lengths and vehicle positions.
        *   **Wait Features**: Cumulative wait times per lane.
        *   **Fingerprints**: Action distributions from neighboring agents.
    *   Each component is processed by its own dedicated Linear layer (embedding) before being concatenated. This ensures the network can learn distinct representations for local state vs. neighbor context.

3.  **Fingerprint Mechanism**:
    *   To promote coordination, each agent broadcasts its current **Policy** (action probability distribution) to its downstream neighbors.
    *   These "fingerprints" are included in the neighbor's observation for the *next* time step, allowing agents to anticipate neighbor behavior.

4.  **Training Loop**:
    *   **On-Policy Learning**: Agents collect trajectories in a local buffer and perform updates every batch (e.g., 120 steps).
    *   **Advantage Estimation**: Uses N-step returns to compute advantages ($A_t = R_t - V(s_t)$).
    *   **Optimization**: Trained via **RMSprop** with Gradient Clipping (norm 40) and **Entropy Regularization** (coef 0.01) to encourage exploration.

---

### Neural Network Variants
We evaluated four distinct "backbone" architectures that process the output of the Shared Feature Encoder:

#### A. MLP (Baseline)
*   **Structure**: Feature Encoder $\rightarrow$ Fully Connected (FC) Layer $\rightarrow$ LayerNorm $\rightarrow$ ReLU $\rightarrow$ Heads.
*   **Characteristics**: Stateless. Decisions are based purely on the immediate snapshot of traffic.
*   **Role**: Serves as a baseline to quantify the value of temporal memory.

#### B. GRU (Recurrent)
*   **Structure**: Feature Encoder $\rightarrow$ **GRUCell** (Hidden Size 64) $\rightarrow$ Heads.
*   **Characteristics**: Maintains a hidden state vector $h_t$ that evolves over time.
*   **Role**: Lightweight memory, capable of capturing short-term traffic dynamics (e.g., "queue was growing").

#### C. LSTM (Recurrent / Classic)
*   **Structure**: Feature Encoder $\rightarrow$ **LSTMCell** (Hidden Size 64) $\rightarrow$ Heads.
*   **Characteristics**: Uses both hidden state $h_t$ and cell state $c_t$ to manage long-term dependencies.
*   **Role**: The architecture used in the original MA2C paper. Robust but computationally heavier than GRU.

#### D. Transformer (Attention)
*   **Structure**: Feature Encoder $\rightarrow$ Sliding Window Buffer (Seq Len 16) $\rightarrow$ **TransformerEncoder** $\rightarrow$ Heads.
*   **Implementation**: 
    *   maintains a FIFO buffer of the last 16 observations.
    *   Applies **Positional Encodings** to retain temporal order.
    *   Uses **Causal Masking** in the Self-Attention layer to prevent looking into the future.
*   **Role**: Explicitly attends to specific past events (e.g., "10 seconds ago, neighbor cleared a platoon") rather than compressing everything into a fixed vector.

### Neptune Trial IDs (Reference)
**Workspace**: sathyakumar/Tensorcell-test

| Run ID | Architecture | Composition |
| :--- | :--- | :--- |
| **TEN-101** | **MLP** | FC + LayerNorm |
| **TEN-102** | **GRU** | GRUCell |
| **TEN-103** | **LSTM** | LSTMCell |
| **TEN-104** | **Transformer** | Sliding Window (16) + Self-Attention |

---

## 3. Training Progress

### Reward Curves
Lower (more negative) reward implies higher queues/wait times. Ideally, the curve should rise towards 0.

![IMA2C Training Rewards](plots/ima2c_training_rewards.png)

**Observations**:
- **Transformer** and **GRU** (Memory-based models) ended up performing best, validating the hypothesis that temporal context requires recurrence or attention.
- **MLP** and **LSTM** converged to identical performance, slightly worse than GRU/Transformer. The identical performance suggests the LSTM cell may have saturated or behaved linearly in this specific setup.
- **Transformer** showed the most consistent learning curve, steadily improving throughout the 55 episodes.

---

## 4. Evaluation Results: Throughput & Delays

### Final Performance (Episode 55 - Validation)

| Model | Throughput (Veh) | Delay Index (TTI) |
| :--- | :---: | :---: |
| **GRU** | 3125 | **3.41** |
| **Transformer**| 3127 | 3.45 |
| **MLP** | **3128** | 3.51 |
| **LSTM** | **3128** | 3.51 |

*Note: Episode 55 represents the final validation checkpoint. Metrics are standardized: Throughput (Completed Vehicles) and TTI (Travel Time Index).*

### Combined Metrics (Wait & Queue)
The following plot shows the average wait time and queue length across all validation episodes.

![IMA2C Combined Metrics](plots/ima2c_combined_metrics.png)

### Key Findings
1. **Memory Matters**: Both **GRU** and **Transformer** outperformed the memory-less **MLP**, confirming that capturing temporal traffic patterns (e.g., growing queues) is beneficial.
2. **Efficiency**: **GRU** achieved the best efficiency (Delay Index **3.41**), suggesting that its simpler recurrent structure might be slightly more sample-efficient than the Transformer (3.45) given the limited training data (55 episodes).
3. **Performance Gap**: While improved, the IMA2C Delay Index (~3.4) still lags behind the fully tuned IDQN (~2.38), indicating room for hyperparameter optimization.

---


## 5. COMPARISON: IMA2C vs IDQN vs IPPO

Comparing the best configurations from all three agent types:

### Summary of Best Configurations (Averaged Over Training)

| Agent | Best Net | Throughput (Veh/Ep) | Delay (Index)* |
| :--- | :--- | :---: | :---: |
| **IDQN** | MLP | **3321** | **3.51** |
| **IPPO** | Default | **3341** | **4.09** |
| **IMA2C** | Transformer | **3300** | **4.41** |

*\*Note: Delay Index (Travel Time Index) = Average Ratio of Actual Travel Time to Free Flow Time. Lower is better. IDQN achieving 3.51 implies travel takes ~3.5x longer than ideal.*


### Comparative Analysis: IMA2C vs IDQN vs IPPO

The plot below presents a head-to-head comparison of the top-performing configurations from each algorithm. Metrics (Throughput and Delay Index) are derived directly from standardized simulation logs (`tripinfo.xml`) to ensure consistency.

#### Models Compared (6 Lines)
1.  **IDQN - MLP**: Independent DQN with fully connected layers.
2.  **IDQN - DoubleConv**: Independent DQN using convolutional input processing.
3.  **IPPO - Default**: Independent PPO with standard architecture.
4.  **IPPO - GNN**: Independent PPO using Graph Neural Networks for state encoding.
5.  **IMA2C - Transformer**: Actor-Critic with Transformer-based temporal attention.
6.  **IMA2C - GRU**: Actor-Critic using Gated Recurrent Units for temporal memory.

> [!NOTE]
> **Performance Observations**:
> - **Efficiency (Delay Index)**: IDQN remains the most efficient, achieving a **Delay Index of ~3.5** (meaning travel takes 3.5x the ideal free-flow time). IPPO follows at ~4.1, and IMA2C trails at ~4.4.
> - **Capacity (Throughput)**: all algorithms handle similar traffic volumes (~3300 vehicles/episode), suggesting no major bottlenecks in capacity, but significant differences in travel speed.
> - **Stability**: IDQN consistently maintains the lowest travel time ratio across episodes.

![Comparison: IMA2C vs IDQN vs IPPO](plots/algo_comparison_full.png)

### Conclusions
1. **IDQN Superiority**: The independent DQN approach remains the most effective for minimizing travel time delay (lowest TTI).
2. **Architecture Impact**: Use of memory (Transformer/GRU) in IMA2C improves upon the baseline MLP but the overall policy efficiency still lags behind IDQN and IPPO.
3. **Future Work**: Tuning hyperparameters (learning rate, entropy) via Optuna is recommended to bridge the efficiency gap.

---
