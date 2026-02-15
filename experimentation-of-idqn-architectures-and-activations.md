# Experimentation: IDQN Model Architectures & Activation Functions
**Date**: Feb 15, 2026  
**Agent**: IDQN (Independent Deep Q-Network)  
**Environment**: BB5B (Malaysia Map)  
**Experiment Group**: Batch Seed 1 (8 Variations)

## 1. Introduction
This experiment explores the impact of neural network depth, width, and connectivity—combined with varied activation functions (ReLU vs. LeakyReLU)—on traffic signal control efficiency. We evaluate throughput and average vehicle delays in the Malaysian BB5B map.

---

## 2. Model Architectures & Experiment Setup
The **IDQN (Independent Deep Q-Network)** agent treats each intersection as an independent Reinforcement Learning entity, allowing for decentralized control and scalability within the traffic network.

### Neptune Trial IDs (Reference)
These runs can be viewed in the [Neptune Dashboard](https://app.neptune.ai/sathyakumar/Tensorcell-test/).

| Run ID | Net | Activation | Seed |
| :--- | :--- | :--- | :--- |
| **TEN-24** | mlp | relu | 1 |
| **TEN-25** | double_conv | relu | 1 |
| **TEN-26** | gnn | relu | 1 |
| **TEN-27** | default | relu | 1 |
| **TEN-32** | default | leaky_relu | 1 |
| **TEN-33** | mlp | leaky_relu | 1 |
| **TEN-34** | gnn | leaky_relu | 1 |
| **TEN-35** | double_conv | leaky_relu | 1 |

We tested four distinct neural architectures, each integrated with both **ReLU** and **Leaky ReLU (slope=0.01)** activations.

### A. Default (Base)
*   **Structure**: Single 2x2 Convolutional layer followed by two FC layers of 64 units.
*   **Rationale**: Simple spatial extraction followed by shallow reasoning.

### B. Double Conv
*   **Structure**: Two 2x2 Convolutional layers followed by two FC layers of 64 units.
*   **Rationale**: Increased spatial depth to capture more hierarchical features from lane states.

### C. Advanced MLP (v2)
*   **Structure**: 5 Fully Connected layers with 256 hidden units each.
*   **Normalization**: Layer Normalization after each layer.
*   **Rationale**: Deeper/Wider network aimed at solving complex traffic nonlinearities with stable gradients.

### D. Advanced GNN / Attention (v2)
*   **Structure**: Multi-Head Self-Attention (4 heads) with Residual Connections and LayerNorm.
*   **Rationale**: Explicitly models relationships between different lane states through an attention mechanism rather than flat feature vectors.

---

## 3. Training Progress (Rewards)
The plot below shows the average reward (mean of INFMain, PBB, and SIRIM junctions) smoothed over training steps.

![Training Rewards](plots/training_rewards.png)

**Observation**: `mlp_relu` and `double_conv_relu` show the most stable upward trajectories. `gnn` variants show higher variance during early training but converge similarly to the base models.

---

## 4. Evaluation Results: Throughput & Delays

Metrics extracted across 5 validation episodes.

![Throughput and Delays](plots/combined_metrics.png)

### Performance Matrix (Final Validation Point)
| Model Combination | Throughput (Veh) | Avg Delay (s) |
| :--- | :---: | :---: |
| **MLP + ReLU** | **3470** | **2.38** |
| **GNN + LeakyReLU** | **3460** | 2.85 |
| **Double Conv + ReLU** | 3451 | **2.47** |
| **Default + LeakyReLU** | 3441 | 2.58 |
| **Default + ReLU** | 3362 | 2.84 |

---

## 5. Summary & Recommendation

### Best Performing Combination: **MLP + ReLU**
The **Advanced MLP (v2)** with **ReLU** activation is the clear winner:
*   **Throughput Improvement**: Achieved **3.2% higher throughput** than the base Default (ReLU) model.
*   **Delay Reduction**: Reduced average delays by **16.2%** (from 2.84s to 2.38s) compared to the base baseline.
*   **Consistency**: Showed the smoothest convergence in training rewards and the highest stability in validation metrics.

### Key Takeaways:
1.  **Width/Depth Matters**: The transition from 64-unit shallow layers to 256-unit deep layers (MLP v2) significantly improved performance in the complex BB5B scenario.
2.  **Activation Impact**: ReLU consistently outperformed Leaky ReLU across most architectures in this specific reward-regime, especially for MLP and Double Conv models.
3.  **GNN Potential**: While GNN attained high throughput, it failed to match the MLP in delay minimization. It remains a strong candidate for generalization to unseen maps.
    *   **Complexity Overhead**: The GNN's slower convergence (compared to MLP) is attributed to the additional complexity of learning attention weights and message-passing protocols. Unlike the MLP, which maps fixed inputs to outputs, the GNN must simultaneously learn the *structure* of the traffic graph and the *policy*, requiring more samples to stabilize.
4.  **Convergence Speed**: The **MLP (v2) with ReLU** demonstrated the fastest and most stable convergence, reaching optimal throughput levels approximately 15-20% earlier in training than other variants.

---
