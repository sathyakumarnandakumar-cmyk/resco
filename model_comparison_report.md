# Research Report: Advanced Neural Network Architectures for Traffic Control
**Date**: Feb 14, 2026  
**Agent**: IDQN (Independent Deep Q-Network)  
**Environment**: BB5B (Malaysia Map)

## 1. Introduction
This report summarizes the performance of two upgraded neural network architectures (v2) implemented to improve progress in the RESCO benchmark. Both models utilize the GPU (NVIDIA L4) and were tested over a 22-episode run.

---

## 2. Model Architectures (v2 Upgrades)

### Advanced MLP (v2)
*   **Depth**: 5 Fully Connected Layers (Increased from 3).
*   **Width**: 256 Hidden Units (Increased from 64).
*   **Normalization**: Added **Layer Normalization** after each hidden layer.
*   **Rationale**: Deeper networks with normalization can capture more complex traffic patterns and converge faster.

### Advanced GNN / Multi-Head Attention (v2)
*   **Heads**: 4 Attention Heads (Multi-Head Self-Attention).
*   **Connections**: Added **Residual (Skip) Connections**.
*   **Normalization**: Added **Layer Normalization**.
*   **Rationale**: Multi-head attention allows the agent to simultaneously attend to different aspects of neighboring lane states (e.g., queue vs. wait time).

---

## 3. Comparative Performance Metrics
These metrics were extracted from the validation episodes (11 and 22) and an early training checkpoint (5).

| Episode | Model Variant | Avg Queue (Veh) | Avg Wait (s) | Avg Travel Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **5** (Early) | MLP (v2) | 6.46 | 19.61 | 247.05 |
| | **GNN (v2)** | **6.18** | **18.02** | **203.89** |
| **11** (Valid)| **MLP (v2)** | **5.03** | **14.25** | **177.96** |
| | GNN (v2) | 5.07 | 15.24 | 190.36 |
| **22** (Valid)| **MLP (v2)** | **4.48** | **12.17** | **154.01** |
| | GNN (v2) | 4.68 | 13.20 | 162.51 |

### Key Findings
1.  **GNN Early Advantage**: The GNN model showed significantly better performance in the first 5 episodes, particularly in Travel Time (203s vs 247s). This suggests the Attention mechanism identifies local lane relationships faster than a flat MLP.
2.  **MLP Convergence**: As training progressed to Episode 22, the MLP overtook the GNN in efficiency, achieving the lowest overall wait times recorded in this sequence.
3.  **Stability**: Both models showed monotonic improvement, indicating that the added Layer Normalization and increased depth successfully resolved the "no progress" issue seen in previous iterations.

---

## 4. Hardware Optimization
To maximize throughput on the 4-core machine:
*   **Parallelization**: Experiments are now grouped into batches of 4.
*   **Resource Usage**: Each experiment occupies ~1 CPU core and ~400MB GPU VRAM.
*   **Execution**: Automated via `parallel_idqn_batch.sh`.

---

## 5. Conclusion & Next Steps
Both Advanced (v2) architectures are viable and show strong learning curves. 
**Next Steps**: Complete the full 4x4 batch experiment (ReLU vs Leaky ReLU across all seeds) to identify the most robust configuration for the final RESCO benchmark submission.
