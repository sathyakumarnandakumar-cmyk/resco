# Reward Modelling for BB5B (Malaysia)

**Date**: Feb 15, 2026
**Map**: BB5B (PBB $\leftrightarrow$ INFMain $\leftrightarrow$ SIRIM)
**Default Reward**: `queue_maxwait`

This report analyzes available reward functions *specifically in the context of the BB5B arterial map*. BB5B is a tightly coupled North-South arterial corridor where the central intersection (`INFMain_Junc`) acts as a critical bottleneck.

---

## 1. The Chosen Default: `queue_maxwait`
**Why it works for BB5B**: This is the most balanced reward for an arterial with heavy side-street pressure.

### Structure & BB5B Impact
| Component | Formula Term | Impact on BB5B |
| :--- | :--- | :--- |
| **Queue** | $\sum Queue_l$ | **Anti-Spillback**: Prevents queues at `INFMain` from blocking `PBB` (upstream). Essential for maintaining the "green wave". |
| **Max Wait** | $0.4 \times MaxWait_l$ | **Side-Street Fairness**: The arterial flow is huge. Without this term, agents would permanently green-light the North-South flow, starving East-West traffic. |

**Formula**:
$$ R_t = - \sum_{l} (Queue_l + 0.4 \times MaxWait_l) $$

---

## 2. Alternative Rewards vs. BB5B Reality

We analyze how other standard rewards would likely perform on this specific map topology.

### A. `wait` (Minimize Total Delay)
*   **Formula**: $R_t = - \sum TotalWait_l$
*   **BB5B Analysis**: 
    *   **Risk**: High variance. The sheer volume of cars on the main road means total wait can explode quickly.
    *   **Instability**: In early training, a bad policy could cause gridlock at `INFMain`, leading to massive negative rewards (e.g., -10,000) that destabilize the gradient updates.

### B. `pressure` (Max Pressure)
*   **Formula**: $R_t = - | \sum Queue_{in} - \sum Queue_{out} |$
*   **BB5B Analysis**:
    *   **Potential**: Theoretically optimal for maximizing throughput. It would likely push traffic aggressively through `INFMain`.
    *   **Flaw**: It does not penalize delay. In BB5B, side-street cars could wait forever if their queue length ($Queue_{in}$) doesn't exceed the downstream pressure. This is unacceptable for real-world deployment.

### C. `queue_maxwait_neighborhood` (Cooperative)
*   **Formula**: $R_{local} + 0.9 \sum R_{neighbors}$
*   **BB5B Analysis**: **Highly Recommended for Future Testing**.
    *   **Why**: BB5B is a chain. If `INFMain` is congested, `PBB` *should* stop sending cars.
    *   **Mechanism**: A standard agent at `PBB` only sees its own queue. A neighborhood agent sees `INFMain`'s penalty and learns to "gate" traffic, preventing the bottleneck from worsening.

---

## 3. Suggested Reward Modelling for Future Work

To further optimize performance on this specific map, we propose these targeted modifications:

### A. Throughput-Biased (Arterial Focus)
*   **Concept**: Explicitly reward successful departures on the main road.
*   **Formula**: $R = R_{queue\_maxwait} + \alpha \times N_{departures(Arterial)}$
*   **Hypothesis**: Will increase North-South bandwidth but may slightly increase East-West delay.

### B. Bottleneck penalty
*   **Concept**: Add a specific penalty if the queue at `INFMain` exceeds a threshold (e.g., 20 vehicles).
*   **Formula**: $R_{PBB} = R_{default} - \beta \times \mathbb{I}(Queue_{INFMain} > 20)$
*   **Hypothesis**: Would force `PBB` to learn "gating" behavior much faster than standard neighborhood rewards.

### C. Speed Maximization (Free Flow)
*   **Formula**: $R = \sum (Speed / MaxSpeed)$
*   **BB5B Analysis**: Good for light traffic, but in heavy congestion (saturation), speeds drop to near zero, providing no learning signal. Likely to fail during rush hour peaks unless combined with Queue rewards.
