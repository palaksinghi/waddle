## Environment

* **Environment:** `Walker2d-v5`
* **Algorithm:** TRPO
* **Action Space:** Continuous
* **Goal:** Learn a stable forward-walking gait
<img src="../gif_collection/walker_2d.gif" alt="FlashSAC Open Duck Demo" width="100%"/>

## TRPO Pipeline

```text
Environment
    ↓
Collect Trajectories
    ↓
GAE & Advantage Estimation
    ↓
Policy Gradient
    ↓
Conjugate Gradient
    ↓
Trust-Region Update
    ↓
Line Search
    ↓
Policy Update
```

## Key Features

* Policy & Value Networks
* Generalized Advantage Estimation (GAE)
* Conjugate Gradient Solver
* KL-Divergence Constraint
* Backtracking Line Search
* Stable continuous-control learning

TRPO restricts policy updates using a **KL-divergence constraint**, reducing excessively large updates and improving training stability.