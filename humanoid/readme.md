## Environment


<img src="../gif_collection/humanoid.gif" alt="FlashSAC Open Duck Demo" width="100%"/>

* **Environment:** `Humanoid-v5`
* **Algorithm:** TRPO
* **Framework:** PyTorch + Gymnasium
* **Action Space:** Continuous
* **Observation Space:** Continuous

## TRPO Pipeline

```text
Environment
     ↓
Collect Trajectories
     ↓
Estimate Advantages
     ↓
Compute Policy Gradient
     ↓
Conjugate Gradient
     ↓
KL-Constrained Step
     ↓
Line Search
     ↓
Update Policy
```

## Key Components

* **Policy Network** — outputs the mean and standard deviation of continuous actions.
* **Value Network** — estimates the state value \(V(s)\).
* **GAE** — calculates advantage estimates.
* **Conjugate Gradient** — approximates the natural-gradient direction.
* **Fisher-Vector Product** — estimates curvature without explicitly constructing the Hessian.
* **KL Constraint** — limits how much the new policy can differ from the old policy.
* **Backtracking Line Search** — ensures the update satisfies the trust-region constraint.

## Objective

TRPO maximizes the expected policy improvement while constraining the KL divergence:

$$
\max_\theta L(\theta)
$$

subject to

$$
D_{KL}(\pi_{\theta_{old}} \parallel \pi_\theta) \leq \delta
$$

This prevents excessively large policy updates and improves training stability.

## Output

The trained policy can be evaluated in the Humanoid environment to observe the learned locomotion behavior.