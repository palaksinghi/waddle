# Walker2d with TRPO

Training a 2D bipedal walker to walk forward using **Trust Region Policy Optimization (TRPO)**, a policy-gradient reinforcement learning algorithm, on the Gymnasium [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) MuJoCo environment.

<video src="merged_walker2d.mp4" controls width="600"></video>

## The Environment

A two-legged robot (torso, two thighs, two legs, two feet) moves in a 2D plane. The goal is to walk forward as fast as possible without falling.

| Item | Details |
|------|---------|
| Observation | 17 continuous values (joint positions and velocities) |
| Actions | 6 continuous values in `[-1, 1]` (torques applied to the joints) |
| Reward | Healthy bonus per step + forward velocity - control cost |
| Termination | The walker falls (torso height or angle leaves the healthy range) |
| Truncation | Episode reaches 1000 steps |

Because actions are continuous, the policy outputs a **Gaussian distribution** (a mean and a standard deviation for each joint), and the action is sampled from it.

## Why TRPO?

In basic policy gradient methods (like REINFORCE), one bad update can ruin the policy: the walker suddenly forgets how to walk and never recovers. The step size is hard to pick: too small is slow, too large is unstable.

TRPO fixes this by guaranteeing that **each update keeps the new policy close to the old one**. It takes the largest improvement step it can while staying inside a "trust region". This makes training stable and steadily improving.

## The Algorithm in Simple Terms

The agent has two neural networks:

- **Policy network** (actor): decides the action from the state.
- **Value network** (critic): estimates how good a state is, used to compute advantages.

Each training iteration does this:

### 1. Collect experience
Run the current policy in the environment for a batch of steps (e.g. 5,000) and record states, actions, and rewards.

### 2. Estimate advantages (GAE)
For each step, compute the **advantage**: was the action better or worse than what the value network expected?

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + (gamma * lambda) * A_{t+1}
```

A positive advantage means "do this action more"; a negative one means "do it less".

### 3. Define the objective
Maximize the surrogate objective, which measures how much better the new policy is than the old one:

```
L(theta) = E [ ( pi_new(a|s) / pi_old(a|s) ) * A ]
```

### 4. Constrain the step with KL divergence
The update must satisfy:

```
KL( pi_old || pi_new ) <= delta      (e.g. delta = 0.01)
```

KL divergence measures how different two policies are. This is the trust region: the new policy may not stray too far from the old one.

### 5. Solve it efficiently
Solving the constrained problem exactly needs the Fisher information matrix `H` (the second-order curvature of the KL), which is far too big to invert for a neural network. TRPO avoids this:

- Compute the policy gradient `g`.
- Use **conjugate gradient** to solve `H x = g` using only Hessian-vector products (no explicit matrix).
- Scale the direction so the KL constraint is exactly met:
  ```
  step = sqrt( 2 * delta / (x^T H x) ) * x
  ```

### 6. Backtracking line search
Apply the step, then check that the KL is within the limit **and** the surrogate objective actually improved. If not, shrink the step (multiply by `0.5`, `0.25`, ...) until both hold. If nothing works, keep the old policy.

### 7. Update the value network
Fit the critic to the observed returns with a few gradient steps (regression, MSE loss).

Then repeat from step 1.

## Key Hyperparameters

| Name | Typical value | Meaning |
|------|---------------|---------|
| `gamma` | 0.99 | Discount factor for future rewards |
| `lambda` (GAE) | 0.95 to 0.97 | Bias/variance trade-off in advantage estimation |
| `max_kl` (delta) | 0.01 | Size of the trust region |
| `cg_iters` | 10 | Conjugate gradient iterations |
| `damping` | 0.1 | Added to `H` for numerical stability |
| `backtrack_coeff` | 0.5 to 0.8 | Line search shrink factor |
| `batch_size` | ~5000 steps | Experience collected per iteration |
| `value_lr` | 1e-3 | Learning rate of the critic |

> Adjust these to match the values used in your code.

## Usage

Install dependencies:

```bash
pip install "gymnasium[mujoco]" torch numpy matplotlib imageio imageio-ffmpeg
```

Train and evaluate (replace with your script name):

```bash
python trpo_walker2d.py
```

## Expected Result

At the start the walker falls within a few steps and earns very low rewards. Over training, it learns to balance, then to move forward, and eventually to walk steadily for the full 1000 steps. Typical returns rise from about `10` to over `3000` for a well-trained agent.

## TRPO vs. Related Methods

| Method | Idea |
|--------|------|
| REINFORCE / vanilla policy gradient | Simple, but unstable and sample-inefficient |
| **TRPO** | Hard KL constraint, second-order optimization, very stable but more complex |
| PPO | Approximates TRPO with a clipped objective; simpler and faster |