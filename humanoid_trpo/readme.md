# Humanoid with TRPO

Training a 3D humanoid robot to walk forward using **Trust Region Policy Optimization (TRPO)**, a policy-gradient reinforcement learning algorithm, on the Gymnasium [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) MuJoCo environment.

## The Environment

A 3D humanoid (torso, head, two arms, two legs) must move forward without falling. It is one of the hardest standard continuous-control benchmarks because the body is unstable, high-dimensional, and falls easily.

| Item | Details |
|------|---------|
| Observation | Continuous vector of joint positions, velocities, body inertia, and contact forces (about 348 values in v5, 376 in v4) |
| Actions | 17 continuous values in `[-0.4, 0.4]` (torques applied to the joints) |
| Reward | Healthy bonus (+5 per step alive) + forward velocity - control cost (v4 also includes a contact cost) |
| Termination | The torso height leaves the healthy range (the humanoid falls) |
| Truncation | Episode reaches 1000 steps |

Because actions are continuous, the policy outputs a **Gaussian distribution** (a mean and standard deviation for each of the 17 joints), and the action is sampled from it.

## Why TRPO?

In basic policy gradient methods, one bad update can destroy a policy: the humanoid forgets how to stand and never recovers. Picking a fixed step size is hard, since too small is slow and too large is unstable.

TRPO fixes this by guaranteeing that **each update keeps the new policy close to the old one**. It takes the largest improvement step it can while staying inside a "trust region". This stability is especially valuable for the Humanoid, where a slightly wrong policy makes the robot fall immediately.

## The Algorithm in Simple Terms

The agent has two neural networks:

- **Policy network** (actor): decides the action from the state.
- **Value network** (critic): estimates how good a state is, used to compute advantages.

Each training iteration does this:

### 1. Collect experience
Run the current policy for a large batch of steps (Humanoid usually needs 10,000 or more) and record states, actions, and rewards.

### 2. Estimate advantages (GAE)
For each step, compute the **advantage**: was the action better or worse than what the value network expected?

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + (gamma * lambda) * A_{t+1}
```

A positive advantage means "do this action more"; a negative one means "do it less". Advantages are usually normalized (zero mean, unit variance) for stability.

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
Solving the constrained problem exactly needs the Fisher information matrix `H` (the curvature of the KL), which is far too large to invert for a neural network. TRPO avoids this:

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

## Humanoid-Specific Tips

Compared to Walker2d, the Humanoid is much harder, so these settings matter more:

- **Larger batches:** 10,000 to 50,000 steps per iteration give less noisy gradients.
- **Observation normalization:** Observations have very different scales, so keeping a running mean and standard deviation helps a lot.
- **More training time:** Expect millions of environment steps (often 5M or more) before stable walking appears.
- **Larger networks:** Hidden layers such as `(256, 256)` or `(64, 64)` with `tanh` are both common; larger networks may need more data.
- **Reward scaling:** The +5 healthy bonus dominates early on, so the agent first learns simply to stay alive, then to move forward.

## Key Hyperparameters

| Name | Typical value | Meaning |
|------|---------------|---------|
| `gamma` | 0.99 | Discount factor for future rewards |
| `lambda` (GAE) | 0.95 to 0.97 | Bias/variance trade-off in advantage estimation |
| `max_kl` (delta) | 0.01 | Size of the trust region |
| `cg_iters` | 10 | Conjugate gradient iterations |
| `damping` | 0.1 | Added to `H` for numerical stability |
| `backtrack_coeff` | 0.5 to 0.8 | Line search shrink factor |
| `batch_size` | 10,000+ steps | Experience collected per iteration |
| `value_lr` | 1e-3 | Learning rate of the critic |

> Adjust these to match the values used in your code.

## Usage

Install dependencies:

```bash
pip install "gymnasium[mujoco]" torch numpy matplotlib imageio imageio-ffmpeg
```

Train and evaluate (replace with your script name):

```bash
python trpo_humanoid.py
```

## Expected Result

At the start the humanoid collapses within about 20 steps, earning a return of roughly `100`. As training progresses it learns to stay upright longer, then to lean and step forward. Well-tuned runs reach returns in the thousands, though TRPO on Humanoid is slow and results vary a lot between random seeds, so run a few seeds if you can.

## TRPO vs. Related Methods

| Method | Idea |
|--------|------|
| REINFORCE / vanilla policy gradient | Simple, but unstable and sample-inefficient |
| **TRPO** | Hard KL constraint, second-order optimization, very stable but more complex |
| PPO | Approximates TRPO with a clipped objective; simpler and faster |
| SAC | Off-policy and far more sample-efficient; often the strongest choice for Humanoid |