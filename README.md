# Waddle
 
**Teaching a tiny robot duck to walk, using reinforcement learning .**
 
Nobody tells a robot exactly how to walk. Instead, it tries things in a simulator, gets rewarded when it stays upright and moves the way we asked, and slowly works out a walking style on its own. The "brain" that decides how every joint should move is called a *policy*, and Waddle is a step-by-step journey to build one.


 The duck's gait comes entirely from reward design: deciding what counts as good walking (stay upright, follow the command, take even steps, don't waste energy) and tuning how much each of those matters. The duck is trained with two algorithms, **PPO** and **FlashSAC** (Flash Soft Actor-Critic) .

<p align="center">
<img src="FlashSAC/duck.png" alt="Open Duck Mini v2" width="60%">
</p>

## Overview

Waddle is a progression of five stages, each in its own folder:

- **Tabular RL** (`frozenlake/`, `taxi/`): value-based methods on small discrete environments, where you can inspect every Q-value.
- **Deep Q-learning** (`lunar landing/dqn`, `lunar landing/ddqn`): neural function approximation, experience replay, and Double DQN to fix Q-value overestimation.
- **Neural network fundamentals** (`neural network/`): the supporting deep learning building blocks used by the later stages.
- **Continuous-control PPO** (`halfcheetah_ppo/`, `walker_2d/`, `humanoid/`, `multi_humanoid/`): policy-gradient training on MuJoCo locomotion tasks of increasing difficulty.
- **Open Duck Mini v2** (`FlashSAC/,open_duck_bipedal`): It trains the 17-DoF duck with PPO and FlashSAC .
### Workflow

1. **Start with the environment.** Each stage defines its observation space, action space, reward, and termination conditions explicitly, so you can see exactly what the agent is optimizing.
2. **Train an agent.** Discrete tasks use tabular Q-learning and DQN/DDQN; continuous tasks use PPO (and FlashSAC for the duck).
3. **Shape the reward.** For the duck, the gait is engineered by combining, weighting, and tuning distinct reward components until the robot balances and walks stably (see [Reward engineering](#reward-engineering)).


### Learning path

| Stage | Folder | Environment | Algorithm | Concept introduced |
|-------|--------|-------------|-----------|--------------------|
| 1 | `frozenlake/` | FrozenLake | Tabular RL | Value functions, exploration |
| 2 | `taxi/` | Taxi-v3 | Tabular RL | Larger discrete state spaces |
| 3 | `lunar landing/dqn` | LunarLander | DQN | Function approximation, replay buffer |
| 4 | `lunar landing/ddqn` | LunarLander | Double DQN | Overestimation bias |
| 5 | `halfcheetah_ppo/` | HalfCheetah (MuJoCo) | PPO | Continuous actions, clipped surrogate objective |
| 6 | `walker_2d/` | Walker2d (MuJoCo) | PPO | Balance and falling termination |
| 7 | `humanoid/` | Humanoid (MuJoCo) | PPO | High-dimensional control |
| 8 | `multi_humanoid/` | Multi-Humanoid | PPO | Multi-agent setup |
| 9 | `FlashSAC/` | Open Duck Mini v2 (MuJoCo / Isaac Lab) |  FlashSAC | Reward engineering, on- vs off-policy comparison |

### Classic control

<table>
  <tr>
    <th align="center">FrozenLake</th>
    <th align="center">LunarLander</th>
    <th align="center">Taxi-v3</th>
  </tr>
  <tr>
    <td align="center"><img src="gif_collection/frozenlake.gif" width="220"/></td>
    <td align="center"><img src="gif_collection/lunar_lander_trained.gif" width="220"/></td>
    <td align="center"><img src="gif_collection/taxi.gif" width="220"/></td>
  </tr>
</table>


## The robot: Open Duck Mini v2

Open Duck Mini v2 is an open-source bipedal robot with **17 degrees of freedom**. Both simulators load the same robot description, so the policy and reward code can be shared to Mujoco .

| Property | Value |
|----------|-------|
| Degrees of freedom | 17 |
| Simulators | MuJoCo |
| Algorithms | PPO, FlashSAC |


### Observation and action space

```
Observation (dim = 43):
  base angular velocity          3    (wx, wy, wz)
  projected gravity vector       3    (gravity direction in the body frame)
  joint positions                17
  joint velocities               17
  velocity command               3    (vx, vy, wz)
                                 --
                                 43
Action (dim = 17):

```

### Reward engineering
 
The gait is the result of a weighted sum of shaped terms, `r = Σ wᵢ · rᵢ(s, a)`, computed in `compute_reward(e)` and shared by PPO and FlashSAC. The function returns the total reward plus an `info` dict that logs every weighted term as `rew/<name>`, so you can see which terms dominate during training.
 
Design choices worth knowing before you tune anything:
 
- **Tracking rewards.** The duck gets a score between 0 and 1 for matching the speed it was told to walk at. The score is high only when it is very close to the target (roughly within 0.06 m/s) and drops quickly the further away it gets.
- **Walking in a straight line.** Heading and sideways drift are measured against where the duck started (its spawn position and direction), not against the previous step. Small drifts can't add up unnoticed, so the duck can't slowly curve into a circle.
- **Capped penalties.** Some penalties (`yaw_penalty`, `pelvis_vel_tracking`) have a maximum of 5. One bad step can't produce a huge penalty that throws off training.
- **Left and right legs mirror each other.** Walking is two legs taking turns, half a step apart. `symmetry` checks that each leg's pose matches the mirrored pose of the other leg from half a step ago. It looks at the 10 leg joints (5 per leg: yaw, roll, pitch, knee, ankle).
- **No stepping in place when standing still.** `feet_air_time_reward` pays nothing when the commanded speed is almost zero, so the duck isn't rewarded for stepping when it should stand still.
- **Tuned by trial and error.** The weights changed many times along the way. The old values are kept as comments in the reward file.

#### Reward terms (PPO and FlashSAC)

**Command tracking**

| Term | Definition | Weight |
|------|------------|--------|
| `track_lin_vel_xy_exp` | `exp(-‖v_cmd,xy − v_xy‖² / σ²)`, σ = 0.06 | 2.0 |
| `track_ang_vel_z_exp` | `exp(-(ω_cmd − ω_z)² / σ²)`, σ = 0.06 | 0.5 |
| `forward_progress` | Net world +x displacement this step | 8.0 |
| `pelvis_vel_tracking` | `‖v − v_cmd‖² / max(0.12, 0.5‖v_cmd‖²)`, clipped to [0, 5] | -1.0 |

**Heading and straight-line walking**

| Term | Definition | Weight |
|------|------------|--------|
| `heading_drift` | Squared wrapped yaw error vs. spawn heading | -1.0 |
| `lateral_path_deviation` | Squared perpendicular distance from the line through spawn along spawn heading | -2.0 |
| `yaw_penalty` | `5·tanh((ω_z − ω_cmd)² / 5)` | -1.0 |

**Gait**

| Term | Definition | Weight |
|------|------------|--------|
| `gait_phase_tracking` | Desired stance from `sin(phase)` per leg, matched against actual foot contact | 1.0 |
| `gait_phase_contact` | Binary phase-vs-contact match using the phase vector | 1.0 |
| `feet_air_time_reward` | On touchdown, `min(air_time, target_feet_air_time)`; zero if command < 0.05 | 2.0 |
| `symmetry` | Squared error to the mirrored leg pose from half a cycle ago | -0.5 |
| `lateral_spread` | Feet lateral distance beyond 0.25 m | -3.0 |

**Base stability**

| Term | Definition | Weight |
|------|------------|--------|
| `flat_orientation_l2` | `‖g_xy‖²` of projected gravity | -2.5 |
| `base_height_l2` | `(h − h_target)²` | -1.0 |
| `lin_vel_z_l2` | Vertical base velocity squared | -2.0 |
| `ang_vel_xy_l2` | Roll/pitch angular velocity squared | -0.05 |

**Regularization**

| Term | Definition | Weight |
|------|------------|--------|
| `joint_pos_limits` | Amount by which joints exceed their limits | -1.0 |
| `joint_penalty` | Squared deviation of leg joints from default pose | -0.001 |
| `joint_vel` | Squared leg joint velocity | -0.0005 |
| `joint_acc` | Squared leg joint acceleration | -2e-7 |
| `torque` | Approximate mechanical power, abs(τ · q̇) summed | -0.0001 |
| `action_rate_l2` | `‖aₜ − aₜ₋₁‖²` | -0.03 |
| `action_smoothness2_l2` | `‖aₜ − 2aₜ₋₁ + aₜ₋₂‖²` (action acceleration) | -0.015 |

**Survival and termination**

| Term | Definition | Weight |
|------|------------|--------|
| `alive_cost` | Constant per-step bonus | 1.0 |
| `is_terminated` | One-off penalty when the episode terminates | -25.0 |

**Termination condition:** `bad_orientation` ends the episode when `‖g_xy‖ > sin(tilt_limit)`, where `tilt_limit` is in radians (projected gravity's horizontal norm equals `sin(tilt)`, so the comparison is made against the sine, not the raw angle). 
#### Reward parity between PPO and FlashSAC

FlashSAC is trained with the same reward terms and weights as PPO (the tables above apply to both). Keeping the reward identical is what makes the [PPO vs. FlashSAC](#ppo-vs-flashsac) comparison meaningful: the only intentional difference between the two runs is the learning algorithm.

## Results

### Open Duck Mini v2

<table>
  <tr>
    <th align="center">PPO</th>
    <th align="center">FlashSAC</th>
  </tr>
  <tr>
    <td align="center"><img src="gif_collection/flash_sac.gif" width="320"/></td>
    <td align="center"><img src="gif_collection/waddle.gif" width="320"/></td>
  </tr>
</table>

### PPO vs. FlashSAC

PPO and FlashSAC are trained on the same robot with the same reward, so any difference in the results comes from the algorithm and not from the task setup.
x

#### Algorithm comparison

| Property | PPO | FlashSAC |
|----------|-----|----------|
| Family | Clipped-surrogate policy gradient | Soft actor-critic (maximum entropy, off-policy) |
| Sample efficiency | Lower: needs large amounts of fresh data | Typically higher: reuses past transitions |
| Exploration | Stochastic policy plus entropy bonus | Entropy is part of the objective |
| Stability | Generally robust, few sensitive knobs | More components to tune (critics, temperature, replay) |


#### Results

| Metric | PPO | FlashSAC |
|--------|-----|----------|
| Environment steps to first stable walk | 500 | 100000 |
| Wall-clock training time | 30 min | 15 min |
<!-- | Final mean episode return | `TODO` | `TODO` |
| Velocity tracking error (m/s) | `TODO` | `TODO` |
| Heading drift over a straight walk (rad) | `TODO` | `TODO` |
| Lateral path deviation (m) | `TODO` | `TODO` |
| Max push survived (N, duration) | `TODO` | `TODO` |
| Rough-terrain success rate | `TODO` | `TODO` | -->


Return is comparable across the two runs only because the reward is identical.


## Repo layout

```
waddle/
├── frozenlake/        -- tabular RL on FrozenLake
├── taxi/              -- tabular RL on Taxi-v3
├── lunar landing/
│   ├── dqn/           -- Deep Q-Network on LunarLander
│   └── ddqn/          -- Double DQN on LunarLander
├── neural network/    -- neural network fundamentals used by later stages
├── halfcheetah_ppo/   -- PPO on HalfCheetah (MuJoCo)
├── walker_2d/         -- PPO on Walker2d (MuJoCo)
├── humanoid/          -- PPO on Humanoid (MuJoCo)
├── multi_humanoid/    -- multi-agent humanoid
├── FlashSAC/          -- FlashSAC and PPO training for Open Duck Mini v2
└── gif_collection/    -- trained-policy rollouts shown in Results
```

## Getting started

### Clone

```bash
git clone https://github.com/palaksinghi/waddle.git
cd waddle
```

### Create and activate a virtual environment

```bash
python3.10 -m venv waddle_env
```

```bash
# macOS / Linux
source waddle_env/bin/activate

# Windows (PowerShell)
.\waddle_env\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt  
```


## Running



```bash
# Tabular RL
cd frozenlake
python train.py   

cd taxi        
python train.py                 

# DQN / Double DQN
python "lunar landing/dqn/train.py"   
python "lunar landing/ddqn/train.py"  


python halfcheetah_ppo/train.py       

# Open Duck Mini v2 

#FlashSAC
cd FlashSAC
uv run python train.py --overrides num_env_steps=100000
uv run python view_policy.py 

#PPO
cd open_duck_bipedal/mujoco
python trainm.py

```
