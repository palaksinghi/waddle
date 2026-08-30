
The main focus is learning a stable, forward-moving walking gait through carefully designed reward engineering.



## Environment

* **Environment:** Custom OpenDuck Bipedal MuJoCo
* **Algorithm:** PPO
* **Action Space:** Continuous
* **Parallel Environments:** 8
* **Training:** 1M timesteps
* **Device:** CUDA / CPU

## Reward Design

The reward combines multiple objectives:

* Linear and angular velocity tracking
* Forward progress
* Heading and lateral path control
* Gait phase and foot air-time rewards
* Left-right symmetry
* Base orientation and height stability
* Pelvis velocity tracking
* Joint limits and joint penalties
* Joint velocity/acceleration penalties
* Torque and action smoothness penalties
* Alive and termination rewards

The reward terms are individually logged during training for reward tuning and analysis.

## PPO Configuration

```text
Learning Rate    : 3e-4
Gamma            : 0.99
GAE Lambda       : 0.95
Clip Range       : 0.2
Batch Size       : 1024
N Steps          : 1024
Epochs           : 5
Entropy Coef     : 0.005
```

The PPO policy uses an MLP architecture of:

```text
512 → 256 → 128
```

for both policy and value networks.

## Training Pipeline

```text
MuJoCo Environment
        ↓
8 Parallel Environments
        ↓
Collect Rollouts
        ↓
Compute Advantages
        ↓
PPO Clipped Update
        ↓
Reward Logging & Evaluation
        ↓
Checkpoint
```

## Features

* PPO with Stable-Baselines3
* Parallel environment training
* Observation & reward normalization
* Reward-term logging
* Periodic policy rendering
* Model checkpoints
* VecNormalize checkpoints
* TensorBoard logging
* Inference video recording

The implementation also saves matching `VecNormalize` statistics with checkpoints to ensure intermediate models are evaluated with the correct observation normalization.

## Output

```text
checkpoints/
logs/
videos/
```

