
## Algorithms

<p align="center">
  <img src="../gif_collection/bipedal_walker.gif" alt="FlashSAC Open Duck Demo" width="100%"/>
</p>

### DQN

The target network selects and evaluates the maximum-valued next action:

$$
y = r + \gamma \max_{a'}Q_{target}(s',a')
$$

### Double DQN

The online network selects the best action, while the target network evaluates it:

$$
y = r + \gamma Q_{target}(s',\arg\max_{a'}Q_{online}(s',a'))
$$

This reduces **Q-value overestimation** compared with standard DQN.

## Main Components

* Experience Replay — buffer size: `100,000`
* Epsilon-Greedy exploration
* Online & Target Q-Networks
* Soft Target Updates (`τ = 0.005`)
* Huber Loss (`SmoothL1Loss`)
* Gradient Clipping
* Adam Optimizer
* 500 training episodes

## Network

```text
24 Observations
      ↓
Linear(128) + ReLU
      ↓
Linear(128) + ReLU
      ↓
81 Q-values
```

## Outputs

```text
dqn_bipedal.pt
ddqn_bipedal.pt
bipedal_walker_trained.mp4
```

The trained agents are evaluated using a greedy policy, and their performance can be recorded as a video.
