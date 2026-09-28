# BipedalWalker: DQN & Double DQN

Train an AI to make a two-legged robot walk in `BipedalWalker-v3` (Gymnasium) using **DQN** (`dqn.py`) and **Double DQN** (`ddqn.py`), built with PyTorch.

## Demo

<video src="bipedal_walker_trained.mp4" controls width="600"></video>

If the video doesn't play here, [open bipedal_walker_trained.mp4](bipedal_walker_trained.mp4).

## Basics

**Reinforcement learning** = learning by trial and error. The agent tries actions, gets rewards (forward movement = good, falling = bad), and learns which actions pay off.

The robot has 4 motors, each set between -1 and 1. DQN needs a finite list of actions, so each motor is limited to `[-1, 0, 1]`:

$$3^4 = 81 \text{ possible actions}$$

A neural network (the **Q-network**) looks at the robot's 24 state values and gives a score (**Q-value**) for each of the 81 actions. The agent picks the highest one.

## DQN vs Double DQN

| | Target | Effect |
|---|---|---|
| **DQN** | $y = r + \gamma \max_{a'}Q_{target}(s',a')$ | One network picks *and* scores the action, so it becomes over-optimistic |
| **Double DQN** | $y = r + \gamma Q_{target}(s',\arg\max_{a'}Q_{online}(s',a'))$ | Online net picks, target net scores, so less overestimation |

## Main Components

- **Experience Replay** (100,000): stores past steps, trains on random batches
- **Epsilon-Greedy**: random actions early (ε = 1.0), fewer over time (min 0.05)
- **Online & Target Networks**: the target one changes slowly for stable learning
- **Soft Updates** (τ = 0.005), **Huber Loss**, **Gradient Clipping**, **Adam**
- **500 training episodes**

## Network

`24 inputs -> 128 -> 128 -> 81 outputs` (ReLU activations)

## Run

```bash
pip install torch numpy gymnasium[box2d] imageio imageio-ffmpeg
python dqn.py     # or: python ddqn.py
```

Outputs: a `.pt` checkpoint and `bipedal_walker_trained.mp4`.

## Notes

- Epsilon decay differs: `0.99` (DQN) vs `0.995` (DDQN).
- Training rewards are clipped to `[-1, 1]`, so they aren't on the official score scale.
- Set `render_mode=None` for much faster training.