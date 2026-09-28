# FrozenLake: Q-Learning

A simple **Q-learning** agent that learns to cross a frozen lake in `FrozenLake-v1` (Gymnasium) using only NumPy.

## Demo
<p align="center">
  <img src="frozenlake.gif" alt="Taxi Environment" width="85%"/>
</p>

## The Game

The agent stands on a 4x4 grid and must walk from the start (top-left) to the goal (bottom-right) without falling into a hole.

- **States:** 16 (one per grid square)
- **Actions:** 4 (left, down, right, up)
- **Reward:** `+1` for reaching the goal, `0` otherwise
- `is_slippery=False`, so the agent always moves exactly where it intends

## How Q-Learning Works

The agent keeps a **Q-table**: a 16 x 4 grid of scores, one for each (state, action) pair. A higher score means "this move in this square is probably good."

After every step, the agent updates one score:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

- `α` (alpha, `0.5`): learning rate, how fast old scores are replaced by new information
- `γ` (gamma, `0.9`): discount factor, how much future rewards matter
- `r`: reward just received
- `s'`: the next state

The reward from the goal slowly spreads backward through the table, square by square, until the agent knows the full path.

## Settings

| Parameter | Value |
|-----------|-------|
| Episodes | 1000 |
| Learning rate (`alpha`) | 0.5 |
| Discount factor (`gamma`) | 0.9 |

**Action choice:** if the agent already has a positive score for the current square, it takes the best action. Otherwise, it picks a random one. There is no epsilon-greedy here; this works because the map is small and not slippery.

## Run

```bash
pip install numpy gymnasium imageio matplotlib
python frozenlake.py
```

Outputs:

- `frozenlake.pkl`: the trained Q-table
- `frozenlake.gif`: the agent's path after training

## Notes

- Q-learning uses a table, so it only works for small problems like this one. Big state spaces (like BipedalWalker) need a neural network instead, which is what **DQN** does.
- Turning on `is_slippery=True` makes the task much harder, and it would need epsilon-greedy exploration and more episodes.