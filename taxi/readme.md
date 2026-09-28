# Taxi-v3 with Q-Learning

A simple implementation of **Q-learning** (a reinforcement learning algorithm) that teaches an agent to solve the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment.

<video src="taxi.mp4" controls width="600"></video>

## The Problem

A taxi drives on a 5x5 grid. It must:

1. Go to the passenger's location
2. **Pick up** the passenger
3. Drive to the destination
4. **Drop off** the passenger

| Item | Details |
|------|---------|
| States | 500 (taxi position x passenger location x destination) |
| Actions | 6: `0` south, `1` north, `2` east, `3` west, `4` pickup, `5` dropoff |
| Rewards | `-1` per step, `+20` for a successful dropoff, `-10` for an illegal pickup/dropoff |

## What is Q-Learning?

The agent keeps a **Q-table**: a table with one row per state and one column per action (500 x 6). Each cell `Q[state, action]` stores how good it is to take that action in that state.

At the start the table is all zeros, so the agent knows nothing. It learns by trial and error:

1. Look at the current state.
2. Choose an action.
3. Observe the reward and the new state.
4. Update the Q-table with what it learned.
5. Repeat until the episode ends, then start a new episode.

### The Update Rule

```
Q[s, a] = Q[s, a] + alpha * ( reward + gamma * max(Q[s', :]) - Q[s, a] )
```

- `s` = current state, `a` = action taken, `s'` = next state
- `reward + gamma * max(Q[s', :])` is the **target**: the reward we got plus the best value we expect from the next state.
- `target - Q[s, a]` is the **error**: how wrong our old estimate was.
- We nudge the old value toward the target by a fraction `alpha`.

### Hyperparameters

| Name | Value | Meaning |
|------|-------|---------|
| `alpha` (learning rate) | 0.9 | How strongly new information overrides old values |
| `gamma` (discount factor) | 0.9 | How much future rewards matter compared to immediate ones |
| `epsilon` | 0.1 | Probability of taking a random action (exploration) |
| `epsilon_decay_rate` | 0.0001 | Amount epsilon drops after each episode |

### Exploration vs. Exploitation (Epsilon-Greedy)

- With probability `epsilon`, the agent picks a **random** action (explore).
- Otherwise, it picks the **best known** action: `argmax(Q[state, :])` (exploit).

Epsilon decreases every episode. Once it reaches `0`, the agent always exploits, and the learning rate is dropped to `0.0001` so the learned values stay stable.

## How the Code Works

- `make_taxi_env()` creates the environment (falls back to `Taxi-v4` if `Taxi-v3` is deprecated in your Gymnasium version).
- `run(episodes, is_training, render)` does both training and testing:
  - **Training** (`is_training=True`): starts with a zero Q-table, applies the update rule, and saves the result to `taxi.pkl`.
  - **Testing** (`is_training=False`): loads `taxi.pkl` and always picks the best action, with no learning.
  - **Rendering** (`render=True`): records frames and saves a video to `taxi.mp4`.
- After the run, a plot of the trailing 100-episode reward sum is saved as `taxi.png`.

## Usage

Install dependencies:

```bash
pip install gymnasium[toy-text] numpy matplotlib imageio imageio-ffmpeg
```

Run:

```bash
python taxi.py
```

By default the script:

1. Trains for **15,000 episodes** (no rendering)
2. Runs **10 test episodes** with the trained agent and saves a video

## Output Files


| File | Description |
|------|-------------|
| `taxi.pkl` | Saved Q-table (the trained "brain") |
| `taxi.png` | Training progress plot (reward should rise and level off) |
| `taxi.mp4` | Video of the trained agent driving |

## Expected Result

Early on, the agent wanders randomly and collects large negative rewards. As the Q-table fills in, it learns the shortest route: pick up the passenger, drive to the destination, drop off. Rewards per episode climb from strongly negative to roughly `+7` to `+9`.

## Ideas to Extend

- Try different values of `alpha`, `gamma`, and `epsilon`.
- Use an exponential epsilon decay instead of a linear one.
- Compare with SARSA or Deep Q-Networks (DQN).