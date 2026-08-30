# Taxi Q-Learning
<img src="../gif_collection/taxi.gif" alt="FlashSAC Open Duck Demo" width="100%"/>

A tabular Q-learning implementation using Gymnasium's `Taxi-v3` (with fallback to `Taxi-v4`). The agent learns to navigate a grid, pick up a passenger, and drop them off at the designated destination.

## Q-Learning Rule

The state-action value table $Q(s, a)$ is updated using the standard Temporal Difference equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

## Features

* **Automatic Fallback:** Gracefully handles Gymnasium version deprecations (`Taxi-v3` to `Taxi-v4`).
* **Linear Epsilon Decay:** Gradually transitions from exploration to exploitation.
* **Model Serialization:** Saves trained Q-values for evaluation without re-training.
* **Visualization:** Exports a trailing 100-episode reward plot and an evaluation video.

## Requirements

```bash
pip install gymnasium imageio matplotlib numpy