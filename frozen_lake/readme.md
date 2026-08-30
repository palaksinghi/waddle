# FrozenLake Q-Learning

<img src="../gif_collection/frozenlake.gif" alt="FlashSAC Open Duck Demo" width="100%"/>

A tabular Q-learning implementation for Gymnasium's non-slippery 4x4 `FrozenLake-v1` environment. The agent learns an optimal path from the starting position `(S)` to the goal `(G)` while avoiding holes `(H)`.

## Environment Overview

The environment consists of a $4 \times 4$ grid (16 states) and 4 discrete actions (0: Left, 1: Down, 2: Right, 3: Up):

```text
SFFF  (S: Start, safe)
FHFH  (F: Frozen, safe)
FFFH  (H: Hole, terminal state)
HFFG  (G: Goal, terminal state, reward +1)