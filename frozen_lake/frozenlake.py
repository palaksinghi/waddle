import os
import pickle

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n
print(f"num of states: {n_states}\nnum of actions: {n_actions}")

# ---------------- Training ----------------
Q = np.zeros([n_states, n_actions])

episodes = 1000
alpha = 0.5
gamma = 0.9

for episode in range(1, episodes + 1):
    state = env.reset()[0]
    terminated = False
    truncated = False
    G = 0  # sum of rewards this episode

    while not (terminated or truncated):
        if np.max(Q[state]) > 0:
            action = np.argmax(Q[state])
        else:
            action = env.action_space.sample()

        # step() returns 5 values in this exact order
        new_state, reward, terminated, truncated, info = env.step(action)

        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
        G += reward
        state = new_state

    if episode % 100 == 0:
        print(f"episode {episode} sum of reward: {G}")

print("\nFinal Q-table:")
print(Q)

with open("frozenlake.pkl", "wb") as f:
    pickle.dump(Q, f)
print(f"\nQ-table saved to {os.path.abspath('frozenlake.pkl')}")

# ---------------- Evaluation + video ----------------
frames = []
state = env.reset()[0]
terminated = False
truncated = False

frames.append(env.render())  # initial frame

max_steps = 100  # safety cap in case the policy loops
steps = 0

while not (terminated or truncated) and steps < max_steps:
    if np.max(Q[state]) > 0:
        action = np.argmax(Q[state])
    else:
        action = env.action_space.sample()

    new_state, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    state = new_state
    steps += 1

env.close()

print(f"\nEvaluation finished in {steps} steps. terminated={terminated}, truncated={truncated}")
print(f"Frames collected: {len(frames)}")

if frames:
    

    # also save a gif, handy since it's a short clip and needs no ffmpeg-specific codec
    gif_path = os.path.abspath("frozenlake.gif")
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"GIF saved to: {gif_path}")