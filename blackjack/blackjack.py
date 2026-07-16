import gymnasium as gym
from collections import defaultdict
import numpy as np

env = gym.make("Blackjack-v1", natural=True, sab=True)
obs, info = env.reset()
print(obs, info)
print(env.action_space)       #Player's current sum, dealer's face-up card value, whether the player has a usable ace (True/False)
print(env.observation_space)   
#Two possible actions. `0` = stick, `1` = hit --> discrete(2)

discount_factor = 0.95
epsilon = 0.1
n_episodes = 10000
#The interaction loop:
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward, terminated, truncated)
episode = []

def store_transition(episode, obs, action, reward):
    episode.append((obs, action, reward))    

    done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    store_transition(episode, obs, action, reward)
    done = terminated or truncated

q_values = defaultdict(lambda: np.zeros(env.action_space.n))
#G= reward + discount_factor * G
returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

def update(episode, q_values, returns_sum, returns_count, discount_factor):
    G = 0
    visited = set()
    for obs, action, reward in reversed(episode):
        G = reward + discount_factor * G
        if (obs, action) not in visited:
            visited.add((obs, action))
            returns_sum[obs][action] += G
            returns_count[obs][action] += 1
            q_values[obs][action] = returns_sum[obs][action] / returns_count[obs][action]

def get_action(env, obs, q_values, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_values[obs]))
    
for ep in range(n_episodes):
    obs, info = env.reset()
    episode = []
    done = False
    while not done:
        action = get_action(env, obs, q_values, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        store_transition(episode, obs, action, reward)
        done = terminated or truncated
        obs = next_obs
    update(episode, q_values, returns_sum, returns_count, discount_factor)