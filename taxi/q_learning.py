import os
import pickle

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np


def make_taxi_env(render_mode=None):
    """Taxi-v3 is deprecated in newer gymnasium versions; fall back to v4."""
    try:
        return gym.make('Taxi-v3', render_mode=render_mode)
    except gym.error.DeprecatedEnv:
        return gym.make('Taxi-v4', render_mode=render_mode)


def run(episodes, is_training=True, render=False):

    render_mode = 'rgb_array' if render else None
    env = make_taxi_env(render_mode=render_mode)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # 500 x 6
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9        # alpha
    discount_factor_g = 0.9      # gamma
    epsilon = 0.1               #
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    frames = []  # collected only when render=True

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        if render:
            frames.append(env.render())

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # 0=south,1=north,2=east,3=west,4=pickup,5=dropoff
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

            if render:
                frames.append(env.render())

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

        if is_training and (i + 1) % 1000 == 0:
            print(f"Episode {i + 1}/{episodes}  avg reward (last 100): "
                  f"{np.mean(rewards_per_episode[max(0, i - 99):i + 1]):.2f}")

    env.close()

    # Plot rolling reward
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.figure()
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards (trailing 100 episodes)')
    plt.title('Taxi-v3 Q-learning training progress')
    plt.savefig('taxi.png')
    plt.close()

    if is_training:
        with open('taxi.pkl', 'wb') as f:
            pickle.dump(q, f)
        print(f"Q-table saved to {os.path.abspath('taxi.pkl')}")
        print(f"Training plot saved to {os.path.abspath('taxi.png')}")

    print(f"Current working directory: {os.getcwd()}")
    print(f"render={render}, frames collected: {len(frames)}")

    if render and frames:
        output_path = os.path.abspath('taxi.mp4')
        imageio.mimsave(output_path, frames, fps=4)
        print(f"Video saved to: {output_path}")
    elif render:
        print("WARNING: render=True but no frames were captured — video not saved.")


if __name__ == '__main__':
    run(15000, is_training=True, render=False)
    run(10, is_training=False, render=True)