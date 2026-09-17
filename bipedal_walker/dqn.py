import gymnasium as gym
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import numpy as np
import imageio
CHECKPOINT_PATH = "dqn_bipedal.pt"
VIDEO_PATH = "bipedal_walker_trained.mp4"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)
env = gym.make("BipedalWalker-v3", render_mode="human")
'''
replay buffer
'''
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size )
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor( np.array(states), dtype=torch.float32,device=DEVICE )
        actions = torch.as_tensor(np.array(actions),dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.as_tensor(np.array(rewards),dtype=torch.float32,device=DEVICE).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states),dtype=torch.float32,device=DEVICE)
        dones = torch.as_tensor(np.array(dones),dtype=torch.float32,device=DEVICE).unsqueeze(1)
        return states,actions,rewards,next_states,dones
    def __len__(self):
        return len(self.buffer)
'''
DQN
'''
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)
# Discretized action space
action_table = list(product([-1.0, 0.0, 1.0],repeat=4))
action_dim = len(action_table)      # 81
state_dim = env.observation_space.shape[0]
# Networks
q_network = QNetwork(state_dim,action_dim).to(DEVICE)
target_network = QNetwork(state_dim,action_dim).to(DEVICE)
target_network.load_state_dict(q_network.state_dict())
# Replay Buffer
replay_buffer = ReplayBuffer(100000)
# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
batch_size = 64
optimizer = optim.Adam(
    q_network.parameters(),
    lr=1e-4,
    amsgrad=True
)
loss_fn = nn.SmoothL1Loss()
# Training
num_episodes = 500
episode_rewards = []
last_loss = 0
print("Starting training...")
for episode in range(1,num_episodes + 1):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated):
        # Epsilon Greedy
        if random.random() < epsilon:
            action_idx = random.randint(0,action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action_idx = torch.argmax(q_network(state_tensor)).item()   #int conversion by suing item
        action = action_table[action_idx]
        next_state, reward, terminated, truncated, info = env.step(action)
        reward = np.clip(reward, -1.0, 1.0)  
        done = terminated or truncated
        replay_buffer.push(state,action_idx,reward,next_state,done)
        state = next_state
        total_reward += reward
       # Learn

        if len(replay_buffer) >= batch_size:

            (states, actions,rewards,next_states,dones) = replay_buffer.sample(batch_size)
            current_q = q_network(states).gather(1,actions)

            with torch.no_grad():
                next_q = target_network(next_states).max(dim=1,keepdim=True)[0]

            target_q = (rewards+ gamma * next_q * (1 - dones))

            loss = loss_fn(current_q,target_q)
            last_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(),1.0)
            optimizer.step()
            tau = 0.005
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    # Episode End
    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % 10 == 0:   #10 change>>>5
        avg_reward = np.mean(episode_rewards[-10:])
        print(
            f"Episode {episode} | "
            f"Avg Reward = {avg_reward:.2f} | "
            f"Loss = {last_loss:.4f} | "
            f"Epsilon = {epsilon:.3f}"
        )
        # target_network.load_state_dict(q_network.state_dict())
# Save checkpoint so you don't lose the trained weights
torch.save(q_network.state_dict(), CHECKPOINT_PATH)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")
# Results
print("Last 100 Episode Average Reward =",np.mean(episode_rewards[-100:]))
# Evaluation
state, info = env.reset()
terminated = False
truncated = False
eval_reward = 0
while not (terminated or truncated):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action_idx = torch.argmax(q_network(state_tensor)).item()
    action = action_table[action_idx]
    state, reward, terminated, truncated, info = env.step(action)
    eval_reward += reward
print("Evaluation Reward =", eval_reward)
env.close()
# Evaluation + video rendering
def record_video(network:nn.Module, num_episodes: int = 3, out_path: str = VIDEO_PATH):
    video_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    frames = []
    episode_scores = []
    for ep in range(num_episodes):
        state, _ = video_env.reset(seed=42 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            frames.append(video_env.render())
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action_idx = torch.argmax(network(state_tensor)).item()
            action = action_table[action_idx]
            # action = agent.act(state, epsilon=0.0)  # greedy, no exploration
            state, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_scores.append(ep_reward)
    video_env.close()
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Saved video to {out_path}")
    print(f"Evaluation scores over video episodes: {episode_scores}")
if __name__ == "__main__":
    # trained_agent, history = train()
    record_video(q_network, num_episodes=3)