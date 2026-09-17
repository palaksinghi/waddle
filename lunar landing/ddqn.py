import gymnasium as gym
from collections import deque
import random
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import time


SEED=12
random.seed(SEED)

env=gym.make("LunarLander-v3",render_mode='human')

CHECKPOINT_PATH = "ddqn_lunarlander.pt"
VIDEO_PATH = "lunar_lander_trained2.mp4"

'''
device
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

'''
replay buffer
'''
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class replay_buffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
'''
DQN network
'''
class DDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
'''
hyperparameters
'''

BATCH_SIZE =64
TAU = 0.005
LR = 1e-4
BUFFER_SIZE =100_000
MIN_BUFFER_SIZE = 2_000
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99        # multiplicative decay per episode
TARGET_UPDATE_EVERY = 1    # soft-update every learn step
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 3000
GRAD_CLIP = 10

'''
DQQN agent
'''

class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.q_network=DDQN(state_dim,action_dim).to(device)
        self.target_network=DDQN(state_dim,action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer=optim.Adam(self.q_network.parameters(),lr=LR,amsgrad=True)
        self.replay_buffer=replay_buffer(BUFFER_SIZE)
        self.step_count=0
    
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    def learn(self):
        if len(self.replay_buffer) < max(BATCH_SIZE, MIN_BUFFER_SIZE):
            return None
        transitions=self.replay_buffer.sample(BATCH_SIZE)
        batch=Transition(*zip(*transitions))
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
        #online network
        current_q =self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_action =self.q_network(next_states).argmax(dim=1, keepdim=True)
            #target_network
            next_q = self.target_network(next_states).gather(1, next_action)
            target=reward+GAMMA*next_q*(1-dones)
        
        self.optimizer.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(current_q, target)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(),GRAD_CLIP)
        self.optimizer.step()
        self._soft_update_target()
        return loss.item()
    
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def _soft_update_target(self):
        for target_param,local_param in zip(self.target_network.parameters(),self.q_network.parameters()):
            target_param.data.copy_(
                TAU*local_param.data+(1.0-TAU)*target_param.data
            )

    def save(self,path:str):
        torch.save({
            "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
            },
            path,
        )
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

'''
training
'''

def train():
    random.seed(SEED)
    env.reset(seed=SEED)
    reward=0
    all_score=[]
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    agent=DDQNAgent(state_dim,action_dim)

    epsilon = EPS_START
    scores = deque(maxlen=100)
    start_time=time.time()

    for e in range(1,NUM_EPISODES):
        state,_=env.reset()
        episode_reward=0.0
        episode_losses=[]
        env.render()

        for t in range(MAX_STEPS_PER_EPISODE):
            action=agent.act(state,epsilon)
            next_state,reward,terminated,truncated,_=env.step(action)
            done=truncated or terminated
            agent.store(state, action, reward, next_state, done)

            loss=agent.learn()
            if loss is not None:
                episode_losses.append(loss)
            state=next_state
            episode_reward+=reward
            agent.step_count+=1

            if done:
                break
        scores.append(episode_reward)
        all_score.append(episode_reward)
        epsilon=max(EPS_END,EPS_DECAY*epsilon)

        avg_score=np.mean(scores)
        avg_loss=np.mean(episode_losses)
        elapsed = time.time() - start_time
        print(
            f"Episode {e:4d}/{NUM_EPISODES} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg(100): {avg_score:8.2f} | "
            f"Loss: {avg_loss:8.4f} | "
            f"Eps: {epsilon:5.3f} | "
            f"Steps: {t+1:4d} | "
            f"Elapsed: {elapsed:6.1f}s"
        )

    env.close()
    agent.save(CHECKPOINT_PATH)
    return agent, all_score

# ----------------------------------------------------------------------
# Evaluation + video rendering
# ----------------------------------------------------------------------
def record_video(agent: DDQNAgent, num_episodes: int = 3, out_path: str = VIDEO_PATH):
    env = gym.make("LunarLander-v3", render_mode="human")
    frames = []
    episode_scores = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + ep)
        done = False
        ep_reward = 0.0

        while not done:
            frames.append(env.render())
            action = agent.act(state, epsilon=0.0)  # greedy, no exploration
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        episode_scores.append(ep_reward)

    env.close()

    imageio.mimsave(out_path, frames, fps=30)
    print(f"Saved video to {out_path}")
    print(f"Evaluation scores: {episode_scores}")


if __name__ == "__main__":
    trained_agent, history = train()
    record_video(trained_agent, num_episodes=5)
