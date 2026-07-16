import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
import gymnasium as gym

env=gym.make("FrozenLake-v1",map_name="4x4",render_mode="rgb_array",is_slippery=False)

# env=gym.make(
#     "FrozenLake-v1",
#     desc=None,
#     map_name="4x4",
#     is_slippery=False,
#     render_mode="rgb_array",
#     success_rate=1.0/3.0,
#     reward_schedule=(1, 0, 0)
# )

env.reset()
img=env.render()
plt.imshow(img)
plt.show()


n_states=env.observation_space.n
n_actions=env.action_space.n


print(f" num of states:{n_states}\n num of actions:{n_actions}")

state,info=env.reset()
img=plt.imshow(env.render())
while True:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    img.set_data(env.render())
    display.display(plt.gcf())
    display.clear_output(wait=True)

    if terminated:
            break
    
Q=np.zeros([n_states,n_actions])
Q.shape

episodes=100
alpha=0.5
gamma=0.9
G=0 #G is sum of rewards


for episode in range(1,episodes+1):
  state=env.reset()[0] # I use env.reset()[0] because there two variables are coming from env.reset().
  #The state value is env.reset()[0].You can print to see that what i mean >> print(env.reset())
  done=False
  G=0
  while not done:
    # Select the action that has the highest value in the current state.
    if np.max(Q[state]) > 0:
        action = np.argmax(Q[state])

    # If there's no best action (only zeros), take a random one
    else:
        action = env.action_space.sample()

    new_state,reward,done,info,a=env.step(action)
    Q[state,action]+=alpha*(reward+gamma*np.max(Q[new_state])-Q[state,action])
    G+=reward
    state=new_state
  if episode%100==0:
      print(f"episode {episode} sum of  reward :{G}")



Q
state=env.reset()[0]
done=False

while not done:
  if np.max(Q[state])>0:
    action=np.argmax(Q[state])

  else:
    action=env.action_space.sample()

  new_state,reward,done,info,x=env.step(action)
  img=env.render()
  plt.imshow(img)
  plt.show()
  state=new_state

state=env.reset()[0]
done=False


while not done:
  if np.max(Q[state])>0:
    action=np.argmax(Q[state])

  else:
    action=env.action_space.sample()

  new_state,reward,done,info,x=env.step(action)

  img=plt.imshow(env.render())
  display.display(plt.gcf())
  display.clear_output(wait=True)

  state=new_state