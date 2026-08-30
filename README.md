# WADDLE
Waddle involves learning Reinforcement Learning, Deep Learning, Bipedal Locomotion, and Robotic Simulation.

Robots don’t intuitively walk; they’re taught using a policy. Waddle teaches you how to build that policy. You’ll learn to implement reinforcement learning algorithms by solving mini environments. Then you will apply them in MuJoCo and Isaac Lab to train Open Duck, an open-source bipedal robot.  The final stretch benchmarks PPO against SAC on the same robot, adds velocity-command control, and stress-tests the policy with pushes and rough terrain.

# Overview 

<p align="center">
  <img src="./FlashSAC/duck.png" width="350" alt="Open Duck Mini v2"/>
</p>

Open Duck Mini v2 features 17 Degrees of Freedom (DoF). To achieve stable, autonomous walking, we applied Proximal Policy Optimization (PPO) and Flash Soft Actor-Critic (FlashSAC). The walking gait is implemented using detailed RL reward engineering—combining, weighting, and fine-tuning distinct reward components to ensure proper balance and gait stability.


<p align="center">
  <img src="./gif_collection/flash_sac.gif" alt="FlashSAC Open Duck Demo" width="100%"/>
</p>


---
---

| Frozen Lake | Lunar Lander |
| :---: | :---: |
| ![frozen lake](gif_collection/frozenlake.gif) |![lunar lander](gif_collection/lunar_lander_trained.gif) |
| taxi |
### Taxi-v3

<p align="center">
  <img src="gif_collection/taxi.gif" alt="Taxi Environment" width="85%"/>
</p>


## 📁 Repository Structure

```text
.
├── FlashSAC/            
├── frozenlake/          
├── halfcheetah_ppo/     
├── humanoid/            
├── lunar landing/ 
├── multi_humanoid/      
├── neural network/      
├── taxi/                
└── walker_2d/

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/palaksinghi/waddle.git
cd waddle
```

### Create a Virtual Environment

```bash
python3.10 -m venv waddle_env
```

### Activate the Virtual Environment

**macOS/Linux:**

```bash
source waddle_env/bin/activate
```

**Windows (PowerShell):**

```bash
.\waddle_env\Scripts\activate
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.10+
- MuJoCo physics engine
- PyTorch/TensorFlow for deep learning
- Stable Baselines3 for RL algorithms
- Additional dependencies listed in `requirements.txt`