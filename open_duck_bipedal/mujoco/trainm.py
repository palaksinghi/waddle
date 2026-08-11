#IMPORTS
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback ,BaseCallback

from envm import OpenDuckBipedalEnv

NUM_ENVS = 8
TOTAL_TIMESTEPS = 8_000_000
LOG_DIR = "logs/open_duck_bipedal"
CKPT_DIR = "checkpoints/open_duck_bipedal"
N_STEPS=1024  #1024*8=8192
BATCH_SIZE=1024

def make_env():
    def _init():
        return OpenDuckBipedalEnv()
    return _init

class RenderCallback(BaseCallback):

    def __init__(self, render_freq=10_000, render_steps=1000, verbose=0):  #OVERWRITE
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_steps = render_steps
        self.eval_env = None

    def _on_training_start(self):
        print(">>> Render callback started")
        self.eval_env = OpenDuckBipedalEnv()
        self.eval_env.reset()

    def _on_step(self):

        if self.num_timesteps % self.render_freq == 0:

            print(f"\n[RENDER] Showing policy at "f"{self.num_timesteps:,} timesteps")

            obs, _ = self.eval_env.reset()

            for _ in range(self.render_steps):

                action, _ = self.model.predict(obs,deterministic=True)
                obs,reward, terminated, truncated, info =self.eval_env.step(action)

                self.eval_env.render()

                time.sleep(0.01)

                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
        return True

    def _on_training_end(self):
        if self.eval_env is not None:
            self.eval_env.close()

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE, #changing the batch_ize from 4096-->512
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // NUM_ENVS, 8),
        save_path=CKPT_DIR,
        name_prefix="ppo_duck",
    )
    render_callback = RenderCallback(
    render_freq=50_000,
    render_steps=500
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)
    #model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, render_callback], progress_bar=True)
    model.save(os.path.join(CKPT_DIR, "final_model"))


if __name__ == "__main__":
    main()