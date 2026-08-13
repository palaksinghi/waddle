#IMPORTS
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor,VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from envm import OpenDuckBipedalEnv

NUM_ENVS = 8
TOTAL_TIMESTEPS = 8_000_000
LOG_DIR = "logs/open_duck_bipedal"
CKPT_DIR = "checkpoints/open_duck_bipedal"
N_STEPS = 1024  # 1024*8 = 8192
BATCH_SIZE = 1024


def make_env():
    def _init():
        return OpenDuckBipedalEnv()
    return _init


class RewardLoggingCallback(BaseCallback):
    """Logs every individual reward term (averaged across parallel envs)
    to the console and to the SB3 logger (tensorboard / wandb via
    sync_tensorboard), so every reward component's contribution is
    visible and updates live during training."""

    def __init__(self, print_freq=2000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._accum = {}
        self._count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            terms = info.get("reward_terms")
            if terms is None:
                continue
            for k, v in terms.items():
                self._accum[k] = self._accum.get(k, 0.0) + v
        self._count += len(infos)

        if self.num_timesteps % self.print_freq == 0 and self._count > 0:
            print(f"\n[REWARD TERMS] at {self.num_timesteps:,} timesteps (avg over {self._count} env-steps):")
            for k, total in self._accum.items():
                avg = total / self._count
                print(f"    {k:25s}: {avg: .5f}")
                # also push to tensorboard/wandb
                self.logger.record(f"reward_terms/{k}", avg)
            self._accum = {}
            self._count = 0

        return True


class RenderCallback(BaseCallback):

    def __init__(self, render_freq=10_000, render_steps=1000, verbose=0):
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
            print(f"\n[RENDER] Showing policy at {self.num_timesteps:,} timesteps")

            obs, _ = self.eval_env.reset()

            for _ in range(self.render_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02, #13-->0.0005-->0.02
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
        render_steps=500,
    )
    reward_logging_callback = RewardLoggingCallback(
        print_freq=2000,   # print + log every 2000 timesteps
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, reward_logging_callback],   # add render_callback here too if you want live rendering
        progress_bar=True,
    )
    model.save(os.path.join(CKPT_DIR, "final_model"))
    env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))

if __name__ == "__main__":
    main()