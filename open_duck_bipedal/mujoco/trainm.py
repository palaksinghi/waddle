import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from envm import OpenDuckBipedalEnv

NUM_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
LOG_DIR = "logs/open_duck_bipedal"
CKPT_DIR = "checkpoints/final_combined_v1"
N_STEPS = 1024  # 1024*8 = 8192
BATCH_SIZE = 1024

# FIX: was never set -> SB3 defaults to CPU ("auto" picks CPU for small MLPs
# by default heuristics too). RTX 3050 laptop GPU is small (4GB VRAM) but
# fine for this tiny [512,256,128] MLP policy. NUM_ENVS stays CPU-bound
# (SubprocVecEnv spins physics on CPU cores) -- only the PPO forward/backward
# pass moves to GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        ep_lens = [info["episode"]["l"] for info in infos if "episode" in info]
        if ep_lens:
            avg_len = sum(ep_lens) / len(ep_lens)
            print(f"[EP LENGTH] avg episode length this batch: {avg_len:.1f} steps")
            self.logger.record("custom/ep_len_avg", avg_len)

        if self.num_timesteps % self.print_freq == 0 and self._count > 0:
            print(f"\n[REWARD TERMS] at {self.num_timesteps:,} timesteps (avg over {self._count} env-steps):")
            for k, total in self._accum.items():
                avg = total / self._count
                print(f"    {k:25s}: {avg: .5f}")
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
                obs, rewards, terminated, truncated, info = self.eval_env.step(action)

                self.eval_env.render()
                time.sleep(0.01)

                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
        return True

    def _on_training_end(self):
        if self.eval_env is not None:
            self.eval_env.close()


class VecNormalizeCheckpointCallback(BaseCallback):
    """FIX: CheckpointCallback only saves the PPO model weights, not the
    VecNormalize running stats. Testing an intermediate checkpoint with the
    final/stale vecnormalize.pkl feeds the policy wrongly-scaled observations
    -> policy collapses to a frozen/degenerate action (identical reward,
    zero displacement, every episode -- exactly what you're seeing). This
    saves a matching vecnormalize_<step>.pkl alongside every model checkpoint."""

    def __init__(self, save_freq, save_path, name_prefix="vecnormalize", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(path)
            if self.verbose:
                print(f"Saved VecNormalize stats to {path}")
        return True


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
        device=DEVICE,
    )
    print(f">>> Training on device: {DEVICE}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // NUM_ENVS, 8),
        save_path=CKPT_DIR,
        name_prefix="ppo_duck",
    )
    vecnorm_checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=max(500_000 // NUM_ENVS, 8),
        save_path=CKPT_DIR,
        verbose=1,
    )
    render_callback = RenderCallback(
        render_freq=50_000,
        render_steps=500,
    )
    reward_logging_callback = RewardLoggingCallback(
        print_freq=2000,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, vecnorm_checkpoint_callback, reward_logging_callback, render_callback],
        progress_bar=True,
    )
    model.save(os.path.join(CKPT_DIR, "final_model"))
    env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))


if __name__ == "__main__":
    main()