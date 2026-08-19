import os
import time
import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import mujoco.viewer
from envm import OpenDuckBipedalEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="checkpoints/final_combined_v1/final_model.zip",
        help="Path to the trained model .zip (e.g. a checkpoint like ppo_duck_500000_steps.zip)"
    )
    parser.add_argument(
        "--vecnormalize", type=str, default="checkpoints/final_combined_v1/vecnormalize.pkl",
        help="Path to the saved VecNormalize stats -- REQUIRED to match training-time obs scale"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--deterministic", dest="deterministic", action="store_true",
        help="Use deterministic actions (recommended for inference, default on)"
    )
    parser.add_argument(
        "--stochastic", dest="deterministic", action="store_false",
        help="Sample actions stochastically instead of deterministically"
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--cmd", type=float, nargs=3, default=None,
        help="Fixed command [vx, vy, wz] to override random sampling, e.g. --cmd 0.3 0 0"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Save each episode as an mp4 instead of (or alongside) live viewer rendering"
    )
    parser.add_argument(
        "--video_dir", type=str, default="inference_videos",
        help="Directory to save videos into"
    )
    parser.add_argument(
        "--live_render", action="store_true",
        help="Also show the live MuJoCo viewer window (ignored if --save_video is set)"
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")

    raw_env = DummyVecEnv([lambda: OpenDuckBipedalEnv()])
    if os.path.exists(args.vecnormalize):
        vec_env = VecNormalize.load(args.vecnormalize, raw_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {args.vecnormalize}")
    else:
        vec_env = raw_env
        print(f"WARNING: {args.vecnormalize} not found -- running WITHOUT obs normalization. "
              f"Results will not match training behavior.")

    env = vec_env.envs[0]  # underlying OpenDuckBipedalEnv, for direct access (render, cmd, qpos)

    renderer = None
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            raise SystemExit(
                "imageio is required for video saving. Install with: pip install imageio imageio-ffmpeg"
            )
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    for ep in range(args.episodes):
        obs = vec_env.reset()
        if args.cmd is not None:
            env.cmd = np.array(args.cmd, dtype=np.float32)

        ep_reward = 0.0
        step = 0
        terminated = truncated = False
        frames = []
        start_x = env.data.qpos[0]

        print(f"\n=== Episode {ep + 1} | cmd={env.cmd} ===")
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            terminated = bool(infos[0].get("terminated", False))
            truncated = bool(infos[0].get("TimeLimit.truncated", False)) or (dones[0] and not terminated)
            ep_reward += reward[0]
            step += 1

            if args.save_video:
                renderer.update_scene(env.data)
                frames.append(renderer.render())
            elif args.live_render:
                env.render()
                time.sleep(env.dt)

        status = "FELL" if terminated else "TIME LIMIT"
        end_x = env.data.qpos[0]
        net_displacement = end_x - start_x
        print(f"Episode {ep + 1} done: steps={step}, reward={ep_reward:.2f}, end={status}, forward_dist={net_displacement:.3f}m")

        if args.save_video and frames:
            out_path = os.path.join(args.video_dir, f"episode_{ep + 1}.mp4")
            fps = int(round(1.0 / env.dt))
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"Saved video: {out_path}")

    env.close()


if __name__ == "__main__":
    main()