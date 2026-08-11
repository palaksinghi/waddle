import os
import time
import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO
import mujoco.viewer
from envm import OpenDuckBipedalEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="checkpoints/open_duck_bipedal/final_model.zip",
        help="Path to the trained model .zip (e.g. a checkpoint like ppo_duck_4000000_steps.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions (recommended for inference)"
    )
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
        help="Also show the live MuJoCo viewer window (ignored if --save_video is set, since offscreen and passive viewer can't run together cleanly)"
    )
    args = parser.parse_args()

    # device="cpu" here too -- single-env inference, no benefit from GPU
    model = PPO.load(args.model, device="cpu")

    env = OpenDuckBipedalEnv()

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
        obs, info = env.reset()
        if args.cmd is not None:
            env.cmd = np.array(args.cmd, dtype=np.float32)

        ep_reward = 0.0
        step = 0
        terminated = truncated = False
        frames = []

        print(f"\n=== Episode {ep + 1} | cmd={env.cmd} ===")

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            if args.save_video:
                renderer.update_scene(env.data)
                frames.append(renderer.render())
            elif args.live_render:
                env.render()
                time.sleep(env.dt)  # real-time playback

        status = "FELL" if terminated else "TIME LIMIT"
        print(f"Episode {ep + 1} done: steps={step}, reward={ep_reward:.2f}, end={status}")

        if args.save_video and frames:
            out_path = os.path.join(args.video_dir, f"episode_{ep + 1}.mp4")
            fps = int(round(1.0 / env.dt))
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"Saved video: {out_path}")

    env.close()


if __name__ == "__main__":
    main()