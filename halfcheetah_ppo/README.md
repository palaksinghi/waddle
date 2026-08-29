# Custom HalfCheetah PPO

The environment rewards forward velocity and staying healthy, while penalizing
large motor commands:

`reward = 1.0 * x_velocity + 1.0 * healthy - 0.1 * sum(action^2)`

Healthy means torso height is in `[0.25, 1.0]` metres and torso pitch is in
`[-1.0, 1.0]` radians. An unhealthy cheetah ends the episode.

Install and train:

```bash
cd /Users/hindrajprakashmali/Documents/Codex/2026-07-27/no/outputs/halfcheetah_ppo
pip install -r requirements.txt
python train.py --total-timesteps 1000000
```

To open the live MuJoCo viewer after training, add `--render`:

```bash
python train.py --total-timesteps 1000000 --render
```

`train.py` uses a `SyncVectorEnv` with eight environments by default. It includes
observation normalization/clipping, discounted-return reward normalization and
reward clipping, value-target normalization, PPO value clipping, and gradient
norm clipping. Checkpoints are written to `checkpoints/halfcheetah_ppo.pt`.
