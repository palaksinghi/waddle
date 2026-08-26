"""Custom ManagerBasedRLEnv for open_duck_mini_v2.

Adds two lightweight per-env buffers that both the observation manager and the
reward terms read:
  - self.gait_phase        (N,) running phase clock in [0, 2*pi), advanced by
                            gait_cycle_dt each physics step.
  - self.episode_start_pos (N, 3) root world position captured at each reset,
                            used for forward_progress_reward / straight_path_penalty.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class DuckGaitEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.gait_cycle_dt = 2.0 * torch.pi * (cfg.sim.dt * cfg.decimation) / cfg.gait_cycle_period_s
        self.gait_phase = torch.zeros(num_envs, device=device)
        self.episode_start_pos = torch.zeros(num_envs, 3, device=device)

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # now that the scene exists, capture the real starting root position
        self.episode_start_pos = self.scene["robot"].data.root_pos_w.clone()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.gait_phase[env_ids] = 0.0
        self.episode_start_pos[env_ids] = self.scene["robot"].data.root_pos_w[env_ids].clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        self.gait_phase = torch.remainder(self.gait_phase + self.gait_cycle_dt, 2.0 * torch.pi)