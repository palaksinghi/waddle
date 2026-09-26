"""Open Duck Bipedal env (vendored: envm.py, rewardsm.py, robots/)."""

import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

from flash_rl.types import F32NDArray
from .envm import OpenDuckBipedalEnv

__all__ = ["OpenDuckBipedalEnv", "make_duck_env"]

_VENDORED_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robots/open_duck_mini_v2/scene.xml")


def make_duck_env(
    env_name: str,
    seed: int,
    **kwargs: object,
) -> gym.Env[F32NDArray, F32NDArray]:
    if os.path.exists(_VENDORED_XML):
        xml_path = _VENDORED_XML
    else:
        # legacy fallback: external repo via OPEN_DUCK_BIPEDAL_PATH
        _DUCK_REPO_PATH = os.environ.get(
            "OPEN_DUCK_BIPEDAL_PATH",
            os.path.expanduser("~/waddle-palak/open_duck_bipedal/mujoco"),
        )
        if _DUCK_REPO_PATH not in sys.path:
            sys.path.insert(0, _DUCK_REPO_PATH)
        from envm import OpenDuckBipedalEnv as _ExternalEnv  # type: ignore

        xml_path = os.path.join(_DUCK_REPO_PATH, "robots/open_duck_mini_v2/scene.xml")
        env = _ExternalEnv(xml_path=xml_path, render_mode="rgb_array")
        env.reset(seed=seed)
        return _attach_offscreen_renderer(env)

    env = OpenDuckBipedalEnv(xml_path=xml_path, render_mode="rgb_array")
    env.reset(seed=seed)
    return _attach_offscreen_renderer(env)


def _attach_offscreen_renderer(env):  # type: ignore[no-untyped-def]
    # Upstream OpenDuckBipedalEnv.render() launches an interactive viewer and
    # returns None, but FlashSAC's record_video() needs rgb_array frames.
    # Attach an offscreen renderer without touching the vendored env.
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    orig_close = env.close

    def _render() -> np.ndarray:
        renderer.update_scene(env.data)
        return renderer.render()

    def _close() -> None:
        try:
            renderer.close()
        except Exception:
            pass
        orig_close()

    env.render = _render  # type: ignore[method-assign]
    env.close = _close  # type: ignore[method-assign]
    return env
