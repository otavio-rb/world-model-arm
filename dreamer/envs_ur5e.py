"""UR5e (Menagerie) reach env for DreamerV3 - state and pixel observations.

Copied into dreamerv3-torch/envs/ur5e.py by setup_models.py. Wraps the repo's
envs/ur5e_reach.py. State variant is fast (no render) for the world model; pixel
variant renders the realistic arm from the fixed 'cam' camera.

Requires the repo's envs/ dir on PYTHONPATH (env var WMA_ENVS, set by setup).
"""
import os
import sys
import gym
import numpy as np
import mujoco

_ENVS = os.environ.get("WMA_ENVS")
if _ENVS and _ENVS not in sys.path:
    sys.path.insert(0, _ENVS)
from ur5e_reach import UR5eReachEnv, N_DOF


class UR5eState:
    metadata = {}

    def __init__(self, size=(64, 64), action_repeat=1, seed=0):
        self._env = UR5eReachEnv()
        self._env.reset(seed=seed)
        self._u = self._env.unwrapped
        self._action_repeat = action_repeat
        self._dim = self._env.observation_space.shape[0]
        self.reward_range = [-np.inf, np.inf]
        self._blank = np.zeros((16, 16, 3), dtype=np.uint8)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (self._dim,), dtype=np.float32),
            "image": gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8)})

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (N_DOF,), dtype=np.float32)

    def _obs(self, vec, is_first, is_terminal):
        return {"state": vec.astype(np.float32), "image": self._blank,
                "is_first": is_first, "is_terminal": is_terminal}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        reward = 0.0
        term = trunc = False
        obs_v = None
        for _ in range(self._action_repeat):
            obs_v, r, term, trunc, _ = self._env.step(action)
            reward += float(r)
            if term or trunc:
                break
        info = {"discount": np.array(1.0, np.float32)}
        return self._obs(obs_v, False, bool(term)), reward, bool(term or trunc), info

    def reset(self):
        obs_v, _ = self._env.reset()
        return self._obs(obs_v, True, False)

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), np.uint8)


class UR5ePixel:
    metadata = {}

    def __init__(self, size=(64, 64), action_repeat=1, seed=0):
        self._env = UR5eReachEnv()
        self._env.reset(seed=seed)
        self._u = self._env.unwrapped
        self._size = size
        self._action_repeat = action_repeat
        self._ren = mujoco.Renderer(self._u.model, size[0], size[1])
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)})

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (N_DOF,), dtype=np.float32)

    def _image(self):
        self._ren.update_scene(self._u.data, camera="cam")
        return self._ren.render().astype(np.uint8)

    def _obs(self, is_first, is_terminal):
        return {"image": self._image(), "is_first": is_first, "is_terminal": is_terminal}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        reward = 0.0
        term = trunc = False
        for _ in range(self._action_repeat):
            _, r, term, trunc, _ = self._env.step(action)
            reward += float(r)
            if term or trunc:
                break
        info = {"discount": np.array(1.0, np.float32)}
        return self._obs(False, bool(term)), reward, bool(term or trunc), info

    def reset(self):
        self._env.reset()
        return self._obs(True, False)

    def render(self, *args, **kwargs):
        return self._image()
