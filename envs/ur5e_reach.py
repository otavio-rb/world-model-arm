"""UR5e (MuJoCo Menagerie) reaching task — realistic 6-DOF industrial arm.

Same interface as arm_env.ArmEnv (obs 21-d, action 6-d) so it drops into the SAC
demos and the DreamerV3 world-model pipeline. The policy commands joint position
targets (around the home pose); reward = -distance(end-effector, target).
"""
import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MEN = os.environ.get("WMA_MENAGERIE", os.path.join(_REPO, "menagerie"))
XML = os.path.join(_MEN, "universal_robots_ur5e", "reach_task.xml")

N_DOF = 6
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
# per-joint swing (rad) the policy can command around home (clamped to limits)
SWING = np.array([2.6, 1.8, 1.8, 3.0, 3.0, 3.0])
SHOULDER = np.array([0.0, 0.0, 0.163])      # shoulder pivot in world
REACH_LO, REACH_HI = 0.35, 0.70


class UR5eReachEnv(MujocoEnv):
    metadata = {"render_modes": [], "render_fps": 25}

    def __init__(self, **kwargs):
        obs_dim = N_DOF + N_DOF + 3 + 3 + 3
        obs_space = Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self._t = 0
        self._target = np.array([0.4, 0.3, 0.5])
        MujocoEnv.__init__(self, XML, 20, observation_space=obs_space, **kwargs)
        self.action_space = Box(-1.0, 1.0, (N_DOF,), np.float32)
        self._lo = self.model.actuator_ctrlrange[:, 0]
        self._hi = self.model.actuator_ctrlrange[:, 1]
        self._eid = self.model.site("attachment_site").id

    def _ctrl_from_action(self, action):
        a = np.clip(np.asarray(action), -1, 1)
        return np.clip(HOME + a * SWING, self._lo, self._hi)

    def _ee(self):
        return self.data.site_xpos[self._eid].copy()

    def _sample_target(self):
        while True:
            v = np.random.randn(3); v /= np.linalg.norm(v) + 1e-9
            p = SHOULDER + v * np.random.uniform(REACH_LO, REACH_HI)
            if p[2] > 0.15:
                return p

    def _get_obs(self):
        ee = self._ee()
        return np.concatenate([self.data.qpos[:N_DOF], self.data.qvel[:N_DOF],
                               ee, self._target, ee - self._target]).astype(np.float64)

    def step(self, action):
        self.do_simulation(self._ctrl_from_action(action), self.frame_skip)
        dist = float(np.linalg.norm(self._ee() - self._target))
        reached = dist < 0.05
        reward = -dist + (1.0 if reached else 0.0) - 0.002 * float(np.square(action).sum())
        self._t += 1
        trunc = self._t >= 200
        return self._get_obs(), reward, False, trunc, {"dist": dist, "reached": reached}

    def reset_model(self):
        qpos = HOME + np.random.uniform(-0.1, 0.1, N_DOF)
        qvel = np.zeros(N_DOF)
        self.set_state(np.concatenate([qpos, self.init_qpos[N_DOF:]]),
                       np.concatenate([qvel, self.init_qvel[N_DOF:]]))
        self._target = self._sample_target()
        self.data.mocap_pos[0] = self._target
        self._t = 0
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()


if __name__ == "__main__":
    env = UR5eReachEnv()
    env.reset(seed=0)
    print("obs", env.observation_space.shape, "act", env.action_space.shape,
          "nu", env.model.nu, "ngeom", env.model.ngeom)
    # range of ee over random actions (is the workspace covered?)
    ees = []
    for _ in range(40):
        env.reset()
        for _ in range(30):
            env.step(env.action_space.sample())
        ees.append(env._ee())
    ees = np.array(ees)
    print("ee min", np.round(ees.min(0), 2), "max", np.round(ees.max(0), 2))
    print("dist alvo inicial exemplo:", round(float(np.linalg.norm(env._ee()-env._target)), 3))
