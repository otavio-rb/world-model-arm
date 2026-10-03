"""Franka Panda pick-up task: reach a cube, close the gripper, lift it.

Contact-rich, multi-stage, near-sparse — the kind of task where a world model's
lookahead should beat model-free RL. Action = 7 arm joint targets + 1 gripper
(open/close). Staged reward: reach -> close-when-near -> lift.
"""
import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MEN = os.environ.get("WMA_MENAGERIE", os.path.join(_REPO, "menagerie"))
XML = os.path.join(_MEN, "franka_emika_panda", "grasp_task.xml")

HOME_ARM = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
SWING = np.array([1.6, 1.2, 1.6, 1.2, 1.8, 1.6, 1.8])      # per-joint action range around home
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
BOX_Z0 = 0.03
LIFT_OK = 0.12                                              # box centre height = success
# pre-grasp arm pose (gripper down over the workspace) for the curriculum
PREGRASP = np.array([0.2897, 0.50732, -0.140016, -2.176, -0.0310497, 2.51592, -0.49251])
CURRICULUM_P = 0.6                                          # frac of episodes starting box-in-gripper


class PandaGraspEnv(MujocoEnv):
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, **kwargs):
        obs_dim = 7 + 7 + 1 + 3 + 3 + 3   # armq, armv, grip, grasp_pt, box, box-grasp
        obs_space = Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self._t = 0
        MujocoEnv.__init__(self, XML, 10, observation_space=obs_space, **kwargs)
        self.action_space = Box(-1.0, 1.0, (8,), np.float32)
        self._alo = self.model.actuator_ctrlrange[:7, 0]
        self._ahi = self.model.actuator_ctrlrange[:7, 1]
        self._hand = self.model.body("hand").id
        self._home_key = self.model.key("home").id

    def _grasp_pt(self):
        R = self.data.xmat[self._hand].reshape(3, 3)
        return self.data.xpos[self._hand] + R @ np.array([0, 0, 0.103])

    def _box(self):
        return self.data.qpos[9:12].copy()

    def _ctrl(self, action):
        a = np.clip(np.asarray(action), -1, 1)
        arm = np.clip(HOME_ARM + a[:7] * SWING, self._alo, self._ahi)
        grip = (a[7] + 1) / 2 * GRIP_OPEN            # -1 closed, +1 open
        return np.concatenate([arm, [grip]])

    def _get_obs(self):
        gp, box = self._grasp_pt(), self._box()
        return np.concatenate([self.data.qpos[:7], self.data.qvel[:7],
                               [self.data.qpos[7]], gp, box, box - gp]).astype(np.float64)

    def step(self, action):
        self.do_simulation(self._ctrl(action), self.frame_skip)
        gp, box = self._grasp_pt(), self._box()
        d = float(np.linalg.norm(gp - box))
        grip_open = float(self.data.qpos[7])          # 0 closed .. 0.04 open
        lifted = max(0.0, box[2] - BOX_Z0)
        aligned = d < 0.035
        closed_frac = (GRIP_OPEN - grip_open) / GRIP_OPEN     # 0 open .. 1 closed
        grasped = aligned and closed_frac > 0.5              # box in a closed gripper
        reach_r = 1.0 - np.tanh(10.0 * d)             # sharper: strong pull to the box
        success = box[2] > LIFT_OK
        # lifting DOMINATES so "grab and hold low" is never as good as lifting
        reward = (2.0 * reach_r + 1.0 * float(grasped) + 80.0 * lifted
                  + (30.0 if success else 0.0) - 0.001 * float(np.square(action).sum()))
        self._t += 1
        fell = box[2] < -0.05
        trunc = self._t >= 250
        return self._get_obs(), reward, fell, trunc, {"dist": d, "box_z": float(box[2]), "success": success}

    def reset_model(self, force_curriculum=None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        qpos = self.data.qpos.copy()
        curr = (np.random.rand() < CURRICULUM_P) if force_curriculum is None else force_curriculum
        if curr:
            # CURRICULUM: gripper open in a pre-grasp pose, box placed between the
            # fingers -> policy just has to close + lift (sees the lift reward often).
            qpos[:7] = PREGRASP + np.random.uniform(-0.03, 0.03, 7)
            qpos[7] = qpos[8] = GRIP_OPEN
            self.set_state(qpos, np.zeros(self.model.nv))
            mujoco.mj_forward(self.model, self.data)
            gp = self._grasp_pt()
            qpos = self.data.qpos.copy()
            qpos[9:12] = gp                            # box exactly between the fingers
            qpos[12:16] = [1, 0, 0, 0]
            self.set_state(qpos, np.zeros(self.model.nv))
        else:
            qpos[:7] += np.random.uniform(-0.05, 0.05, 7)
            qpos[9] = np.random.uniform(0.38, 0.60)   # box x
            qpos[10] = np.random.uniform(-0.22, 0.22)  # box y
            qpos[11] = BOX_Z0
            qpos[12:16] = [1, 0, 0, 0]
            self.set_state(qpos, np.zeros(self.model.nv))
        self._t = 0
        return self._get_obs()


if __name__ == "__main__":
    env = PandaGraspEnv()
    env.reset(seed=0)
    print("obs", env.observation_space.shape, "act", env.action_space.shape,
          "nu", env.model.nu, "nq", env.model.nq)
    print("grasp_pt:", np.round(env._grasp_pt(), 3), " box:", np.round(env._box(), 3),
          " dist:", round(float(np.linalg.norm(env._grasp_pt()-env._box())), 3))
    r = 0.0
    for _ in range(50):
        o, rew, term, trunc, info = env.step(env.action_space.sample())
        r += rew
    print("50 passos random: reward", round(r, 1), "box_z", round(info["box_z"], 3), "dist", round(info["dist"], 3))
