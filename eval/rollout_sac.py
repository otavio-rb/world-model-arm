"""Render the trained SAC policy reaching on the UR5e -> figures/sac_ur5e_reach.gif.

Applies action smoothing (EMA) + a freeze once on target, so the rendered arm
moves smoothly instead of jittering around the goal.
"""
import os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "envs"))
import numpy as np, mujoco
from PIL import Image
from stable_baselines3 import SAC
from ur5e_reach import UR5eReachEnv

# fixed, camera-facing, reachable targets so the ball is always visible and reached
TARGETS = [[0.42, -0.05, 0.40], [0.40, -0.28, 0.30], [0.35, -0.15, 0.38], [0.48, -0.10, 0.34]]
EMA = 0.5                  # light action low-pass (reaches well, less jitter)
model = SAC.load(str(REPO / "results" / "sac_ur5e"), device="cpu")
env = UR5eReachEnv()
m, d = env.unwrapped.model, env.unwrapped.data
ren = mujoco.Renderer(m, 480, 480)

def frame():
    ren.update_scene(d, camera="cam"); return ren.render()

gif, dists, jitter = [], [], []
for ep in range(4):
    env.reset(seed=ep)
    tgt = np.array(TARGETS[ep % len(TARGETS)]); env._target = tgt; d.mocap_pos[0] = tgt
    mujoco.mj_forward(m, d); obs = env._get_obs()
    a_smooth, a_hold, near = None, None, 0
    info = {"dist": None}
    for _ in range(80):
        a, _ = model.predict(obs, deterministic=True)
        a_smooth = a if a_smooth is None else EMA * a_smooth + (1 - EMA) * a
        dist = float(np.linalg.norm(env._ee() - env._target))
        if dist < 0.05:                         # reached -> settle a few frames, then freeze
            near += 1
            if near >= 3 and a_hold is None:
                a_hold = a_smooth.copy()
        else:
            near = 0; a_hold = None              # drifted out -> re-engage
        cmd = a_hold if a_hold is not None else a_smooth
        obs, r, term, trunc, info = env.step(cmd)
        gif.append(Image.fromarray(frame()))
        if dist < 0.1:                           # tremble proxy: joint speed near target
            jitter.append(float(np.linalg.norm(d.qvel[:6])))
        if term or trunc:
            break
    dists.append(info["dist"])
    print(f"ep {ep}: dist final {info['dist']:.3f} m")
print("tremor (|qvel| medio perto do alvo, menor=melhor):", round(float(np.mean(jitter)), 3))

out = REPO / "figures" / "sac_ur5e_reach.gif"
gif[0].save(out, save_all=True, append_images=gif[1:], duration=55, loop=0)
print("dist media:", round(float(np.mean(dists)), 3), "m -> saved", out)
