"""Render the trained SAC policy reaching on the UR5e -> figures/sac_ur5e_reach.gif."""
import os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "envs"))
import numpy as np, mujoco
from PIL import Image
from stable_baselines3 import SAC
from ur5e_reach import UR5eReachEnv

model = SAC.load(str(REPO / "results" / "sac_ur5e"), device="cpu")
env = UR5eReachEnv()
m, d = env.unwrapped.model, env.unwrapped.data
ren = mujoco.Renderer(m, 480, 480)

def frame():
    ren.update_scene(d, camera="cam"); return ren.render()

gif, dists = [], []
for ep in range(4):
    obs, _ = env.reset(seed=ep)
    info = {"dist": None}
    for _ in range(60):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        gif.append(Image.fromarray(frame()))
        if term or trunc:
            break
    dists.append(info["dist"])
    print(f"ep {ep}: dist final {info['dist']:.3f} m")

out = REPO / "figures" / "sac_ur5e_reach.gif"
gif[0].save(out, save_all=True, append_images=gif[1:], duration=55, loop=0)
print("dist media:", round(float(np.mean(dists)), 3), "m -> saved", out)
