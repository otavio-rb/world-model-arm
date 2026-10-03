"""Render the trained Dreamer (world model) policy reaching -> figures/dreamer_ur5e_reach.gif.

Uses a deterministic latent (obs_step sample=False) + action EMA + freeze on
target, so the rendered arm is smooth (the raw stochastic latent jitters).

Paths via env vars (see dreamer/README.md):
  WMA_ENVS, WMA_DREAMER, WMA_MENAGERIE, WMA_DREAMER_CKPT
"""
import os, sys, argparse, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("WMA_ENVS", str(REPO / "envs"))
DV = os.environ.get("WMA_DREAMER", str(REPO / "dreamerv3-torch"))
sys.path.insert(0, DV)
CKPT = pathlib.Path(os.environ.get(
    "WMA_DREAMER_CKPT", os.path.join(DV, "logdir", "ur5e_state", "ur5e_state_final.pt")))

import numpy as np, torch, mujoco
import ruamel.yaml as yaml
from PIL import Image
import tools
import envs.ur5e as ur5e
from dreamer import Dreamer

configs = yaml.safe_load((pathlib.Path(DV) / "configs.yaml").read_text())
def rupdate(b, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in b: rupdate(b[k], v)
        else: b[k] = v
defaults = {}
for name in ["defaults", "ur5e_proprio"]:
    rupdate(defaults, configs[name])
p = argparse.ArgumentParser()
for k, v in sorted(defaults.items()):
    p.add_argument(f"--{k}", type=tools.args_type(v), default=v)
config = p.parse_args([]); config.num_actions = 6
config.device = "cuda" if torch.cuda.is_available() else "cpu"
dev = config.device
class DL: step = 0

env = ur5e.UR5eState(seed=5); u = env._u
agent = Dreamer(env.observation_space, env.action_space, config, DL(), None).to(dev)
agent.requires_grad_(False)
agent.load_state_dict(torch.load(CKPT, map_location=dev)["agent_state_dict"])
agent.eval()
wm, actor = agent._wm, agent._task_behavior.actor
blank = np.zeros((16, 16, 3), np.uint8)
ren = mujoco.Renderer(u.model, 480, 480)

TARGETS = [[0.42, -0.05, 0.40], [0.40, -0.28, 0.30], [0.35, -0.15, 0.38], [0.48, -0.10, 0.34]]
def batch(o): return {k: np.array([v]) for k, v in o.items()}
def frame():
    ren.update_scene(u.data, camera="cam"); return ren.render()

gif, dists = [], []
for ep in range(4):
    env.reset()
    tgt = np.array(TARGETS[ep % len(TARGETS)]); u._target = tgt; u.data.mocap_pos[0] = tgt
    mujoco.mj_forward(u.model, u.data)
    obs = env._obs(u._get_obs(), True, False)
    prev_lat, prev_act = None, torch.zeros(1, 6, device=dev)
    a_smooth, a_hold, near = None, None, 0
    with torch.no_grad():
        for t in range(80):
            data = wm.preprocess(batch(obs)); embed = wm.encoder(data)
            lat, _ = wm.dynamics.obs_step(prev_lat, prev_act, embed, data["is_first"], sample=False)
            a = actor(wm.dynamics.get_feat(lat)).mode()[0].cpu().numpy()
            a_smooth = a if a_smooth is None else 0.5 * a_smooth + 0.5 * a
            dist = float(np.linalg.norm(u._ee() - u._target))
            if dist < 0.05:
                near += 1
                if near >= 3 and a_hold is None:
                    a_hold = a_smooth.copy()
            else:
                near = 0; a_hold = None
            cmd = a_hold if a_hold is not None else a_smooth
            obs, _, _, _ = env.step(cmd)
            prev_lat = {k: v.detach() for k, v in lat.items()}
            prev_act = torch.tensor([cmd], dtype=torch.float32, device=dev)
            gif.append(Image.fromarray(frame()))
    dists.append(float(np.linalg.norm(u._ee() - u._target)))
    print(f"ep {ep}: dist final {dists[-1]:.3f} m")

out = REPO / "figures" / "dreamer_ur5e_reach.gif"
gif[0].save(out, save_all=True, append_images=gif[1:], duration=55, loop=0)
print("dist media:", round(float(np.mean(dists)), 3), "m -> saved", out)
