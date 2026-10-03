"""Open-loop DREAM vs REALITY for the trained world model, plus the error curve.

Seeds the world model on K real frames, then imagines free (no simulator) using
the SAME recorded actions, and compares the imagined arm to the real one over the
horizon. Writes figures/dream_vs_reality.gif and figures/prediction_error.png.

Requires: a cloned dreamerv3-torch with our integration (see dreamer/README.md)
and a trained Dreamer checkpoint. Paths via env vars (defaults below):
  WMA_DREAMER       -> dreamerv3-torch dir     (default: <repo>/dreamerv3-torch)
  WMA_DREAMER_CKPT  -> trained checkpoint .pt  (default: <dreamer>/logdir/ur5e_state/ur5e_state_final.pt)
  WMA_ENVS          -> set to <repo>/envs so the dreamer env finds ur5e_reach
"""
import os, sys, argparse, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("WMA_ENVS", str(REPO / "envs"))
DV = os.environ.get("WMA_DREAMER", str(REPO / "dreamerv3-torch"))
sys.path.insert(0, DV)
sys.path.insert(0, os.environ["WMA_ENVS"])
CKPT = pathlib.Path(os.environ.get(
    "WMA_DREAMER_CKPT", os.path.join(DV, "logdir", "ur5e_state", "ur5e_state_final.pt")))

import numpy as np, torch, mujoco
import ruamel.yaml as yaml
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tools
import envs.ur5e as ur5e
from ur5e_reach import XML
from dreamer import Dreamer

T, K, SZ = 55, 5, 440                        # horizon, seed frames, image size
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

env = ur5e.UR5eState(seed=4); u = env._u
agent = Dreamer(env.observation_space, env.action_space, config, DL(), None).to(dev)
agent.requires_grad_(False)
agent.load_state_dict(torch.load(CKPT, map_location=dev)["agent_state_dict"])
agent.eval()
wm, actor = agent._wm, agent._task_behavior.actor
blank = np.zeros((16, 16, 3), np.uint8)

def batch(o): return {k: np.array([v]) for k, v in o.items()}
def decode(l):
    return wm.heads["decoder"](wm.dynamics.get_feat(l).unsqueeze(1))["state"].mode()[0, 0].cpu().numpy()

# fixed, camera-facing, reachable target (visible in the render)
FRONT_TARGET = np.array([0.35, -0.15, 0.38])

# Phase 1: real rollout with the policy (record qpos, smoothed actions, ee).
# The applied action is EMA-smoothed so the rendered real arm moves smoothly
# (the raw policy jitters at the target); imagination replays the same actions.
env.reset()
target = FRONT_TARGET.copy(); u._target = target; u.data.mocap_pos[0] = target
mujoco.mj_forward(u.model, u.data)
obs = env._obs(u._get_obs(), True, False)
obss, acts, rqpos, ree = [obs], [], [], []
prev_lat, prev_act = None, torch.zeros(1, 6, device=dev)
a_smooth = None
with torch.no_grad():
    for t in range(T):
        data = wm.preprocess(batch(obs)); embed = wm.encoder(data)
        lat, _ = wm.dynamics.obs_step(prev_lat, prev_act, embed, data["is_first"], sample=False)
        a = actor(wm.dynamics.get_feat(lat)).mode()[0].cpu().numpy()
        a_smooth = a if a_smooth is None else 0.5 * a_smooth + 0.5 * a
        rqpos.append(u.data.qpos[:6].copy()); ree.append(u._ee().copy())
        acts.append(a_smooth.copy())
        obs, _, _, _ = env.step(a_smooth); obss.append(obs)
        prev_lat = {k: v.detach() for k, v in lat.items()}
        prev_act = torch.tensor([a_smooth], dtype=torch.float32, device=dev)

# Phase 2: seed on first K real frames, then imagine OPEN-LOOP
prev_lat, prev_act = None, torch.zeros(1, 6, device=dev)
with torch.no_grad():
    for t in range(K):
        data = wm.preprocess(batch(obss[t])); embed = wm.encoder(data)
        lat, _ = wm.dynamics.obs_step(prev_lat, prev_act, embed, data["is_first"], sample=False)
        prev_lat = {k: v.detach() for k, v in lat.items()}
        prev_act = torch.tensor([acts[t]], dtype=torch.float32, device=dev)
    dqpos = list(rqpos[:K]); dee = list(ree[:K])
    l = lat
    for t in range(K, T):
        l = wm.dynamics.img_step(l, torch.tensor([acts[t-1]], dtype=torch.float32, device=dev), sample=False)
        st = decode(l); dqpos.append(st[:6].copy()); dee.append(st[12:15].copy())

err = [float(np.linalg.norm(np.array(dee[t]) - ree[t])) for t in range(T)]

# Phase 3: render side by side + error plot
rm = mujoco.MjModel.from_xml_path(XML); rd = mujoco.MjData(rm)
dm = mujoco.MjModel.from_xml_path(XML); dd = mujoco.MjData(dm)
rr = mujoco.Renderer(rm, SZ, SZ); drn = mujoco.Renderer(dm, SZ, SZ)
frames = []
for t in range(T):
    rd.qpos[:6] = rqpos[t]; rd.mocap_pos[0] = target; mujoco.mj_forward(rm, rd)
    dd.qpos[:6] = dqpos[t]; dd.mocap_pos[0] = target; mujoco.mj_forward(dm, dd)
    rr.update_scene(rd, camera="cam"); drn.update_scene(dd, camera="cam")
    gap = np.full((SZ, 6, 3), 20, np.uint8)
    im = Image.fromarray(np.concatenate([rr.render(), gap, drn.render()], axis=1))
    dr = ImageDraw.Draw(im)
    phase = "OBSERVADO" if t < K else "IMAGINADO (malha aberta)"
    dr.text((10, 8), "REALIDADE (MuJoCo)", fill=(235, 235, 245))
    dr.text((SZ + 16, 8), "SONHO DO WORLD MODEL", fill=(120, 245, 170))
    dr.text((10, SZ - 20), f"t={t:02d}  {phase}   erro={err[t]*100:.1f} cm", fill=(255, 220, 120))
    frames.append(im)

fig = REPO / "figures"
frames[0].save(fig / "dream_vs_reality.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)

plt.figure(figsize=(8, 4.2))
plt.axvspan(0, K - 1, color="#2a78d6", alpha=0.12, label=f"observado (seed, {K} frames)")
plt.plot(range(T), np.array(err) * 100, color="#eb6834", lw=2.2)
plt.axvline(K - 1, color="#888", ls="--", lw=1)
plt.xlabel("passo imaginado"); plt.ylabel("erro sonho vs real (cm)")
plt.title("Erro de previsao do world model ao longo do horizonte (malha aberta)")
plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
plt.savefig(fig / "prediction_error.png", dpi=130)
print(f"erro em t=15: {err[min(15,T-1)]*100:.1f} cm | t={T-1}: {err[-1]*100:.1f} cm")
print("saved dream_vs_reality.gif e prediction_error.png")
