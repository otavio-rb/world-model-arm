"""Side-by-side: REAL world (left) vs the world MODEL's IMAGINATION (right).

Left  = the actual UR5e, driven by the Dreamer policy (deterministic latent, so
        no tremble).
Right = a second UR5e posed at the joint angles the world model IMAGINES for the
        near future (it decodes full states, incl. joints), animating the plan
        the model is 'thinking' — something a model-free policy simply cannot do.

Click the LEFT view to place the target; Z slider + random button too.
Run:  .venv-robot/Scripts/python.exe -m uvicorn robot_demo.ur5e_split_server:app --port 8095
"""
import os, sys, io, time, base64, threading, asyncio, argparse, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import torch
import mujoco
import ruamel.yaml as yaml
from PIL import Image, ImageDraw
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
DV = os.environ.get("WMA_DREAMER", os.path.join(REPO, "dreamerv3-torch"))
ENVS = os.environ.get("WMA_ENVS", os.path.join(REPO, "envs"))
sys.path.insert(0, DV); sys.path.insert(0, ENVS)
import tools
import envs.ur5e as ur5e
from dreamer import Dreamer
from ur5e_reach import UR5eReachEnv, SHOULDER, XML

CKPT = pathlib.Path(os.environ.get(
    "WMA_DREAMER_CKPT", os.path.join(DV, "logdir", "ur5e_state", "ur5e_state_final.pt")))
SZ, H, HZ = 460, 30, 15   # H = imagination lead (steps the dream runs ahead of reality)

# config
configs = yaml.safe_load((pathlib.Path(DV) / "configs.yaml").read_text())
def rupdate(b, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in b: rupdate(b[k], v)
        else: b[k] = v
defaults = {}
for name in ["defaults", "ur5e_proprio"]:
    rupdate(defaults, configs[name])
parser = argparse.ArgumentParser()
for k, v in sorted(defaults.items()):
    parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)
config = parser.parse_args([]); config.num_actions = 6
config.device = "cuda" if torch.cuda.is_available() else "cpu"
class DummyLogger: step = 0

app = FastAPI()
_state = {"lock": threading.Lock(), "img": None, "dist": None, "reached": False}
_ctrl = {"lock": threading.Lock(), "target": np.array([0.4, 0.3, 0.5]), "z": 0.5}


def _clamp(t):
    v = np.asarray(t, float) - SHOULDER
    r = np.linalg.norm(v) + 1e-9
    r = min(max(r, 0.30), 0.72)
    p = SHOULDER + v / np.linalg.norm(v + 1e-9) * r
    p[2] = max(p[2], 0.12)
    return p


def _run():
    tenv = ur5e.UR5eState(seed=0)
    wm = None
    agent = Dreamer(tenv.observation_space, tenv.action_space, config, DummyLogger(), None).to(config.device)
    agent.requires_grad_(False)
    agent.load_state_dict(torch.load(CKPT, map_location=config.device)["agent_state_dict"])
    agent.eval()
    wm, actor = agent._wm, agent._task_behavior.actor
    blank = np.zeros((16, 16, 3), np.uint8)

    env = UR5eReachEnv(); obs_v, _ = env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    u = env.unwrapped
    dm = mujoco.MjModel.from_xml_path(XML); dd = mujoco.MjData(dm)   # "dream" arm
    ren = mujoco.Renderer(m, SZ, SZ)
    dren = mujoco.Renderer(dm, SZ, SZ)
    cid = m.camera("cam").id
    mujoco.mj_forward(m, d)
    cam_pos = d.cam_xpos[cid].copy(); cam_mat = d.cam_xmat[cid].reshape(3, 3).copy()
    tanv = np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2)

    def unproject(uu, vv, z):
        dirw = cam_mat @ np.array([(2 * uu - 1) * tanv, (1 - 2 * vv) * tanv, -1.0])
        if abs(dirw[2]) < 1e-6: return None
        t = (z - cam_pos[2]) / dirw[2]
        return cam_pos + t * dirw if t > 0 else None
    globals()["_unproject"] = unproject

    prev_lat = None
    prev_act = torch.zeros(1, 6, device=config.device)
    a_smooth = None
    a_hold = None
    dream_q = None
    fc = 0
    frozen = False
    settle = 0
    last_tgt = None
    dt = 1.0 / HZ
    while True:
        t0 = time.time()
        fc += 1
        with _ctrl["lock"]:
            tgt = _clamp(_ctrl["target"])
        u._target = tgt; d.mocap_pos[0] = tgt
        if last_tgt is None or np.linalg.norm(tgt - last_tgt) > 0.004:
            frozen = False; settle = 0          # target moved -> re-engage the policy
        last_tgt = tgt.copy()
        obs_v = u._get_obs()
        # deterministic policy (sample=False) -> stable
        bobs = {"state": np.array([obs_v], np.float32), "image": np.array([blank]),
                "is_first": np.array([prev_lat is None]), "is_terminal": np.array([False])}
        data = wm.preprocess(bobs)
        with torch.no_grad():
            embed = wm.encoder(data)
            lat, _ = wm.dynamics.obs_step(prev_lat, prev_act, embed, data["is_first"], sample=False)
            feat = wm.dynamics.get_feat(lat)
            act = actor(feat).mode()
            # imagine H steps ahead; decode only the endpoint -> the pose the model
            # predicts for ~H steps in the future (the "lead"). cheap: 1 decode.
            if dream_q is None or fc % 2 == 0:
                l = {k: v.clone() for k, v in lat.items()}
                for _ in range(H):
                    f = wm.dynamics.get_feat(l)
                    l = wm.dynamics.img_step(l, actor(f).mode(), sample=False)
                st = wm.heads["decoder"](wm.dynamics.get_feat(l).unsqueeze(1))["state"].mode()
                dream_q = st[0, 0, :6].cpu().numpy().copy()
        a = act[0].cpu().numpy()
        a_smooth = a if a_smooth is None else 0.7 * a_smooth + 0.3 * a
        dist_now = float(np.linalg.norm(u._ee() - tgt))
        # freeze solid once settled on a static target; unfreeze only when target moves
        if not frozen:
            cmd = a_smooth
            settle = settle + 1 if dist_now < 0.045 else 0
            if settle >= 12:
                frozen = True; a_hold = a_smooth.copy()
        else:
            cmd = a_hold
        env.step(cmd); u._t = 0
        prev_lat = {k: v.detach() for k, v in lat.items()}
        prev_act = torch.tensor(cmd[None], dtype=torch.float32, device=config.device)

        # render REAL
        ren.update_scene(d, camera="cam"); left = ren.render()
        # render DREAM: the pose the model predicts ~H steps ahead (leads reality)
        dd.qpos[:6] = dream_q; dd.mocap_pos[0] = tgt
        mujoco.mj_forward(dm, dd)
        dren.update_scene(dd, camera="cam"); right = dren.render()

        gap = np.full((SZ, 6, 3), 20, np.uint8)
        combo = np.concatenate([left, gap, right], axis=1)
        im = Image.fromarray(combo); dr = ImageDraw.Draw(im)
        dr.text((12, 10), "MUNDO REAL", fill=(230, 230, 240))
        dr.text((SZ + 18, 10), "IMAGINACAO DO MODELO (~30 passos a frente)", fill=(120, 245, 170))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=78)
        dist = float(np.linalg.norm(u._ee() - tgt))
        with _state["lock"]:
            _state["img"] = base64.b64encode(buf.getvalue()).decode()
            _state["dist"] = round(dist, 3); _state["reached"] = dist < 0.06
        time.sleep(max(0.0, dt - (time.time() - t0)))


@app.on_event("startup")
def _startup():
    threading.Thread(target=_run, daemon=True).start()


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(HERE, "split.html"), encoding="utf-8").read())


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    async def sender():
        while True:
            with _state["lock"]:
                img = _state["img"]; dist = _state["dist"]; reached = _state["reached"]
            if img is not None:
                await sock.send_json({"img": img, "dist": dist, "reached": reached})
            await asyncio.sleep(1 / 30)
    async def receiver():
        while True:
            msg = await sock.receive_json()
            with _ctrl["lock"]:
                if "z" in msg: _ctrl["z"] = float(msg["z"])
                if "click" in msg and "_unproject" in globals():
                    p = _unproject(float(msg["click"][0]), float(msg["click"][1]), _ctrl["z"])
                    if p is not None: _ctrl["target"] = p
                if msg.get("random"):
                    ang = np.random.uniform(-np.pi, np.pi); rr = np.random.uniform(0.35, 0.65)
                    _ctrl["target"] = np.array([rr*np.cos(ang), rr*np.sin(ang), np.random.uniform(0.2, 0.6)]) + SHOULDER*[1,1,0]
    try:
        await asyncio.gather(sender(), receiver())
    except (WebSocketDisconnect, RuntimeError):
        pass
