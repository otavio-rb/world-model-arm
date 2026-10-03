"""Scripted IK grasp: move over the cube, descend, close, lift. Makes a GIF."""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, mujoco, pathlib
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
MEN = os.environ.get("WMA_MENAGERIE", str(REPO / "menagerie"))
FR = os.path.join(MEN, "franka_emika_panda")
m = mujoco.MjModel.from_xml_path(os.path.join(FR, "grasp_task.xml"))
d = mujoco.MjData(m)
hand = m.body("hand").id
kid = m.key("home").id
mujoco.mj_resetDataKeyframe(m, d, kid)
# place cube in front
d.qpos[9], d.qpos[10], d.qpos[11] = 0.5, 0.05, 0.03
d.qpos[12:16] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)

jlo, jhi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
ren = mujoco.Renderer(m, 480, 480)
frames = []

def grasp_pt():
    R = d.xmat[hand].reshape(3, 3)
    return d.xpos[hand] + R @ np.array([0, 0, 0.103])

def move_to(target, grip, n, cap=True):
    for _ in range(n):
        mujoco.mj_forward(m, d)
        gp = grasp_pt()
        err = np.asarray(target) - gp
        mujoco.mj_jac(m, d, jacp, jacr, gp, hand)
        Jp = jacp[:, :7]
        dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-4 * np.eye(3), err)
        d.ctrl[:7] = np.clip(d.qpos[:7] + 0.5 * dq, jlo, jhi)
        d.ctrl[7] = grip
        for _ in range(8):
            mujoco.mj_step(m, d)
        if cap:
            ren.update_scene(d, camera="cam"); frames.append(Image.fromarray(ren.render()))

box = d.qpos[9:12].copy()
print("cubo:", np.round(box, 3))
move_to(box + [0, 0, 0.18], 0.04, 40)   # 1) above the cube, gripper open
move_to(box + [0, 0, 0.02], 0.04, 45)   # 2) descend onto it
move_to(box + [0, 0, 0.02], 0.0, 25)    # 3) close gripper
move_to(box + [0, 0, 0.30], 0.0, 55)    # 4) lift
box_final = d.qpos[9:12]
print("cubo final z:", round(float(box_final[2]), 3), "-> PEGOU!" if box_final[2] > 0.15 else "-> falhou")

out = REPO / "figures"
frames[0].save(out / "scripted_grasp.gif", save_all=True, append_images=frames[1:], duration=45, loop=0)
# a strip too
idx = np.linspace(0, len(frames) - 1, 8).astype(int)
strip = np.concatenate([np.array(frames[i]) for i in idx], axis=1)
Image.fromarray(strip).save(out / "scripted_grasp_strip.png")
print("saved panda_pick.gif e panda_pick.png")
