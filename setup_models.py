"""One-shot setup of external dependencies for world-model-arm.

Clones the robot models (MuJoCo Menagerie: UR5e + Franka Panda) and the
DreamerV3 implementation (NM512/dreamerv3-torch), then applies the small
integration this repo needs:
  - copies our task XMLs into the Menagerie model dirs (so <include> resolves),
  - copies dreamer/envs_ur5e.py -> dreamerv3-torch/envs/ur5e.py,
  - appends the `ur5e_proprio` config to dreamerv3-torch/configs.yaml,
  - patches dreamerv3-torch/dreamer.py (Windows GL backend + a `ur5e` env branch).

Idempotent: safe to re-run. Requires `git` on PATH.

    python setup_models.py
"""
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent
MEN = REPO / "menagerie"
DV = REPO / "dreamerv3-torch"


def run(cmd, cwd=None):
    print("  $", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def clone_menagerie():
    if (MEN / "universal_robots_ur5e" / "scene.xml").exists() and \
       (MEN / "franka_emika_panda" / "panda.xml").exists():
        print("[menagerie] already present")
        return
    print("[menagerie] sparse-cloning UR5e + Franka Panda ...")
    if not MEN.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/google-deepmind/mujoco_menagerie.git", str(MEN)])
    run(["git", "sparse-checkout", "set",
         "universal_robots_ur5e", "franka_emika_panda"], cwd=str(MEN))


def place_task_xmls():
    pairs = [("reach_task.xml", "universal_robots_ur5e"),
             ("grasp_task.xml", "franka_emika_panda")]
    for xml, model in pairs:
        dst = MEN / model / xml
        shutil.copy(REPO / "envs" / "assets" / xml, dst)
        print(f"[menagerie] placed {xml} -> {model}/")


def clone_dreamer():
    if (DV / "dreamer.py").exists():
        print("[dreamer] already present")
        return
    print("[dreamer] cloning NM512/dreamerv3-torch ...")
    run(["git", "clone", "--depth", "1",
         "https://github.com/NM512/dreamerv3-torch.git", str(DV)])


def integrate_dreamer():
    # 1) env wrapper
    shutil.copy(REPO / "dreamer" / "envs_ur5e.py", DV / "envs" / "ur5e.py")
    print("[dreamer] copied envs/ur5e.py")

    # 2) config block
    cfg = DV / "configs.yaml"
    text = cfg.read_text()
    if "ur5e_proprio:" not in text:
        block = (REPO / "dreamer" / "configs_ur5e.yaml").read_text()
        cfg.write_text(text.rstrip() + "\n\n" + block.rstrip() + "\n")
        print("[dreamer] appended ur5e_proprio to configs.yaml")
    else:
        print("[dreamer] ur5e_proprio already in configs.yaml")

    # 3) patch dreamer.py
    dp = DV / "dreamer.py"
    src = dp.read_text()
    orig = src
    src = src.replace(
        'os.environ["MUJOCO_GL"] = "osmesa"',
        'os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "osmesa")')
    branch = (
        '    elif suite == "ur5e":\n'
        '        import envs.ur5e as ur5e_mod\n'
        '        cls = ur5e_mod.UR5eState if task == "state" else ur5e_mod.UR5ePixel\n'
        '        env = cls(size=tuple(config.size), action_repeat=config.action_repeat,\n'
        '                  seed=config.seed + id)\n'
        '        env = wrappers.NormalizeActions(env)\n'
        '    else:\n        raise NotImplementedError(suite)\n')
    if 'suite == "ur5e"' not in src:
        src = src.replace('    else:\n        raise NotImplementedError(suite)\n', branch, 1)
    if src != orig:
        dp.write_text(src)
        print("[dreamer] patched dreamer.py (GL backend + ur5e branch)")
    else:
        print("[dreamer] dreamer.py already patched")


if __name__ == "__main__":
    clone_menagerie()
    place_task_xmls()
    clone_dreamer()
    integrate_dreamer()
    print("\nDone. For the Dreamer scripts, set:")
    print(f'  WMA_ENVS = {REPO / "envs"}')
    print(f'  WMA_DREAMER = {DV}')
    print("Then: python train/train_sac_ur5e.py   (SAC baseline)")
    print("      cd dreamerv3-torch && python dreamer.py --configs ur5e_proprio "
          "--logdir logdir/ur5e_state --steps 150000  (world model)")
