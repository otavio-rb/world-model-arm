"""SAC on the Panda pick-up task, logging return + success rate (grasp is hard)."""
import os, sys, json, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "envs"))
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from panda_grasp import PandaGraspEnv

N_ENVS = 6
TOTAL = 600_000
RESULTS = os.path.join(_REPO, "results")
os.makedirs(RESULTS, exist_ok=True)
LOG = os.path.join(RESULTS, "sac_grasp.jsonl")
CKPT = os.path.join(RESULTS, "sac_grasp")


class CurveLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.f = open(LOG, "w"); self.t0 = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.f.write(json.dumps({
                    "steps": int(self.num_timesteps), "return": round(float(ep["r"]), 2),
                    "success": bool(info.get("success", False)),
                    "box_z": round(float(info.get("box_z", 0)), 3),
                    "t": round(time.time() - self.t0, 1)}) + "\n")
                self.f.flush()
        return True


if __name__ == "__main__":
    vec = DummyVecEnv([lambda: Monitor(PandaGraspEnv(), info_keywords=("success", "box_z"))
                       for _ in range(N_ENVS)])
    model = SAC("MlpPolicy", vec, device="cuda", verbose=0,
                learning_starts=5000, train_freq=1, gradient_steps=1, buffer_size=600_000)
    print("treinando SAC no Panda grasp...", flush=True)
    model.learn(total_timesteps=TOTAL, callback=CurveLogger())
    model.save(CKPT)
    print("done ->", CKPT, flush=True)
