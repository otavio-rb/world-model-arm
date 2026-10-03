"""SAC baseline on the UR5e reach task, logging return vs REAL env steps."""
import os, sys, json, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "envs"))
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ur5e_reach import UR5eReachEnv

N_ENVS = 4
TOTAL = 250_000
RESULTS = os.path.join(_REPO, "results")
os.makedirs(RESULTS, exist_ok=True)
LOG = os.path.join(RESULTS, "sac_ur5e.jsonl")
CKPT = os.path.join(RESULTS, "sac_ur5e")


class CurveLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.f = open(LOG, "w")
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.f.write(json.dumps({"steps": int(self.num_timesteps),
                                         "return": round(float(ep["r"]), 2),
                                         "t": round(time.time() - self.t0, 1)}) + "\n")
                self.f.flush()
        return True


if __name__ == "__main__":
    vec = DummyVecEnv([lambda: Monitor(UR5eReachEnv()) for _ in range(N_ENVS)])
    model = SAC("MlpPolicy", vec, device="cuda", verbose=0,
                learning_starts=3000, train_freq=1, gradient_steps=1, buffer_size=400_000)
    print("treinando SAC no UR5e...", flush=True)
    model.learn(total_timesteps=TOTAL, callback=CurveLogger())
    model.save(CKPT)
    print("SAC done ->", CKPT, flush=True)
