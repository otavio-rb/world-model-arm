"""TCC comparison: SAC (model-free) vs DreamerV3 (world model) sample efficiency."""
import json, re, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1,2
INK, SUB, GRID, SURF = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"

sac = sorted((r["steps"], r["return"]) for r in map(json.loads, open(ROOT/"results"/"sac_ur5e.jsonl")))
dre = []
_dm = ROOT / "results" / "dreamer_ur5e_metrics.jsonl"
for ln in open(_dm, encoding="utf-8", errors="ignore"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if "train_return" in r and r.get("step", 0) > 2500:
        dre.append((int(r["step"]), float(r["train_return"])))
dre.sort()

def roll(pairs, w=20):
    xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
    if len(ys) < w: return xs, ys
    ym = np.convolve(ys, np.ones(w)/w, mode="valid")
    return xs[w-1:], ym

def steps_to(pairs, thr=130, w=20):
    xs, ym = roll(pairs, w)
    hit = np.where(ym >= thr)[0]
    return int(xs[hit[0]]) if len(hit) else None

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": SUB, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB})
fig, ax = plt.subplots(figsize=(9, 5.4)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

for data, color, name in [(sac, BLUE, "SAC (model-free)"), (dre, ORANGE, "DreamerV3 (world model)")]:
    xs = [p[0] for p in data]; ys = [p[1] for p in data]
    ax.scatter(xs, ys, s=5, color=color, alpha=0.08, linewidths=0)
    xm, ym = roll(data)
    ax.plot(xm, ym, color=color, lw=2.4, solid_capstyle="round")
    ax.annotate(name, (xm[-1], ym[-1]), color=color, fontsize=11, fontweight="bold",
                xytext=(6, 0), textcoords="offset points", va="center")

# sample-efficiency annotation: steps to reach return 100
THR = 100
s_sac, s_dre = steps_to(sac, THR), steps_to(dre, THR)
ax.axhline(THR, color=SUB, lw=0.8, ls=":", alpha=0.6)
for s, color in [(s_dre, ORANGE), (s_sac, BLUE)]:
    if s: ax.axvline(s, color=color, lw=1, ls="--", alpha=0.5, ymax=0.75)
if s_sac and s_dre:
    ax.annotate("", xy=(s_sac, THR-40), xytext=(s_dre, THR-40),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text((s_sac+s_dre)/2, THR-58, f"{s_sac/s_dre:.1f}x menos\ninteracao real",
            ha="center", va="top", color=INK, fontsize=10, fontweight="bold")
    ax.text(s_dre, THR+8, f"Dreamer: {s_dre//1000}k", color=ORANGE, fontsize=9, ha="center")
    ax.text(s_sac, THR+8, f"SAC: {s_sac//1000}k", color=BLUE, fontsize=9, ha="center")

ax.set_xlabel("Passos reais no ambiente (interação com o simulador)")
ax.set_ylabel("Retorno por episódio")
ax.set_title("Eficiência amostral — alcance com o UR5e\nmodel-free vs world model",
             fontsize=13, fontweight="bold", loc="left")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.margins(x=0.02); ax.set_xlim(left=0)
plt.tight_layout()
out = ROOT / "figures" / "sample_efficiency_ur5e.png"
plt.savefig(out, dpi=140, facecolor=SURF)
gain = f"{s_sac/s_dre:.1f}x" if (s_sac and s_dre) else "n/a"
print(f"SAC->{THR}: {s_sac} passos | Dreamer->{THR}: {s_dre} passos | ganho {gain}")
print("saved", out)
