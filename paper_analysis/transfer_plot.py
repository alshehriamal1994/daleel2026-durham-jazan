import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

BLUE, VERM = "#0072B2", "#D55E00"

# (name, val_gain, realized_gain, label_x, label_y, ha)
model = [
    ("DAPT+BT T1",   5.2,  9.6,  5.00,  9.6,  "right"),
    ("rare-aug T1",  1.2,  3.1,  1.42,  3.35, "left"),
    ("synth T1",     3.0,  1.8,  3.24,  1.80, "left"),
    ("synth T2",     2.6,  0.2,  2.82, -0.20, "left"),
    ("compaction T2",1.4,  1.4,  1.62,  1.35, "left"),
    ("blend T2",     0.4,  0.5, -0.20,  2.05, "left"),
    ("32B T1",       1.1,  1.0,  1.28,  0.55, "left"),
]
decision = [
    ("thresholds T1", 1.0, -1.4,  1.22, -1.45, "left"),
    ("fusion T1-C",   2.9, -1.8,  3.12, -1.85, "left"),
    ("fusion T1-O",   2.3,  1.9,  2.52,  2.30, "left"),
    ("recalib. T2",   0.9,  0.0,  0.90, -0.72, "center"),
    ("rules T1",      5.0, -3.8,  4.78, -3.80, "right"),
]

fig, ax = plt.subplots(figsize=(3.03, 2.5))

ax.plot([0, 6.3], [0, 6.3], ls=(0, (4, 3)), lw=0.8, color="#999999", zorder=1)
ax.text(4.35, 4.95, "perfect transfer", rotation=19, fontsize=7, color="#777777",
        ha="center", va="bottom")
ax.plot([0.40, 0.16], [0.78, 1.72], lw=0.6, color="#999999", zorder=2)
ax.axhspan(-4.9, 0, color="#D55E00", alpha=0.045, zorder=0)
ax.axhline(0, lw=0.6, color="#bbbbbb", zorder=1)
ax.text(-0.28, -4.55, "promised, not delivered", fontsize=6.6, color="#a06030",
        style="italic", ha="left", va="bottom")

for name, x, y, lx, ly, ha in model:
    ax.scatter(x, y, s=34, marker="o", color=BLUE, edgecolors="white",
               linewidths=0.7, zorder=3)
    ax.text(lx, ly, name, fontsize=6.6, ha=ha, va="center", color="#1a1a1a", zorder=4)
for name, x, y, lx, ly, ha in decision:
    ax.scatter(x, y, s=40, marker="X", color=VERM, edgecolors="white",
               linewidths=0.5, zorder=3)
    ax.text(lx, ly, name, fontsize=6.6, ha=ha, va="center", color="#1a1a1a", zorder=4)

ax.set_xlim(-0.45, 6.3)
ax.set_ylim(-4.9, 10.5)
ax.set_xlabel("validation gain (F1 points)")
ax.set_ylabel("realized leaderboard gain (F1 points)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, color="#999999")
for s in ["left","bottom"]: ax.spines[s].set_color("#999999")

handles = [
    Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white",
           markersize=6, label="model-level"),
    Line2D([], [], marker="X", ls="", color=VERM, markeredgecolor="white",
           markersize=7, label="decision-layer"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0.1,
          handletextpad=0.2)

fig.tight_layout(pad=0.15)
os.makedirs("/home/amal/Desktop/daleel2026/paper/figs", exist_ok=True)
fig.savefig("/home/amal/Desktop/daleel2026/paper/figs/transfer.pdf")
print("saved")
