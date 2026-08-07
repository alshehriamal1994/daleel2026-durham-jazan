import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0,
})

BLUE, VERM = "#0072B2", "#D55E00"

model = [  # (label, promised, delivered) sorted by promise, descending
    ("DAPT + back-translation (T1)", 5.2, 9.6),
    ("synthetic data (T1)",          3.0, 1.8),
    ("synthetic data (T2)",          2.6, 0.2),
    ("span compaction (T2)",         1.4, 1.4),
    ("rare-class augmentation (T1)", 1.2, 3.1),
    ("32B routing (T1)",             1.1, 1.0),
    ("char-level blend (T2)",        0.4, 0.5),
]
decision = [
    ("per-label rules (T1)",         5.0, -3.8),
    ("span fusion (T1 Closed)",      2.9, -1.8),
    ("span fusion (T1 Open)",        2.3,  1.9),
    ("per-domain thresholds (T1)",   1.0, -1.4),
    ("recalibration (T2)",           0.9,  0.0),
]

fig, ax = plt.subplots(figsize=(3.03, 2.24))

rows = []       # (y, label, promised, delivered, color)
y = 13.0
rows_hdr = [(y + 0.15, "Model changes", BLUE)]
for lab, p, d in model:
    y -= 1.0
    rows.append((y, lab, p, d, BLUE))
y -= 1.55
rows_hdr.append((y + 0.15, "Decision rule changes", VERM))
for lab, p, d in decision:
    y -= 1.0
    rows.append((y, lab, p, d, VERM))

ax.axvspan(-4.6, 0, color="#D55E00", alpha=0.05, zorder=0)
ax.axvline(0, lw=0.7, color="#888888", zorder=1)
for x in (-4, -2, 2, 4, 6, 8, 10):
    ax.axvline(x, lw=0.4, color="#e3e3e3", zorder=0)

for yy, lab, p, d, c in rows:
    ax.plot([p, d], [yy, yy], color="#c9c9c9", lw=1.1, zorder=2,
            solid_capstyle="round")
    ax.scatter(p, yy, s=15, facecolor="#a8a8a8", edgecolor="none", zorder=3)
    ax.scatter(d, yy, s=40, facecolor=c, edgecolor="white", linewidths=0.7, zorder=4)

ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows])
for yy, txt, c in rows_hdr:
    ax.text(-0.02, yy, txt, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.6, fontweight="bold", color=c)

ax.set_xlim(-4.6, 10.3)
ax.set_ylim(-1.15, 13.75)
ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
ax.set_xlabel("F1 points")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="x", length=2.5, color="#999999")
ax.spines["bottom"].set_color("#999999")


handles = [
    Line2D([], [], marker="o", ls="", color="#b0b0b0", markersize=4,
           label="promised"),
    Line2D([], [], marker="o", ls="", color="#444444", markeredgecolor="white",
           markersize=6.5, label="delivered"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, frameon=False, borderaxespad=0.0, handletextpad=0.2,
          columnspacing=2.0, fontsize=7.2)

fig.tight_layout(pad=0.15)
os.makedirs("/home/amal/Desktop/daleel2026/paper/figs", exist_ok=True)
fig.savefig("/home/amal/Desktop/daleel2026/paper/figs/transfer.pdf")
print("saved")
