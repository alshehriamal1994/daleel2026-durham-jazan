import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

BLUE, GRAY, INK = "#0072B2", "#8a8a8a", "#1a1a1a"

# (label, promised, realised) — sorted by promise (desc) within each group
model = [
    ("DAPT + BT (T1)",   5.2,  9.6),
    ("LLM-synth (T1)",   2.7,  1.8),
    ("LLM-synth (T2)",   2.6,  0.2),
    ("compaction (T2)",  1.4,  1.4),
    ("rare-aug 3$\\times$ (T1)", 1.2,  3.1),
    ("LLM routing (T1)", 1.1,  1.0),
    ("char blend (T2)",  0.4,  0.5),
]
decision = [
    ("rules (T1)",       5.0, -3.8),
    ("fusion (T1-C)",    2.9, -1.8),
    ("fusion (T1-O)",    2.3,  1.9),
    ("thresholds (T1)",  1.0, -1.4),
    ("recalib. (T2)",    0.9,  0.0),
]

fig, ax = plt.subplots(figsize=(3.03, 2.32))

# row layout: headers get their own slot; groups separated by a gap
rows, ypos, headers = [], [], []
y = 0
for title, group in [("Model changes", model), ("Decision rules", decision)]:
    headers.append((y, title))
    y -= 1
    for lab, v, r in group:
        rows.append((y, lab, v, r))
        ypos.append(y)
        y -= 1
    y -= 0.35  # gap between groups

ax.axvline(0, lw=0.8, color="#c0c0c0", zorder=1)

for yy, lab, v, r in rows:
    ax.plot([v, r], [yy, yy], lw=1.1, color="#c9c9c9", zorder=2,
            solid_capstyle="round")
    ax.scatter(r, yy, s=30, facecolors=BLUE, edgecolors="white",
               linewidths=0.8, zorder=3)
    ax.scatter(v, yy, s=26, facecolors="none", edgecolors=GRAY,
               linewidths=1.0, zorder=4)

for yy, title in headers:
    ax.text(-4.55, yy, title, fontsize=7.5, fontweight="bold", color=INK,
            ha="left", va="center")

ax.set_yticks([yy for yy, *_ in rows])
ax.set_yticklabels([lab for _, lab, *_ in rows], fontsize=7)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-4.55, 10.3)
ax.set_ylim(y + 0.9, 0.55)
ax.set_xticks(range(-4, 11, 2))
ax.set_xlabel("gain (F1 points)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="x", length=2.5)

handles = [
    Line2D([], [], marker="o", ls="", markerfacecolor="white",
           markeredgecolor=GRAY, markersize=5, label="promised (validation)"),
    Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white",
           markersize=5.5, label="realised (leaderboard)"),
]
ax.legend(handles=handles, loc="lower right", frameon=False,
          borderaxespad=0.1, handletextpad=0.15, labelspacing=0.25)

fig.tight_layout(pad=0.15)
os.makedirs("/home/amal/Desktop/daleel2026/paper/figs", exist_ok=True)
fig.savefig("/home/amal/Desktop/daleel2026/paper/figs/transfer.pdf")
print("saved")
