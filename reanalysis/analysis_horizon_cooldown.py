"""
Real loss curves for horizon transfer 50k --> 100/200k by adapting cooldown.

Run this via:

cd reanalysis/
python analysis_horizon_cooldown.py --model_size 124m 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from data_utils import load_multiple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

# %% Setup

plot_dir = "../plots/reanalysis/horizon_transfer/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

parser = argparse.ArgumentParser(description='Plot reanalysis of horizon transfer.')
parser.add_argument('--model_size', nargs='?', type=str, default='124m', help="Model size (124m or 210m).")
args = parser.parse_args()

model_size = args.model_size
# model_size = "124m"

# %% Configure runs to analyze

iterations = [50000, 100000]

wsd_config_list = [
    {
        "iterations": iter,
        "lr": 0.001,
        "scheduler": "wsd",
        "decay_type": "linear",
        "wsd_fract_decay": fract,
        "model": model,
    }
    for iter in iterations
    for model in [f"{model_size}_horizon"]
    for fract in [0.2]
]

wsd_adapted_cooldown_config_list = [
    {
        "iterations": iter,
        "lr": 0.001,
        "scheduler": "wsd",
        "decay_type": "linear",
        "wsd_fract_decay": fract,
        "model": model,
    }
    for iter in iterations[1:]
    for model in [f"{model_size}_horizon"]
    for fract in [0.6]
]

wsd_twostage_config_list = [
    {
        "iterations": iter,
        "lr": 0.001,
        "scheduler": "wsd_twostage",
        "decay_type": "linear",
        "wsd_fract_decay": 0.2,
        "model": model,
    }
    for iter in iterations[1:]
    for model in [f"{model_size}_horizon"]
]

config_list = wsd_config_list + wsd_adapted_cooldown_config_list +  wsd_twostage_config_list

df, config_df = load_multiple(config_list, data_dir="../data/horizon_transfer")

# %% Load and plot data

fig, ax = plt.subplots(figsize=FIGSIZE11)

# Color palettes for each scheduler type
palettes = {
    ('wsd', 0.2):  sns.color_palette("Greys_r", (len(iterations)+1))[:-1],
    ('wsd_twostage', 0.2): sns.color_palette("Greens_d", 1),
    ('wsd', 0.6): sns.color_palette("Blues_d", 1)
}

axins = ax.inset_axes(
    [0.12, 0.05, 0.2, 0.3]
)  # [x, y, width, height] in relative coordinates

for id in df.id.unique():
    this = df[df.id == id]
    this_iter = config_df.loc[id, "iterations"]
    this_sched = config_df.loc[id, "scheduler"]
    this_fract = config_df.loc[id, "wsd_fract_decay"]
    alpha = 1.0
    palette = palettes[(this_sched, this_fract)]
    col = palette[-min(iterations.index(this_iter), len(palette) - 1)]

    # Plot validation loss
    this_no_nan = ~this["val_loss"].isna()

    # make sure green on top
    zorder = 2 if this_sched == "wsd_twostage" else 1

    ax.plot(
            this.iter[this_no_nan],
            this.val_loss[this_no_nan],
            c=col,
            alpha=alpha,
            lw=2,
            zorder=zorder
    )
    axins.plot(
        this.iter[this_no_nan],
        this.val_loss[this_no_nan],
        c=col,
        alpha=alpha,
        lw=2,
    )

zoomin_ylim = (2.73, 2.77) if model_size == "210m" else (2.85, 2.87)

# Set the zoom bounds
axins.set_xlim(98_000, 101_000)  # 98k-102k
# no x ticks
axins.set_xticklabels([])
axins.set_xticks([])
axins.set_ylim(*zoomin_ylim)
mark_inset(ax, axins, loc1=1, loc2=4, ec="0.5", fc="none")
# yticklabel font smaller
axins.tick_params(axis="y", labelsize=8)

ylim = (2.65, 3.25) if model_size == "210m" else (2.8, 3.35)
ax.set_xlabel(r"Iteration $t$ (unit $10^3$)")
ax.set_ylabel(r"Validation loss")
ax.set_xticks([25_000, 50_000, 75_000, 100_000],
              ['25', '50', '75', '100',]
)
ax.set_ylim(*ylim)
ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
labels = [
    r"$\tt wsd$, "
    + rf"$T\in [ {','.join([str(_iter // 1000) + 'k' for _iter in iterations])} ]$",
    r"$\tt wsd$ (adapted schedule)",
    r"$\tt wsd$ (adapted cooldown)",
]
do_fancy_legend(
    ax,
    labels=labels,
    color_list=list(palettes.values()),
    fontsize=10,
    loc="upper right",
    handleheight=1.0,
    lw=2.0
)

fig.subplots_adjust(
    top=0.99,
    bottom=0.154,
    left=0.137,
    right=0.991,
)

fig.savefig(os.path.join(plot_dir, f"{model_size}_horizon_cooldown.pdf"))
