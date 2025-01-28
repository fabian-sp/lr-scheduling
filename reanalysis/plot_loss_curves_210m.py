"""
Real loss curves for WSD and cosine for 210M models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from data_utils import load_multiple

from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

# %% Setup

plot_dir = "../plots/reanalysis/loss_curves_210m/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

# %% Configure runs to analyze

iterations = [22222, 33333, 44444]

cos_config_list = [
    {"iterations": iter, "lr": 0.001, "scheduler": "cos", "model": "210m"}
    for iter in iterations
]

wsd_config_list = [
    {
        "iterations": iter,
        "lr": 0.0005,
        "scheduler": "wsd",
        "decay_type": "linear",
        "wsd_fract_decay": 0.2,
        "model": "210m",
    }
    for iter in iterations
]

config_list = cos_config_list + wsd_config_list

# %% Load and plot data

df, config_df = load_multiple(config_list, data_dir="..//data")

fig, ax = plt.subplots(figsize=FIGSIZE11)

# Color palettes for each scheduler type
palettes = {
    "cos": sns.color_palette("Reds", len(iterations) + 2)[2:],
    "wsd": sns.color_palette("Blues", len(iterations) + 2)[2:],
}

for id in df.id.unique():
    this = df[df.id == id]
    this_iter = config_df.loc[id, "iterations"]
    this_sched = config_df.loc[id, "scheduler"]

    col = palettes[this_sched][iterations.index(this_iter)]

    # Plot validation loss
    this_no_nan = ~this["val_loss"].isna()
    ax.plot(
        this.iter[this_no_nan],
        this.val_loss[this_no_nan],
        c=col,
        # label=f"{this_sched}-{this_iter//1000}k",
        lw=2,
    )

ax.set_xlabel(r"Iteration $t$")
ax.set_ylabel(r"Validation Loss")
ax.set_ylim(2.8, 3.8)
ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
labels = [
    r"$\tt cosine$" +  "\n"
    + rf"$T\in [ {','.join([str(_iter // 1000) + 'k' for _iter in iterations])} ]$",
    r"$\tt wsd$" + "\n"
    + rf"$T\in [ {','.join([str(_iter // 1000) + 'k' for _iter in iterations])} ]$",
]
do_fancy_legend(
    ax,
    labels=labels,
    color_list=list(palettes.values()),
    fontsize=10,
    loc="upper right",
)

fig.subplots_adjust(top=0.976,
bottom=0.144,
left=0.132,
right=0.99,)

fig.savefig(os.path.join(plot_dir, "210m_horizons.pdf"))
