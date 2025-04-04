"""
Real loss curves for WSD and cosine for OpenWebText2 runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from data_utils import load_multiple

from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

# %% Setup

parser = argparse.ArgumentParser(description='Plot loss curves for Openwebtext runs.')
parser.add_argument('-m', '--model', default="93m", type=str, help="Model size, have 60m, 93m, 166m.")
args = parser.parse_args()

MODEL = args.model

plot_dir = "../plots/reanalysis/loss_curves_owt/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

# %% Configure runs to analyze

cos_lrs_iters = {"60m": [(0.002, 7_500), (0.002, 12_500), (0.002, 17_500)],
                "93m": [(0.002, 10_000), (0.002, 17_500), (0.002, 25_000)],
                "166m": [(0.001, 15_000), (0.001, 25_000), (0.001, 30_000)]
}

wsd_lrs_iters = {"60m": [(0.001, 7_500), (0.001, 12_500), (0.001, 17_500)],
                "93m": [(0.001, 10_000), (0.001, 17_500), (0.001, 25_000)],
                "166m": [(0.0005, 15_000), (0.0005, 25_000), (0.0005, 30_000)]
}

cos_config_list = [
    {"iterations": iter, "lr": lr, "scheduler": "cos", "model": MODEL, "dataset": "openwebtext"}
    for lr, iter in cos_lrs_iters[MODEL]
]

wsd_config_list = [
    {
        "iterations": iter,
        "lr": lr,
        "scheduler": "wsd",
        "decay_type": "linear",
        "wsd_fract_decay": 0.2,
        "model": MODEL,
        "dataset": "openwebtext"
    }
    for lr, iter in wsd_lrs_iters[MODEL]
]

config_list = cos_config_list + wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data")
iterations =list(pd.to_numeric(config_df.iterations).sort_values().unique())

# %% Load and plot data

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
        lw=2,
    )

ax.set_xlabel(r"Iteration $t$")
ax.set_ylabel(r"Validation Loss")
ax.set_ylim(2.8, 3.8)
ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
labels = [
    r"$\tt cosine$" +  "\n"
    + rf"$T\in [ {','.join([str(_iter / 1000) + 'k' for _iter in iterations])} ]$",
    r"$\tt wsd$" + "\n"
    + rf"$T\in [ {','.join([str(_iter / 1000) + 'k' for _iter in iterations])} ]$",
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
right=0.98,)

fig.savefig(os.path.join(plot_dir, f"{MODEL}.pdf"))
