"""
Real loss curves for horizon transfer 50k --> 100/200k.

Run this via:

cd reanalysis/
python analysis_horizon_transfer.py --model_size 124m 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import argparse
from data_utils import load_multiple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

# %% Setup

parser = argparse.ArgumentParser(description='Plot reanalysis of horizon transfer.')
parser.add_argument('--model_size', nargs='?', type=str, default='124m', help="Model size (124m or 210m).")
args = parser.parse_args()

model_size = args.model_size
# model_size = "124m"

plot_dir = "../plots/reanalysis/horizon_transfer/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

# %% Configure runs to analyze

iterations = [50000, 100000, 200000]

wsd_config_list = [
    {
        "iterations": iter,
        "lr": 0.001,
        "scheduler": "wsd",
        "decay_type": "linear",
        "wsd_fract_decay": 0.2,
        "model": model,
    }
    for iter in iterations
    for model in [f"{model_size}_horizon"]
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

config_list =  wsd_config_list  + wsd_twostage_config_list

# Load data
df, config_df = load_multiple(config_list, data_dir="../data/horizon_transfer")


# for 210m, the 200k wsd run only logs from 80k onwards
# so we need to manually insert the logs of the 100k run for iter between 40k and 80k
if model_size == "210m":
    run100k = df[df.id == "wsd_lr0.001_iter100000_linear_0.2"]
    run200k = df[df.id == "wsd_lr0.001_iter200000_linear_0.2"]

    to_insert = run100k[run100k.iter.between(40_000, 79_999)]
    to_insert.loc[:, "id"] = "wsd_lr0.001_iter200000_linear_0.2"

    to_delete = run200k[run200k.iter.between(40_000, 79_999)]

    print("Length before manipulation:", len(df))
    df = df.drop(to_delete.index, axis=0)
    print("Length after deleting 40k-50k:", len(df))
    df = pd.concat([df, to_insert]).sort_values(["id", "iter"]).reset_index(drop=True)
    print("Length after inserting 40k-80k:", len(df))

#%% Plot loss curves

fig, ax = plt.subplots(figsize=FIGSIZE11)

# Color palettes for each scheduler type
palettes = {
    "wsd_twostage": sns.color_palette("Greens", len(iterations) + 1)[1:],
    "wsd": [matplotlib.colors.to_rgb("grey")]
}

axins = ax.inset_axes([0.12, 0.05, 0.1, 0.15])  # [x, y, width, height] in relative coordinates
axins2 = ax.inset_axes([0.62, 0.05, 0.1, 0.15])  # [x, y, width, height] in relative coordinates

for id in df.id.unique():
    this = df[df.id == id]
    this_iter = config_df.loc[id, "iterations"]
    this_sched = config_df.loc[id, "scheduler"]
    alpha = 1.0
    # short run up to 50k is labeled as wsd (not wsd_twostage)
    # but want to plot as dark green up to 40k, light green 40-50k
    if this_sched == 'wsd':
        col =  palettes['wsd_twostage'][0] if this_iter == 50_000 else palettes["wsd"][0]
        alpha = 0.7
    else:
        col = palettes[this_sched][iterations.index(this_iter)]

    # Plot validation loss
    this_no_nan = ~this["val_loss"].isna()

    # make sure that dark green is on top
    if this_sched == "wsd_twostage":
        zorder = 2 + iterations.index(this_iter)
    else:
        zorder = 1

    ax.plot(
            this.iter[this_no_nan],
            this.val_loss[this_no_nan],
            c=col,
            alpha=alpha,
            lw=2,
            zorder=zorder
    )
    
    # inset plots
    for _ax in [axins, axins2]:
        _ax.plot(
                this.iter[this_no_nan],
                this.val_loss[this_no_nan],
                c=col,
                alpha=alpha,
                lw=2,
        )

# Set the zoom bounds
if model_size == "124m":
    ax.set_ylim(2.8, 3.25)
    axins.set_ylim(2.86, 2.87)
    axins2.set_ylim(2.819, 2.83)
else:
    ax.set_ylim(2.65, 3.15)
    axins.set_ylim(2.745, 2.765)
    axins2.set_ylim(2.695, 2.72)


axins.set_xlim(98_000, 102_000)  # 98k-102k
axins2.set_xlim(198_000, 202_000)

# no x label
for _ax in [axins, axins2]:
    _ax.set_xticklabels([])
    ret = mark_inset(ax, _ax, loc1=1, loc2=4, ec="k", fc="none")

    # zorder does apply only on axis level
    # to get the inset markers in background, use this fix
    # https://stackoverflow.com/questions/56932448/how-to-control-zorder-and-clipping-with-mpl-toolkits-axes-grid1-inset-locator-ma
    for bc in ret[1:]:
        bc.remove()
        ax.add_patch(bc)
        bc.set_zorder(0)

    # no ticks for x axis
    _ax.set_xticks([])
    # yticklabel font smaller
    _ax.tick_params(axis='y', labelsize=8)

ax.set_xlabel(r"Iteration $t$ (unit $10^3$)")
ax.set_ylabel(r"Validation loss")
ax.set_xticks([50_000, 100_000, 150_000, 200_000],
              ['50', '100', '150', '200']
)

ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
labels = [
    r"$\tt wsd$ (adapted)"
    + "\n"
    + rf"$T\in [ {','.join([str(_iter // 1000) + 'k' for _iter in iterations])} ]$",
    r"$\tt wsd$ (same LR)"
    + "\n"
    + rf"$T\in [ {','.join([str(_iter // 1000) + 'k' for _iter in iterations])} ]$",
]
do_fancy_legend(
    ax,
    labels=labels,
    color_list=list(palettes.values()),
    fontsize=10,
    loc="upper right",
    lw=2
)

fig.subplots_adjust(
    top=0.991,
    bottom=0.158,
    left=0.132,
    right=0.995,
)

fig.savefig(os.path.join(plot_dir, f"{model_size}_horizon_transfer.pdf"))

#%% Plot schedules 

fig, ax = plt.subplots(figsize=FIGSIZE11)

# Color palettes for each scheduler type
palettes = {
    "wsd_twostage": sns.color_palette("Greens", len(iterations) + 1)[1:],
    "wsd": [matplotlib.colors.to_rgb("grey")]
}

for id in df.id.unique():
    this = df[df.id == id]
    this_iter = config_df.loc[id, "iterations"]
    this_sched = config_df.loc[id, "scheduler"]
    this_lr = config_df.loc[id, "lr"]
    alpha = 1.0
    if this_sched == 'wsd':
        col =  palettes['wsd_twostage'][0] if this_iter == 50_000 else palettes["wsd"][0]
        alpha = 0.7
    else:
        col = palettes[this_sched][iterations.index(this_iter)]

    # Plot validation loss
    this_no_nan = ~this["train_lr"].isna()

    # make sure that dark green is on top
    if this_sched == "wsd_twostage":
        zorder = 2 + iterations.index(this_iter)
    else:
        zorder = 1

    ax.plot(
            this.iter[this_no_nan],
            this.train_lr[this_no_nan] / this_lr,
            c=col,
            alpha=alpha,
            lw=2,
            zorder=zorder
    )
    
ax.set_ylim(0, 1.45)
ax.set_xlabel(r"Iteration $t$ (unit $10^3$)")
ax.set_ylabel(r"Schedule $\eta_t$")
ax.set_xticks([50_000, 100_000, 150_000, 200_000],
              ['50', '100', '150', '200']
)

ax.grid(axis="both", lw=0.2, ls="--", zorder=0)

do_fancy_legend(
    ax,
    labels=labels,
    color_list=list(palettes.values()),
    loc="upper right",
    lw=2,
    ncol=2,
    fontsize=8
)

fig.subplots_adjust(
    top=0.991,
    bottom=0.158,
    left=0.157,
    right=0.995,
)

fig.savefig(os.path.join(plot_dir, f"{model_size}_extended_schedule.pdf"))

#%% estimate how many more steps are needed to decrease by 0.01
# estimate slope via regression over 10k steps

delta = 0.01

def calculate_slope(df, start_iter, end_iter):
    mask = (df.iter >= start_iter) & (df.iter <= end_iter)
    mask &= ~df.val_loss.isna()
    window_data = df[mask]
    
    if len(window_data) == 0:
        return None
        
    # Normalize x values to avoid numerical issues
    x = (window_data.iter - start_iter).values
    y = window_data.val_loss.values
    
    # Linear regression
    slope, off = np.polyfit(x, y, 1)

    # quick plot
    plt.plot(x, y)
    plt.plot(x, slope*x + off)
    return slope, off

# 100k steps
baseline_id = "wsd_lr0.001_iter100000_linear_0.2"
this = df[df.id == baseline_id]
a, _ = calculate_slope(this, 64_000, 84_000)
print(f"Number steps for delta={delta} improvement:", delta/np.abs(a))

# 200k steps
baseline_id = "wsd_lr0.001_iter200000_linear_0.2"
this = df[df.id == baseline_id]
a, _ = calculate_slope(this, 144_000, 164_000)
print(f"Number steps for delta={delta} improvement:", delta/np.abs(a))

#%% scaling law utilities

# def implied_cost_chinchilla(D, delta, B, beta):
#     return (1/(D**beta) - delta/B)**(-1/beta)

# for D in [10.24e9, 20.48e9, 10.24e10, 20.48e10]:
#     D2 = implied_cost_chinchilla(D, delta=0.01, B=2085.43, beta=0.3658)
#     print(f"%.2E tokens: implied cost = %.2E tokens. ratio=%.4f" % (D, D2, D2/D))
