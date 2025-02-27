"""
Plots loss curves and schedules for the 124M model training.
Not needed for any of the paper plots. 
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from data_utils import load_multiple


color_dict = {"wsd": (0.19937337, 0.13719028, 0.27040111), 
              "cos": (0.19937337, 0.13719028, 0.27040111)
}
cmap_dict = {"wsd": "rocket",
             "cos": "mako"
}

palettes = dict((sched, sns.color_palette(cmap_dict[sched], 5)) for sched in cmap_dict.keys())


# Specify here what data you want to load
ALL_LR = [0.0001, 0.0005, 0.001, 0.002]

wsd_config_list = [{"iterations": iter, "lr": lr, "scheduler": 'wsd', "decay_type": "linear", "wsd_fract_decay": 0.2} 
                    for iter in [25_000, 50_000] for lr in ALL_LR]

cos_config_list = [{"iterations": iter, "lr": lr, "scheduler": 'cos'} 
                    for iter in [25_000, 50_000] for lr in ALL_LR]

config_list = cos_config_list + wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data")

#%% Plot validation loss
# %matplotlib qt5

fig, axs = plt.subplots(1, 2, figsize=(8,3))

metric = "val_loss"

rolling_window = (20 if metric == 'train_loss' else 1)

for id in df.id.unique():
    this = df[df.id == id]
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]

    col = palettes[this_sched][ALL_LR.index(this_lr)]

    this_no_nan = ~this[metric].isna()

    ax = axs[0]
    ax.plot(this.iter[this_no_nan], 
            this[metric][this_no_nan].rolling(rolling_window).mean(), 
            c=col
    )

    # add final_metrics, which are computed over full set
    ax.scatter([config_df.loc[id, "iterations"]],
               [config_df.loc[id, f"final_{metric}"]],
               color=col,
    )

    ax = axs[1]
    # at cooldown start we have single nan log, should be correct to just use ffill
    ax.plot(this.iter, this.train_lr.ffill(), c=col)
    

axs[0].grid(which='both', lw=0.2, ls='--')
if metric == 'val_perplexity':
    axs[0].set_ylim(15, 30)
if metric == 'val_loss':
    axs[0].set_ylim(2.9, 4)
axs[0].set_ylabel(metric)

axs[1].grid(which='both', lw=0.2, ls='--')
axs[1].set_ylabel("Learning rate")

fig.tight_layout()