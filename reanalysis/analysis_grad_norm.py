import os
import json
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from data_utils import load_multiple

from scheduled import CosineSchedule, WSDSchedule


cmap_dict = {"wsd": "rocket",
             "cos": "mako"
}

palettes = dict((sched, sns.color_palette(cmap_dict[sched], 5)) for sched in cmap_dict.keys())

# Specify here what data you want to load
ALL_LR = [0.0001, 0.0005, 0.001, 0.002]

wsd_config_list = [{"iterations": 50_000, "lr": lr, "scheduler": 'wsd', "decay_type": "linear", "wsd_fract_decay": 0.2} 
                    for lr in ALL_LR]

cos_config_list = [{"iterations": 50_000, "lr": lr, "scheduler": 'cos'} 
                    for lr in ALL_LR]

config_list = cos_config_list + wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data/grad_norm")

# %% Plot gradient norm
# %matplotlib qt5

fig, axs = plt.subplots(1, 3, figsize=(12,3))

metric = "grad_norm"
mean_grad_norm = {'cos': dict(), 'wsd': dict()}

rolling_window = (20 if metric in ['train_loss', 'grad_norm'] else 1)

for id in df.id.unique():
    this = df[df.id == id]
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]

    col = palettes[this_sched][ALL_LR.index(this_lr)]

    this_no_nan = ~this[metric].isna()
    y = this[metric][this_no_nan]

    ax = axs[0]
    ax.plot(this.iter[this_no_nan], 
            y.rolling(rolling_window).mean(), 
            c=col
    )

    ax = axs[1]
    # at cooldown start we have single nan log, should be correct to just use ffill
    ax.plot(this.iter, this.train_lr.ffill(), c=col)


    mean_grad_norm[this_sched][this_lr] = np.mean(y)


axs[0].grid(which='both', lw=0.2, ls='--')
axs[0].set_yscale("log")
# axs[0].set_ylim(2.9, 4)
axs[0].set_ylabel(metric)

axs[1].grid(which='both', lw=0.2, ls='--')
axs[1].set_ylabel("Learning rate")

for sched in ['cos', 'wsd']:
    axs[2].plot(mean_grad_norm[sched].keys(),
            mean_grad_norm[sched].values(),
            c=palettes[sched][3],
            lw=2,
            label=sched)
axs[2].legend()
axs[2].set_xlabel('base learning rate')
axs[2].set_ylabel('Mean gradient norm')
axs[2].grid(which='both', lw=0.2, ls='--')

fig.tight_layout()

# %% Fit grad norm

from scipy.optimize import curve_fit
cutoff = 300

fig, axs = plt.subplots(1, 2, figsize=(8,3))

for id in df.id.unique():
    this = df[(df.id == id) & (df.iter > cutoff)]
    this_no_nan = ~this["grad_norm"].isna()
    target = this[this_no_nan].grad_norm.values
    time = this[this_no_nan].iter.values
    T = time.max()
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]

    # fun = lambda t, a, b, alpha, p, q, c, d: a * np.exp(alpha * t) + b * (t**p) + c / this_lr + d
    fun = lambda t, a, b, alpha, p, q, c, d: a * np.exp(alpha * (t/T * this_lr) ** q) + b * ((t/T)**p) + c / this_lr + d
    # fun = lambda t, a, b, alpha, p, q, c, d: a*(t**alpha) + b*(t**p) + c

    p0 = {
        "a": 10.0,
        "b": 1.0,
        "alpha": -10,
        "p": 0.0,
        "q": 0.1,
        "c": 0.0,
        # "c": 1.0,
        "d": 0.0,
    }
    p0_arr = np.array(list(p0.values()))

    lb = {
        "a": 9.0,
        "b": 0.0,
        "alpha": -20,
        "p": 0.0,
        "q": 0.0,
        "c": 0.0,
        "d": 0.0,
    }
    ub = {
        "a": 10.0,
        "b": 1.0,
        "alpha": -10,
        "p": 0.5,
        "q": np.inf,
        "c": 2,
        "d": 1.0,
    }

    try:
        res = curve_fit(f=fun,
                    xdata=time,
                    ydata=target,
                    p0=p0_arr,
                    full_output=True,
                    bounds = (list(lb.values()),
                            list(ub.values())),
                    maxfev=1000000
        )
        params = dict(zip(p0.keys(), res[0]))
        print(id, {k: v.item() for k,v in params.items()})
        col = palettes[this_sched][ALL_LR.index(this_lr)]

        ax = axs[0] if this_sched == 'cos' else axs[1]

        ax.plot(time,
                target,
                c=col,
                ls='-',
                lw=0.4
        )

        pred = fun(time, *params.values())
        ax.plot(time,
                pred,
                c=col,
                ls='-',
                lw=2,
                label=f"{this_sched}-{this_lr}"
        )
    except:
        print(f"No successful fit for {id}")


for ax in axs:
    ax.set_yscale('log')
    ax.grid(which='both', lw=0.2, ls='--')    
    ax.legend()
