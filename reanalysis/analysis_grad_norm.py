import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

from data_utils import load_multiple
from scheduled import CosineSchedule, WSDSchedule
from scheduled.utils import FIGSIZE11, FIGSIZE12, set_plot_aesthetics

plot_dir = "../plots/reanalysis/grad_norm/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%% Specify here what data you want to load

ALL_LR = [0.0001, 0.0005, 0.001, 0.002, 0.003]

# data also available for 0.2 cooldown
wsd_config_list = [{"iterations": 50_000, "lr": lr, "scheduler": 'wsd', "decay_type": "linear", "wsd_fract_decay": 0.2} 
                    for lr in ALL_LR]

cos_config_list = [{"iterations": 50_000, "lr": lr, "scheduler": 'cos'} 
                    for lr in ALL_LR]

config_list = cos_config_list + wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data/grad_norm")

# %% Plot gradient norm

cmap_dict = {"wsd": "rocket",
             "cos": "mako"
}

palettes = dict((sched, sns.color_palette(cmap_dict[sched], len(ALL_LR)+2)[1:]) for sched in cmap_dict.keys())

label_dict = {"grad_norm": r"Gradient norm $\|g_t\|$",
              "train_loss": r"Batch loss $f(x_t, \xi_t)$"
}

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE12)

metric = "grad_norm"
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
            lw=1.4,
            c=col
    )

    ax = axs[1]
    # at cooldown start we have single nan log, should be correct to just use ffill
    ax.plot(this.iter, this.train_lr.ffill(), c=col)

axs[0].grid(which='both', lw=0.2, ls='--')
if metric == "grad_norm":
    axs[0].set_yscale("log")
if metric == "train_loss":
    axs[0].set_ylim(2.9, 4.1)

axs[0].set_ylabel(label_dict.get(metric, metric))
axs[0].set_xlabel(r"Iteration $t$")

axs[1].grid(which='both', lw=0.2, ls='--')
axs[1].set_ylabel(r"Learning rate $\gamma \eta_t$")
axs[1].set_xlabel(r"Iteration $t$")

fig.subplots_adjust(top=0.99,
bottom=0.149,
left=0.081,
right=0.993,
wspace=0.25)

fig.savefig(os.path.join(plot_dir, f"log_{metric}.pdf"))

#%%  Fit (mean) grad norm as function of lr

# Collect data
mean_grad_norm = {'cos': dict(), 'wsd': dict()}
for id in df.id.unique():
    this = df[df.id == id]
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]
    this_no_nan = ~this["grad_norm"].isna()
    y = this["grad_norm"][this_no_nan]
    mean_grad_norm[this_sched][this_lr] = np.mean(y)

# Fit and plot
fig, ax = plt.subplots(1,1,figsize=FIGSIZE11)
for sched in ['cos', 'wsd']:
    x = np.array(list(mean_grad_norm[sched].keys()))
    y = np.array(list(mean_grad_norm[sched].values()))

    # move x on log axis
    x = np.log10(x)

    fun = lambda log_gam, G, B, beta: B * np.exp(-log_gam*beta)
    res = curve_fit(f=fun,
                    xdata=x,
                    ydata=y,
                    full_output=True,
                    bounds = ([0, 0, -np.inf],
                              [np.inf, np.inf, np.inf]),
                    maxfev=10000
    )
    _x = np.linspace(-4.1, -1.9, 100)
    params = res[0]
    print(sched, params)
    _y = fun(_x, *params)
    
    ax.plot(x,
            y,
            c=palettes[sched][3],
            lw=0,
            marker="o",
    )
    label = r"$G_t \sim \gamma^{-%.2f} $" % params[-1]
    label = sched + ", " + label
    ax.plot(_x,
            _y,
            c=palettes[sched][3],
            lw=2,
            ls="--",
            label=label
    )

ax.legend()
ax.set_xlabel(r'Base LR $\log_{10}(\gamma)$')
ax.set_ylabel('Mean gradient norm')
ax.grid(which='both', lw=0.2, ls='--')

fig.subplots_adjust(top=0.985,
bottom=0.165,
left=0.14,
right=0.99)

fig.savefig(os.path.join(plot_dir, f"grad_norm_scaling.pdf"))

# %% Fit grad norm as function of time 

cutoff = 0

fig, axs = plt.subplots(1, 2, figsize=(8,3))
LR_FILTER = [0.0001, 0.0005, 0.001, 0.002]

for id in df.id.unique():
    this = df[(df.id == id) & (df.iter > cutoff)]
    this_no_nan = ~this["grad_norm"].isna()
    target = this[this_no_nan].grad_norm.values
    time = this[this_no_nan].iter.values
    T = time.max()
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]

    if this_lr not in LR_FILTER:
        continue

    fun = lambda t, a, b, alpha, p, q, c: a * np.exp(alpha * (t/T * this_lr) ** q) + b * ((t/T)**p) + c
    
    p0 = {
        "a": 10.0,
        "b": 1.0,
        "alpha": -1,
        "p": 0.0,
        "q": 1.0,
        "c": 0.1,
    }
    p0_arr = np.array(list(p0.values()))

    lb = {
        "a": 0.0,
        "b": 0.0,
        "alpha": -np.inf,
        "p": 0.0,
        "q": 0.0,
        "c": 0.0,
    }
    ub = {
        "a": np.inf,
        "b": 1.0,
        "alpha": 0.0,
        "p": np.inf,
        "q": np.inf,
        "c": np.inf,
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
                lw=0.3
        )

        pred = fun(time, *params.values())
        label = this_sched + ", $\gamma$ = %.0e" % this_lr
        ax.plot(time,
                pred,
                c=col,
                ls='-',
                lw=2,
                label=label
        )
    except:
        print(f"No successful fit for {id}")

for ax in axs:
    ax.set_yscale('log')
    ax.set_xlabel(r"Iteration $t$")
    ax.grid(which='both', lw=0.2, ls='--')    
    ax.legend(loc="upper right")

axs[0].set_ylabel(r"Gradient norm $\|g_t\|$")

fig.subplots_adjust(top=0.99,
bottom=0.155,
left=0.085,
right=0.99,
hspace=0.2,
wspace=0.2)

fig.savefig(os.path.join(plot_dir, f"grad_norm_fit.pdf"))

# %% Scatter plot lr vs grad_norm

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE11)

this = df[~df.grad_norm.isna()]

# is -1 for wsd, 1 for cos
ix_wsd = np.array([2*int("cos" in _id) -1 for _id in this.id])

x = this.train_lr * ix_wsd
# x = this.train_lr


ax.scatter(x,
           this.grad_norm,
           #c="k",
           c=this.iter/ 50_000,
           cmap="coolwarm",
           s=1,
           alpha=0.3
)

ax.set_xlabel(r'Learning rate')
ax.set_ylabel('Gradient norm')
ax.set_yscale("log")
ax.grid(which='both', lw=0.2, ls='--')

fig.subplots_adjust(top=0.985,
bottom=0.165,
left=0.165,
right=0.99)

fig.savefig(os.path.join(plot_dir, f"grad_norm_scatter.pdf"))

# %%
