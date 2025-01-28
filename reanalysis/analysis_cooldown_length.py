"""
Re-analysis of cooldown length from real training data.

* Plots validation loss for different cooldown fractions and optimal LR (within sweep).
* Analysis how to transfer base LR from cooldown c to linear-decay (c=1).
We extrapolate the optimal base LR with curve fitting.

"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

from data_utils import load_multiple
from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

plot_dir = '../plots/reanalysis/analysis_cooldown_length/'

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%% Load data

# Specify here what data you want to load
ALL_LR = [0.0001, 0.0005, 0.001, 0.002, 0.003]
ALL_COOLDOWN = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.994]

wsd_config_list = [{"iterations": iter,
                    "lr": lr,
                    "scheduler": 'wsd',
                    "decay_type": "linear",
                    "wsd_fract_decay": c}
                    for c in ALL_COOLDOWN
                    for iter in [50_000] for lr in ALL_LR]

config_list = wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data")
print(f"Found {len(config_df)} logs. Expecting {7 * 5}.")

# Data cleaning
# Dtypes
for col in ['lr', 'wsd_fract_decay', 'final_train_loss', 'final_val_loss', 'final_val_perplexity']:
    config_df[col] = config_df[col].astype(float)

#%% Plot validation loss for entire grid

fig, axs = plt.subplots(1, 2, figsize=(8,3))

metric = "val_loss"

palette = sns.color_palette('rocket_r', 7)
rolling_window = (20 if metric == 'train_loss' else 1)

for id in df.id.unique():
    this = df[df.id == id]
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]
    this_cooldown = config_df.loc[id, "wsd_fract_decay"]

    col = palette[ALL_COOLDOWN.index(this_cooldown)]

    this_no_nan = ~this[metric].isna()

    ax = axs[0]
    ax.plot(this.iter[this_no_nan], 
            this[metric][this_no_nan].rolling(rolling_window).mean(), 
            c=col,
            alpha=0.5
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
    axs[0].set_ylim(2.9, 3.5)
axs[0].set_ylabel(metric)

axs[1].grid(which='both', lw=0.2, ls='--')
axs[1].set_ylabel("Learning rate")

fig.tight_layout()

#%% Plot convergence for best LR

fig, ax = plt.subplots(figsize=FIGSIZE11)

metric = "val_loss"
final_metric = "final_val_loss"

# NOTE:
# we cut out 0.6 because the curve is an outlier
# the loss is too high and only recovers late
COOLDOWNS_SHOWN = [0.1, 0.2, 0.4, 0.8, 0.9, 0.994]

cmap = sns.color_palette("flare", as_cmap=True)

rolling_window = 20 if metric == "train_loss" else 1

for i, cd in enumerate(COOLDOWNS_SHOWN):
    # get data in df that has this cooldown
    ids = config_df[config_df.wsd_fract_decay == cd]
    best_lr_config = ids.iloc[ids[final_metric].argmin()]
    id = best_lr_config.name
    print("Best id for cooldown {cd}:", id)
    this = df[df.id == id]
    
    # now plot
    this_no_nan = ~this[metric].isna()

    ax.plot(
        this.iter[this_no_nan],
        this[metric][this_no_nan].rolling(rolling_window).mean(),
        c=cmap(cd),
        label=f"{cd}-lr{best_lr_config['lr']}",
        lw=2,
    )
    print(f"cd={cd} best lr={best_lr_config['lr']}")

ax.set_ylim(2.9, 3.5)
ax.set_xlabel(r"Iteration $t$")
ax.set_ylabel(r"Validation Loss")
ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
labels = [
    f"Cooldown Fraction \n"
    + rf"$c\in [ {','.join([str(cd) for cd in COOLDOWNS_SHOWN])} ]$",
]
do_fancy_legend(
    ax,
    labels=labels,
    color_list=[palette],
    fontsize=10,
    loc="upper right",
)

fig.subplots_adjust(top=0.975,
bottom=0.149,
left=0.132,
right=0.982)

fig.savefig(os.path.join(plot_dir, "convergence_cooldown_length.pdf"))

#%% Plot cooldown sweep for fixed LR

fig, ax = plt.subplots(figsize=FIGSIZE11)
cpal = sns.color_palette("Blues_r", n_colors=8)

metric = 'final_val_loss'

few_base_lr = [0.003, 0.002, 0.001]

for j, lr in enumerate(few_base_lr):
    this_curve = list()
    this = config_df[config_df.lr == lr]
    assert len(this) == len(ALL_COOLDOWN)

    x = this.wsd_fract_decay
    y = this[metric]

    ax.plot(x,
            y,
            c=cpal[j],
            marker='o',
            lw=2,
            label=rf"$\gamma={lr}$"
    )

    ax.set_ylim(2.94, 3.02)
    ax.set_xlabel(r"Cooldown fraction")
    ax.set_ylabel(r"Final validation loss")
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)

ax.legend()

fig.subplots_adjust(top=0.965,
bottom=0.145,
left=0.155,
right=0.995,)

fig.savefig(os.path.join(plot_dir, "cooldown_range_multiple_lr.pdf"))

#%% LR sweep and fit curve

res = dict([(c, dict()) for c in ALL_COOLDOWN])

metric = 'final_val_loss'
cmap = sns.color_palette("flare", as_cmap=True)

fig, ax = plt.subplots(figsize=(FIGSIZE11))

for cd in ALL_COOLDOWN:

    this = config_df[config_df.wsd_fract_decay == cd]

    x = this.lr.values
    y = this[metric].values

    # inspired by theoretical bound
    fun = lambda gam, A, B, C: A/gam + B*gam + C
    sol = curve_fit(f=fun,
                    xdata=x,
                    ydata=y,
                    full_output=True,
                    bounds = ([0,0,0],
                              [np.inf, np.inf, np.inf]),
    )
    
    _x = np.logspace(-4, -2, 100)
    params = sol[0]
    _y = fun(_x, *params)

    print(f"MAD, cd={cd}", np.abs(fun(x, *params) - y).mean())

    color = cmap(cd)
    ax.scatter(x, y, color=color)
    ax.plot(_x, _y, c=color, label=cd)

    res[cd] = {'best_lr': _x[np.argmin(_y)],
               'params': params
    }

ax.set_ylim(2.94, 3.02)
ax.set_xscale('log')
ax.set_xlabel('Base learning rate')
ax.set_ylabel('Final val loss')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
do_fancy_legend(ax,
                labels=[f"Cooldown fraction {ALL_COOLDOWN}"],
                color_list=[[cmap(cd) for cd in ALL_COOLDOWN]],
                bbox_to_anchor=(1, 1.05),
                framealpha=0.99
)
fig.subplots_adjust(top=0.976,
                    bottom=0.159,
                    left=0.159,
                    right=0.991,)

fig.savefig(os.path.join(plot_dir, f'lr_sweep.pdf'))

#%% LR transfer

fig, ax = plt.subplots(figsize=(FIGSIZE11))

best_lr = dict([(cd, res[cd]['best_lr']) for cd in res.keys()])
y = np.log(best_lr[0.994]) - np.log(list(best_lr.values()))

label1 = r'$\ln(\gamma_{0.994}) - \ln(\gamma_{c})$'
label2 = r'Best LR $\gamma_{c}$'

col1 = "#252422"
col2 = "#eb5e28"

ax.plot(best_lr.keys(),
        y,
        c=col,
        marker='o',
        lw=2,
)
ax.set_xlabel(r'Cooldown fraction $c$')
ax.set_ylabel(label1)

ax.tick_params(direction='in', which='both')
ax.set_yticks(np.linspace(0,2,21), [], minor=True)

ax.set_xlim(0,1.02)
ax.set_ylim(-5e-2,1.75)
ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])

ax2 = ax.twinx()
ax2.plot(list(best_lr.keys()),
         list(best_lr.values()),
         c=col2,
         marker='o',
         lw=2
)
ax2.set_ylabel(label2)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))

# legend in same box
lines = [Line2D([0], [0], color=col1, lw=2),
         Line2D([0], [0], color=col2, lw=2)
]
ax2.legend(lines, [label1, label2])
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
fig.subplots_adjust(top=0.940,
                    bottom=0.138,
                    left=0.124,
                    right=0.872)

fig.savefig(os.path.join(plot_dir, f'best_lr.pdf'))
