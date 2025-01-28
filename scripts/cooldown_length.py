"""
Analysis of cooldown length.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

from scheduled import WSDSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11, FIGSIZE12


parser = argparse.ArgumentParser(description='Run cooldown length analysis.')
parser.add_argument('--ablation', action="store_true", help="Ablation for min bound.")
parser.add_argument('-d', '--decay_type', type=str, default='linear', help="The type of decay for WSD. Choose linear (default) or sqrt.")
args = parser.parse_args()

#%%

ablation = args.ablation
decay_type = args.decay_type
# ablation = False
# decay_type = 'linear'

plot_dir = '../plots/cooldown_length/'

if ablation:
    rate_type = 'standard'
else:
    rate_type = 'refined'

if ablation:
    plot_dir = os.path.join(plot_dir, "ablation")

if decay_type != 'linear':
    plot_dir = os.path.join(plot_dir, decay_type)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)


# %matplotlib qt5
set_plot_aesthetics()

#%% Setup and LR sweep

G = 1
D = 1
all_T = [200, 300, 400, 500, 600]
all_D = dict(zip(all_T, np.ones(len(all_T)) * D))

base_lr = np.logspace(-8, -1, num=20, base=2)

rates = dict([(T, None) for T in all_T])

for T in all_T:
    all_cooldown = np.linspace(1/T, T/(T+1), 10) # should have 1 up to T cooldown steps
    rates[T] = dict([(_cd, None) for _cd in all_cooldown])
    for j, _cd in enumerate(all_cooldown):
        S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=_cd, base_lr=1.0, decay_type=decay_type)
        best_base_lr, best_rate = compute_optimal_base(S, 
                                                       G=G, 
                                                       D=all_D[T], 
                                                       T=T,
                                                       type=rate_type
        )
        rates[T][_cd] = (best_base_lr, best_rate)

#%% Plot LR sweep for all cooldown lengths 
T = 400

cmap = sns.color_palette("flare", as_cmap=True)
fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, _cd in enumerate(rates[T].keys()):
    S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=_cd, base_lr=1.0, decay_type=decay_type)
    this_curve = list()
    for _lr in base_lr:
        S.set_base_lr(_lr)
        _rate = S.compute_rate(grad_norms=G,
                               D=all_D[T],
                               T=T,
                               type=rate_type
        )
        this_curve.append(_rate)

    # Plot    
    col = cmap(_cd)
    label = _cd

    ax.plot(base_lr, 
            this_curve,
            c=col,
            alpha=0.5,
    )
    ax.scatter([rates[T][_cd][0]], rates[T][_cd][1], 
                facecolors=col, 
                edgecolors='k', 
                zorder=5,
                label=label
    )

plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap='flare'), 
             ax=ax,
             orientation='vertical',
             location='right',
             ticks=[0, 0.5, 1],
             label=r'Cooldown fraction'
)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"Base learning-rate $\gamma$")
ax.set_ylabel(r"Final bound $\Omega_T$")

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

fig.subplots_adjust(top=0.975,
                    bottom=0.155,
                    left=0.16,
                    right=0.965,
)

fig.savefig(os.path.join(plot_dir, 'lr_sweep.pdf'))

#%% Vary cooldown length, and plot optimal base_lr + rate
T = 400
time = np.arange(1, T+1)

cmap = sns.color_palette("flare", as_cmap=True)
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE12)

for j, _cd in enumerate(rates[T].keys()):
    S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=_cd, base_lr=1.0, decay_type=decay_type)
    
    best_base_lr, best_rate = rates[T][_cd]
    print(f"Cooldown {_cd}, best base lr {best_base_lr}")

    # compute full rate for optimal base LR
    S.set_base_lr(best_base_lr)
    rate = [S.compute_rate(grad_norms=G, 
                           D=all_D[T], 
                           T=t,
                           type=rate_type) for t in time]

    # Plot schedule
    ax = axs[0]
    ax.plot(np.arange(1,T+2),       # t in [1, T+1]
            S.schedule,
            c=cmap(_cd),
            lw=2
    )

    # Plot rate
    ax = axs[1]
    ax.plot(time,      # t in [1, T]
            rate,                   
            c=cmap(_cd),
            lw=2
    )

axs[0].set_xlabel(r'Iteration $t$')
axs[0].set_xlim(time[0], )
axs[0].set_ylabel(r'Tuned learning rate $\gamma^\star \eta_t$')
axs[0].grid(axis='both', lw=0.2, ls='--', zorder=0)

axs[1].set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
axs[1].set_xlabel(r'Iteration $t$')
axs[1].set_xlim(time[0], )
if ablation:
    axs[1].set_ylim([0.045, 0.35])
else:
    axs[1].set_ylim([0.08, 0.35])
axs[1].grid(axis='both', lw=0.2, ls='--', zorder=0)

fig.subplots_adjust(top=0.97,
                    bottom=0.155,
                    left=0.08,
                    right=0.995
)

fig.savefig(os.path.join(plot_dir, 'convergence_multiple_cooldown.pdf'))

#%% Final bound vs. cooldown fraction for multiple base LRs (cf Figure 5 in https://arxiv.org/pdf/2405.18392)

T = 400

fig, ax = plt.subplots(figsize=FIGSIZE11)
cpal = sns.color_palette("Blues_r", n_colors=8)

if decay_type == 'linear':
    few_base_lr = [0.02, 0.04, 0.05, 0.07][::-1]
elif decay_type == 'sqrt':
    few_base_lr = [0.03, 0.05, 0.07, 0.09][::-1]

for j, _lr in enumerate(few_base_lr):
    this_curve = list()
    for _cd in rates[T].keys():
        S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=_cd, base_lr=1.0, decay_type=decay_type)
        S.set_base_lr(_lr)
        _rate = S.compute_rate(grad_norms=G,
                               D=all_D[T],
                               T=T,
                               type=rate_type
        )
        this_curve.append(_rate)

    # Plot    
    col = cpal[j]

    ax.plot(rates[T].keys(), 
            this_curve,
            lw=2.5,
            marker='o',
            markersize=5,
            c=col,
            alpha=1.0,
            label=r"$\gamma=%.2f$" % _lr
    )

ax.set_xlim([0,1])
ax.set_xlabel(r"Cooldown fraction")
ax.set_ylabel(r"Final bound $\Omega_T$")

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend(ncols=2)

fig.subplots_adjust(top=0.98,
                    bottom=0.159,
                    left=0.163,
                    right=0.974
)
fig.savefig(os.path.join(plot_dir, 'cooldown_range_multiple_lr.pdf'))
