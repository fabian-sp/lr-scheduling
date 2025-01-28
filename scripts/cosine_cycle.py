"""
Analyze cosine cycle length.

To align with Chinchilla paper (see Fig A1 in https://arxiv.org/pdf/2203.15556),
we anneal to 0.1 of base LR
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import os
import argparse

from scheduled import CosineSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11, FIGSIZE12

parser = argparse.ArgumentParser(description='Run cooldown length analysis.')
parser.add_argument('--ablation', action="store_true", help="Ablation for min bound.")
args = parser.parse_args()

#%%
ablation = args.ablation
# ablation = False

final_lr = 0.1
# plot convergence with base_lr tuned for each cycle, or the one tuned for cycle=1
tune_each_base_lr = False

plot_dir = '../plots/cosine_cycle/'

if ablation:
    rate_type = 'standard'
else:
    rate_type = 'refined'

if ablation:
    plot_dir = os.path.join(plot_dir, "ablation")

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)


# %matplotlib qt5
set_plot_aesthetics()

#%%#################################################################
#  Different durations
####################################################################
#  Compute optimal base LR and final rate

G = 1
D = 1
all_T = [400, 800, 1200]

all_cycle = [1.0, 1.5, 2, 4, 5]

base_lr = np.logspace(-8, -1, num=20, base=2)
rates = dict([(T, dict([(c, None) for c in all_cycle])) for T in all_T]) 

for T in all_T:
    for cycle in all_cycle:
        s = CosineSchedule(final_lr=final_lr,
                           steps=int(cycle*(T+1)),
                           base_lr=1.0
        )
        best_base_lr, best_rate = compute_optimal_base(s,
                                                       G=G, 
                                                       D=D, 
                                                       T=T,
                                                       type=rate_type
        )
        rates[T][cycle] = (best_base_lr, best_rate)

#%% Plot LR sweep for all cycle lengths 

T = 400

norm = Normalize(vmin=1, vmax=max(all_cycle), clip=False)

cmap = sns.color_palette("crest_r", as_cmap=True)

fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, c in enumerate(rates[T].keys()):
    S = CosineSchedule(final_lr=final_lr, steps=int(c*(T+1)), base_lr=1.0)
    this_curve = list()
    for _lr in base_lr:
        S.set_base_lr(_lr)
        _rate = S.compute_rate(grad_norms=G,
                               D=D,
                               T=T,
                               type=rate_type
        )
        this_curve.append(_rate)

    # Plot    
    col = cmap(norm(c))
    label = c

    ax.plot(base_lr, 
            this_curve,
            c=col,
            alpha=0.7,
    )
    ax.scatter([rates[T][c][0]], rates[T][c][1], 
                facecolors=col, 
                edgecolors='k', 
                zorder=5,
                label=label
    )

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
             ax=ax,
             orientation='vertical',
             location='right',
             ticks=[1,2,4],   
             label=r'Cycle length'
)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"Base learning-rate $\gamma$")
ax.set_ylabel(r"Final bound $\Omega_T$")

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

fig.subplots_adjust(top=0.985,
                    bottom=0.155,
                    left=0.16,
                    right=1.0,
)

fig.savefig(os.path.join(plot_dir, 'lr_sweep.pdf'))

#%% Vary cycle length, and plot optimal base_lr + rate

T = 400
time = np.arange(1, T+1)

cmap = sns.color_palette("crest_r", as_cmap=True)
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE12)

for j, c in enumerate(rates[T].keys()):
    S = CosineSchedule(final_lr=final_lr, steps=int(c*(T+1)), base_lr=1.0)
    best_base_lr, best_rate = rates[T][c]
    
    # compute full rate for optimal base LR
    if tune_each_base_lr:
        S.set_base_lr(best_base_lr)
    else:
        S.set_base_lr(rates[T][1.0][0])
    
    rate = [S.compute_rate(grad_norms=G, 
                           D=D, 
                           T=t,
                           type=rate_type) for t in time]

    # Plot schedule
    ax = axs[0]
    ax.plot(np.arange(1,T+2),       # t in [1, T+1]
            S.schedule[:T+1],
            c=cmap(norm(c)),
            lw=2
    )

    # Plot rate
    ax = axs[1]
    ax.plot(time,      # t in [1, T]
            rate,                   
            c=cmap(norm(c)),
            lw=2
    )

axs[0].set_xlabel(r'Iteration $t$')
axs[0].set_xlim(time[0], )
if tune_each_base_lr:
    axs[0].set_ylabel(r'Tuned learning rate $\gamma^\star \eta_t$')
    axs[1].set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
else:
    axs[0].set_ylabel(r'Learning rate $\gamma \eta_t$')
    axs[1].set_ylabel(r'Bound $\Omega_t$')

axs[0].grid(axis='both', lw=0.2, ls='--', zorder=0)

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
right=0.995,
hspace=0.2,
wspace=0.2
)

fig.savefig(os.path.join(plot_dir, 'convergence_multiple_cycle.pdf'))
