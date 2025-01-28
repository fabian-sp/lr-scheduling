"""
Plot convergence rate and LR sweep for different schedulers.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from scheduled import CosineSchedule, WSDSchedule, SqrtSchedule, ConstantSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11, FIGSIZE12


parser = argparse.ArgumentParser(description='Run different schedulers.')
parser.add_argument('--ablation', action="store_true", help="Ablation for min bound.")
args = parser.parse_args()

#%% Setup

plot_dir = '../plots/schedule_comparison/'
ablation = args.ablation
# ablation = False

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

T = 400
G = 1
D = 1

time = np.arange(1, T+1)

# Schedule objects
schedules = {'wsd-0.2': WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2),
             'linear': WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=1.0),
             '1/sqrt': SqrtSchedule(steps=T+1),
             'constant': ConstantSchedule(steps=T+1),
             'cosine': CosineSchedule(final_lr=0.0, steps=T+1),
             '1-sqrt': WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=1.0, decay_type='sqrt')
}

rates = dict()

for name, s in schedules.items():
    best_base_lr, best_rate = compute_optimal_base(s,
                                                   G=G,
                                                   D=D,
                                                   T=T,
                                                   type=rate_type
    )
    rates[name] = (best_base_lr, best_rate)

#%% Plot LR sensitivity

base_lr = np.logspace(-8, 1, num=50, base=2)

colors = {'wsd-0.2': "#7587A0",
          'linear': "#414455",
          '1/sqrt': "#9B9D89",
          'constant': "#CFB294",
          'cosine': "#6B5756",
          '1-sqrt': "#A83F39"
}

fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, (name, s) in enumerate(schedules.items()):
    this_curve = list()
    for _lr in base_lr:
        s.set_base_lr(_lr)
        _rate = s.compute_rate(grad_norms=G,
                                D=D,
                                T=T,
                                type=rate_type
        )
        this_curve.append(_rate)

    # Plot    
    col = colors[name]
    label = name

    ax.plot(base_lr, 
            this_curve,
            lw=2,
            c=col,
            alpha=0.8,
            label=label
    )
    ax.scatter([rates[name][0]], rates[name][1],
                facecolors=col, 
                edgecolors='k', 
                zorder=5
    )

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(9e-2, 9e-1) if not ablation else ax.set_ylim(9e-3, 9e-1)

ax.set_xlabel(r"Base learning-rate $\gamma$")
ax.set_ylabel(r"Final bound $\Omega_T$")

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.161,
                    right=0.99
)

fig.savefig(os.path.join(plot_dir, 'lr_sweep.pdf'))

#%% Plot schedule and convergence

fig, axs = plt.subplots(1,2, figsize=FIGSIZE12)

for j, (name, s) in enumerate(schedules.items()):
    _best_lr = rates[name][0]

    print(f"{s.name}, best lr = {_best_lr}")
    label = name

    s.set_base_lr(_best_lr)
    rate = [s.compute_rate(grad_norms=G,
                            D=D,
                            T=t,
                            type=rate_type) for t in time]
    
    ax = axs[0]
    ax.plot(s.schedule,
            c=colors[name],
            lw=2.2
    )

    ax = axs[1]
    ax.plot(np.arange(1, T+1), 
            rate,
            c=colors[name],
            lw=2.2,
            label=label
    )

ax = axs[1]
ax.set_xlim(time[0], )
ax.set_ylim(0.075, 0.32) if not ablation else ax.set_ylim(0.01, 0.32)
ax.set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend(loc='upper right', fontsize=11)

ax = axs[0]
ax.set_ylim(-0.01, 0.09)
ax.set_xlim(time[0], )
ax.set_ylabel(r'Tuned learning rate $\gamma^\star \eta_t$')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)


fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.078,
                    right=0.995,
                    hspace=0.2,
                    wspace=0.2
)

fig.savefig(os.path.join(plot_dir, 'convergence.pdf'))
