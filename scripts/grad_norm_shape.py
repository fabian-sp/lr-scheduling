"""
Plot convergence rate and LR sweep for different gradient norm shapes.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
import os
import argparse

from scheduled import CosineSchedule, WSDSchedule
from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11, FIGSIZE12

parser = argparse.ArgumentParser(description='Run gradient norm analysis.')
parser.add_argument('--ablation', action="store_true", help="Ablation for min bound.")
args = parser.parse_args()

#%% Setup

plot_dir = '../plots/grad_norm_shape/'
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
all_alpha = [0.0, -0.5, -1.0]

time = np.arange(1, T+1)

base_lr = np.logspace(-8, 1, num=20, base=2)
rates = {'cosine': dict([(a, None) for a in all_alpha]), 
         'wsd': dict([(a, None) for a in all_alpha])
         }

# massive empty dictionary
rates = dict([(a, dict([(s, dict([(lr, None) 
                                  for lr in base_lr])) 
                            for s in ["cosine", "wsd"]]))
                        for a in all_alpha]
)

best_rate = dict([(a, dict([(s, (None, np.inf))  
                            for s in ["cosine", "wsd"]]))
                        for a in all_alpha]
)


# Schedule objects
cosine = CosineSchedule(final_lr=0.0, steps=T+1, base_lr=1.0)
wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

for a in all_alpha:
    for s in [cosine, wsd]:
        for _lr in base_lr:
            s.set_base_lr(_lr)

            grad_norms = G * np.arange(1, T+2)**a
            rate = [s.compute_rate(grad_norms=grad_norms,
                                   D=D,
                                   T=t,
                                   type=rate_type) for t in time]
        

            rates[a][s.name][_lr] = rate

            # store best_lr, best_rate
            if rate[-1] <= best_rate[a][s.name][1]:
                best_rate[a][s.name] = (_lr, rate[-1])

#%% Plot sweep

reds = sns.color_palette("Reds", len(all_alpha)+2)[2:]
blues = sns.color_palette("Blues", len(all_alpha)+2)[2:]

fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, a in enumerate(all_alpha):
    for s in [cosine, wsd]:
        this_curve = [rates[a][s.name][_lr][-1] for _lr in base_lr]

        # Plot    
        col = reds[j] if s.name == 'cosine' else blues[j]
        label = s.name + rf" $\alpha={a}$"

        ax.plot(base_lr, 
                this_curve,
                c=col,
                alpha=0.5,
        )
        ax.scatter([best_rate[a][s.name][0]], best_rate[a][s.name][1], 
                   facecolors=col, 
                   edgecolors='k', 
                   zorder=5,
                   label=label
        )

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"Base learning-rate $\gamma$")
ax.set_ylabel(r"Final bound $\Omega_T$")

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

labels = [f"wsd  \n" + rf"$\alpha\in [ {','.join([str(_a) for _a in all_alpha])} ]$",
          f"cosine  \n" + rf"$\alpha\in [{','.join([str(_a) for _a in all_alpha])} ]$"
          ]

do_fancy_legend(ax,
                labels=labels,
                color_list=[blues, reds],
                fontsize=10,
                loc='upper right',
                framealpha=0.9
)

fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.161,
                    right=0.99
)

fig.savefig(os.path.join(plot_dir, 'lr_sweep.pdf'))

#%% Plot convergence

fig, axs = plt.subplots(1,2, figsize=FIGSIZE12)

for j, a in enumerate(all_alpha):
    for s in [cosine, wsd]:
        best_base_lr, _ = best_rate[a][s.name]
        
        col = reds[j] if s.name == 'cosine' else blues[j]

        # plot convergence
        axs[1].plot(time,
                    rates[a][s.name][best_base_lr],
                    c=col,
                    lw=2,
                    alpha=0.8,
        )

        # plot G_t
        if s.name == 'wsd':
            grad_norms = G * np.arange(1, T+2)**a
            axs[0].plot(time,
                    grad_norms[:-1],
                    c=col,
                    lw=2,
            )

ax = axs[1]
ax.axvline(x=wsd._cooldown_start_iter, ymin=0, ymax=2, color='lightgrey', ls='--', zorder=0)
ax.set_yscale('log')
ax.set_ylim(2e-3, 2e0)
ax.set_xlim(time[0], )
ax.set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)


ax = axs[0]
ax.set_yscale('log')
ax.set_xlim(time[0], )
ax.set_ylabel(r'Assumed $G_t \propto t^{\alpha}$')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)


labels = [f"wsd  \n" + rf"$\alpha\in [ {','.join([str(_a) for _a in all_alpha])} ]$",
          f"cosine  \n" + rf"$\alpha\in [{','.join([str(_a) for _a in all_alpha])} ]$"
          ]

do_fancy_legend(axs[1],
                labels=labels,
                color_list=[blues, reds],
                loc='upper left',
                fontsize=10,
                # bbox_to_anchor=(0.01, 1.1),
                lw=2.0,
                ncol=1,
                framealpha=0.9
)

fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.083,
                    right=0.995,
                    hspace=0.2,
                    wspace=0.205
)

fig.savefig(os.path.join(plot_dir, 'convergence_multiple_alpha.pdf'))
