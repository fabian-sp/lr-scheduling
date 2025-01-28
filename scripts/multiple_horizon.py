"""
Plot convergence rate and LR sweep.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import argparse

from scheduled import CosineSchedule, WSDSchedule, compute_optimal_base
from scheduled.utils import do_fancy_legend, set_plot_aesthetics, FIGSIZE11, FIGSIZE12


parser = argparse.ArgumentParser(description='Run multiple horizon analysis.')
parser.add_argument('--ablation', action="store_true", help="Ablation for min bound.")
args = parser.parse_args()

#%% Setup

plot_dir = '../plots/multiple_horizon/'
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

G = 1
D = 1
all_T = [200, 300, 400, 500, 600]
all_D = dict(zip(all_T, np.ones(len(all_T)) * D))

base_lr = np.logspace(-8, -1, num=20, base=2)
rates = {'cosine': dict([(T, None) for T in all_T]), 
         'wsd': dict([(T, None) for T in all_T])
         }

for T in all_T:
    cosine = CosineSchedule(final_lr=0.0, steps=T+1, base_lr=1.0)
    wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

    for s in [cosine, wsd]:
        best_base_lr, best_rate = compute_optimal_base(s, 
                                                       G=G, 
                                                       D=all_D[T], 
                                                       T=T,
                                                       type=rate_type
        )
        rates[s.name][T] = (best_base_lr, best_rate)

#%% Plot LR sweep

reds = sns.color_palette("Reds", len(all_T)+2)[2:]
blues = sns.color_palette("Blues", len(all_T)+2)[2:]
fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, T in enumerate(all_T):
    cosine = CosineSchedule(final_lr=0.0, steps=T+1, base_lr=1.0)
    wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

    for s in [cosine, wsd]:
        this_curve = list()
        for _lr in base_lr:
            s.set_base_lr(_lr)
            _rate = s.compute_rate(grad_norms=G,
                                   D=all_D[T],
                                   T=T,
                                   type=rate_type
            )
            this_curve.append(_rate)

        # Plot    
        col = reds[j] if s.name == 'cosine' else blues[j]
        label = s.name + rf" $T={T}$"

        ax.plot(base_lr, 
                this_curve,
                c=col,
                alpha=0.5,
        )
        ax.scatter([rates[s.name][T][0]], rates[s.name][T][1], 
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

labels = [f"wsd  \n" + rf"$T\in [ {','.join([str(_T) for _T in all_T])} ]$",
          f"cosine  \n" + rf"$T\in [{','.join([str(_T) for _T in all_T])} ]$"
          ]

do_fancy_legend(ax,
                labels=labels,
                color_list=[blues, reds],
                fontsize=10,
                loc='upper right'
)

fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.161,
                    right=0.99
)

fig.savefig(os.path.join(plot_dir, 'lr_sweep.pdf'))

#%% Plot entire convergence path for best LR

# infer optimal LR
best_lr = {'cosine': dict(),
           'wsd': dict()
        }

fig, axs = plt.subplots(1,2, figsize=FIGSIZE12)

for j, T in enumerate(all_T):
    cosine = CosineSchedule(final_lr=0.0, steps=T+1, base_lr=1.0)
    wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

    time = np.arange(1, T+1)
    for s in [cosine, wsd]:
        _best_lr = rates[s.name][T][0]

        print(f"{s.name}, T={T}, best lr = {_best_lr}")

        col = reds[j] if s.name == 'cosine' else blues[j]
        label = s.name + f", T={T}"
    
        s.set_base_lr(_best_lr)
        rate = [s.compute_rate(grad_norms=G,
                                D=all_D[T],
                                T=t,
                                type=rate_type) for t in time]
        
        ax = axs[0]
        ax.plot(np.arange(1, T+2),
                s.schedule,
                c=col,
                lw=2,
        )

        ax = axs[1]
        ax.plot(time, 
                rate,
                c=col,
                lw=2,
                alpha=0.8,
                label=label
        )

ax = axs[1]
ax.set_xlim(time[0], )
ax.set_ylim(0.02, 0.42)
ax.set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

ax = axs[0]
ax.set_ylim(-0.01, )
ax.set_xlim(time[0], )
ax.set_ylabel(r'Tuned learning rate $\gamma^\star \eta_t$')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)


labels = [f"wsd  \n" + rf"$T\in [ {','.join([str(_T) for _T in all_T])} ]$",
          f"cosine  \n" + rf"$T\in [{','.join([str(_T) for _T in all_T])} ]$"
          ]

do_fancy_legend(ax,
                labels=labels,
                color_list=[blues, reds],
                loc='upper right',
                fontsize=10,
                # bbox_to_anchor=(0.01, 1.1),
                ncol=1
)


fig.subplots_adjust(top=0.985,
                    bottom=0.164,
                    left=0.078,
                    right=0.995,
                    hspace=0.2,
                    wspace=0.175
)

fig.savefig(os.path.join(plot_dir, 'convergence_multiple_horizon.pdf'))

#%% optimal base LR vs horizon
"""
fit function of form

gamma* = A * (T**B)
"""
from scipy.optimize import curve_fit

fig, ax = plt.subplots(figsize=FIGSIZE11)

for sched in ['cosine', 'wsd']:
    vT = np.array(list(rates[sched].keys()))
    vG = np.array([v[0] for v in rates[sched].values()])

    c = reds[-2] if sched =='cosine' else blues[-2]

    fun = lambda T, A, B: A*(T**B)
    res = curve_fit(f=fun,
                    xdata=vT,
                    ydata=vG,
                    full_output=True,
                    bounds = ([0,-np.inf],
                              [np.inf, np.inf])
        )
    fit_params = res[0]
    _x = np.linspace(0.8*vT.min(), 1.2*vT.max(), 100)
    label = sched + r", $\gamma^\star = %.2f \cdot T^{%.3f}$" % (fit_params[0], fit_params[1])
    ax.plot(_x,
            fun(_x, fit_params[0], fit_params[1]),
            ls= '--',
            lw=2,
            c=c,
            label=label
            )

    # plot data
    ax.plot(vT,
            vG,
            lw=0,
            marker='o',
            c=c
    )

ax.set_xlabel(r'Training horizon $T$')
ax.set_ylabel(r'Optimal base learning-rate $\gamma^\star$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend(loc='upper right')

fig.subplots_adjust(top=0.985,
                bottom=0.155,
                left=0.155,
                right=0.99
)

fig.savefig(os.path.join(plot_dir, 'fit_gamma_star.pdf'))

#%% create single plot (for Figure 1)

if not ablation:
    D = 1
    G = 1
    # few_T = [2_000, 3_000, 4_000]
    few_T = [22_000, 33_000, 44_000]
    # base_lrs = {'cosine': 0.015, 'wsd': 0.0075}
    base_lrs = {'cosine': 0.005, 'wsd': 0.0025}

    few_reds = sns.color_palette("Reds", len(few_T)+2)[2:]
    few_blues = sns.color_palette("Blues", len(few_T)+2)[2:]

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE11)

    for j, T in enumerate(few_T):
        cosine = CosineSchedule(final_lr=0.0, steps=T+1, base_lr=1.0)
        wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

        time = np.arange(1, T)
        for s in [cosine, wsd]:
            best_base_lr, _ = compute_optimal_base(s,
                                                G=G,
                                                D=D,
                                                T=T,
            )
            print(s.name, T, best_base_lr)

            col = few_reds[j] if s.name == 'cosine' else few_blues[j]
            label = s.name + f", T={T}"
        
            # s.set_base_lr(best_base_lr)
            s.set_base_lr(base_lrs[s.name])

            rate = [s.compute_rate(grad_norms=G,
                                    D=D,
                                    T=t,) for t in time]
            
            ax.plot(time, 
                    rate,
                    c=col,
                    lw=2,
                    alpha=0.9,
                    label=label
            )

    ax.set_xlim(time[0], )
    # ax.set_ylim(0.02, 0.22)
    ax.set_ylim(0., 0.1)
    ax.set_ylabel(r'Suboptimality bound')
    ax.set_xlabel(r'Iteration $t$')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

    # {','.join([str(_T) for _T in few_T])}
    labels = [r"$\tt cosine$" +  "\n" + rf"$T\in [22k, 33k, 44k]$",
              r"$\tt wsd$"  + "\n" + rf"$T\in [22k, 33k, 44k]$",  
            ]

    do_fancy_legend(ax,
                    labels=labels,
                    color_list=[few_reds, few_blues],
                    loc='upper right',
                    fontsize=10,
                    # bbox_to_anchor=(0.01, 1.1),
                    ncol=1
    )


    fig.subplots_adjust(top=0.980,
                        bottom=0.164,
                        left=0.158,
                        right=0.995,
                        hspace=0.2,
                        wspace=0.175
    )

    fig.savefig(os.path.join(plot_dir, 'only_convergence.pdf'))