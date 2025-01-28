"""
Detailled look at WSD bound.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import argparse

from scheduled import WSDSchedule, ConstantSchedule, compute_optimal_base
from scheduled.utils import do_fancy_legend, set_plot_aesthetics, FIGSIZE11, FIGSIZE12, harmonic_number

parser = argparse.ArgumentParser(description='Run multiple horizon analysis.')
args = parser.parse_args()

#%% Setup

plot_dir = '../plots/wsd_detail/'

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
all_T = [500, 1000, 2000, 4000, 8000]

rates = {'constant': dict([(T, None) for T in all_T]), 
         'wsd': dict([(T, None) for T in all_T])
         }

for T in all_T:
    constant = ConstantSchedule(steps=T+1, base_lr=1.0)
    wsd = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)

    for s in [constant, wsd]:
        best_base_lr, best_rate = compute_optimal_base(s, 
                                                       G=G, 
                                                       D=D, 
                                                       T=T
        )
        rates[s.name][T] = (best_base_lr, best_rate)

#%% fit 1/sqrt(T)
from scipy.optimize import curve_fit

fun = lambda T, A: A/np.sqrt(T)

res = curve_fit(f=fun,
                xdata=np.array(all_T),
                ydata=np.array([rates['wsd'][T][1] for T in all_T]),
                full_output=True,
)
A = res[0][0]

#%% Plot WSD bound vs constant and 1/sqrt(T)

blues = sns.color_palette("Blues", len(all_T)+2)[2:]
fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, T in enumerate(all_T):
    S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)
    
    time = np.arange(1, T+1)
    best_base_lr = rates[S.name][T][0]
    S.set_base_lr(best_base_lr)

    rate = [S.compute_rate(grad_norms=G,
                            D=D,
                            T=t) for t in time]

    # Plot rate of WSD
    col = blues[j]
    label = 'wsd' if j == len(all_T)-1 else None
    ax.plot(time,
            rate,
            c=col,
            lw=1.5,
            zorder=1,
            alpha=0.7,
            label=label
    )

_t = np.arange(10, max(all_T)+1)

# Plot constant bound
H = np.array([harmonic_number(t-1) for t in _t])
ax.plot(_t,
        D*G/np.sqrt(_t) * np.sqrt(1+H),
        ls='-',
        lw=2.4,
        c='grey',
        zorder=2,
        label=r'$\frac{DG}{\sqrt{T}}\sqrt{1+H_{T-1}}$'
)

# Plot O(1/sqrt(T))
ax.plot(_t,
        A / np.sqrt(_t),
        ls='-',
        lw=2.4,
        c='silver',
        zorder=3,
        label=r'$\mathcal{O}(1/\sqrt{T})$'
)

ax.set_xlim(1, )
ax.set_ylim(0, 0.1)
ax.set_ylabel(r'Bound $\Omega_t$ (for $\gamma^\star$)')
ax.set_xlabel(r'Iteration $t$')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend()

fig.subplots_adjust(top=0.975,
bottom=0.15,
left=0.16,
right=0.99,)

fig.savefig(os.path.join(plot_dir, 'no_log_terms.pdf'))

#%% Plot each term of WSD bound
# colors from: coolors.co/fffcf2-ccc5b9-403d39-252422-eb5e28

fig, ax = plt.subplots(figsize=FIGSIZE11)

T = 4000
S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)


time = np.arange(1, T+1)
best_base_lr = rates[S.name][T][0]
S.set_base_lr(best_base_lr)

rate = [S.compute_rate(grad_norms=G,
                       D=D,
                       T=t,
                       return_split=True) for t in time]

term1 = np.array([r[0] for r in rate])
term2 = np.array([r[1] for r in rate]) + np.array([r[2] for r in rate])

ax.plot(time, term1, c="#252422", label=r"$\mathcal{T}_1 / \gamma$")
ax.plot(time, term2, c="#eb5e28", label=r"$\gamma\mathcal{T}_2$")

ax.set_xlabel(r'Iteration $t$')
ax.set_ylim(0, 0.2)
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend()

fig.subplots_adjust(top=0.97,
bottom=0.15,
left=0.11,
right=0.995,)

fig.savefig(os.path.join(plot_dir, 'single_terms.pdf'))

#%% Sensitivity constant part vs cooldown part (see Olmo2 paper)

# fig, axs = plt.subplots(1,2, figsize=FIGSIZE12)

# T = 4000
# assert T in all_T

# gamma_star = rates['wsd'][T][0]
# fractions = np.array([0.7, 0.9, 1.1, 1.3])
# few_base_lr = fractions * gamma_star

# S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2)
# T0 = S._cooldown_start_iter

# K = int(0.02*T)
# time = np.arange(1, T+1)

# for lr in few_base_lr:
#     S.set_base_lr(lr)
#     rate = [S.compute_rate(grad_norms=G,
#                             D=D,
#                             T=t) for t in time]
    
#     axs[0].plot(np.arange(1, T0+1),
#                 rate[:T0]
#     )
#     axs[1].plot(np.arange(T0-K, T+1),
#                 rate[T0-(K+1):],
#                 label='$\gamma=%.4f $'%lr
#     )

# axs[0].set_ylim(0.03, 0.2)
# axs[1].set_ylim(0.03, 0.075)
# axs[1].legend()
# axs[0].set_ylabel(r'Bound $\Omega_t$')

# for ax in axs:
#     ax.set_xlabel(r'Iteration $t$')
#     ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    
# fig.subplots_adjust(top=0.96,
# bottom=0.15,
# left=0.09,
# right=0.995,)

# fig.savefig(os.path.join(plot_dir, 'reproduce_olmo.pdf'))