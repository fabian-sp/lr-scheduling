"""
Analysis of transfering optimal base lr from wsd with any cooldown to linear decay.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
import os
import argparse

from scheduled import WSDSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, do_fancy_legend, FIGSIZE11

parser = argparse.ArgumentParser(description='Transfer lr from wsd to linear decay.')
args = parser.parse_args()

#%%

plot_dir = '../plots/lr_transfer/'

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%% Setup and LR sweep

G = 1
D = 1
all_T = [400, 1600, 6400, 25600, 102400]

rates = {'linear': dict([(T, None) for T in all_T]),
         'sqrt': dict([(T, None) for T in all_T])
}

all_cooldown = np.linspace(0, 1, 31)

for decay_type in ['linear', 'sqrt']:
    for T in all_T:
        rates[decay_type][T] = dict([(_cd, None) for _cd in all_cooldown])
        for j, cd in enumerate(all_cooldown):
            
            # replace no cooldown with one-step cooldown
            _cd = 1/(T+1) if cd == 0.0 else cd

            S = WSDSchedule(final_lr=0.0,
                            steps=T+1,
                            cooldown_len=_cd,
                            base_lr=1.0,
                            decay_type=decay_type
            )
            
            best_base_lr, best_rate = compute_optimal_base(S, 
                                                           G=G, 
                                                           D=D, 
                                                           T=T
            )
            
            rates[decay_type][T][cd] = (best_base_lr, best_rate)

#%% Optimal base LR as function of cooldown
""" 
Fit (for each T) 

log(gamma*) = poly(c)

where c is the cooldown fraction
"""

cpal = sns.color_palette("Blues", n_colors=8)[-len(all_T):]

poly_degree = 6
use_cd_zero_for_fit = True
fit_params = {"linear": dict(), "sqrt": dict()}

for decay_type in ['linear', 'sqrt']:
    fig, ax = plt.subplots(figsize=FIGSIZE11)
    labels = list()
    handles = list() 
    for j, T in enumerate(all_T):
    
        x = np.array(list(rates[decay_type][T].keys()))
        y = np.log(np.array([v[0] for v in rates[decay_type][T].values()]))
        y1 = np.log(rates[decay_type][T][1][0]) # ln(gamma*(1))

        if use_cd_zero_for_fit:
            params = np.polyfit(x, y, deg=poly_degree)
        else:
            params = np.polyfit(x[1:], y[1:], deg=poly_degree)
        
        fit_params[decay_type][T] = params
        
        # plot ln(gamma*(1)) - ln(gamma*(c))
        _x = np.linspace(0, 1, 100)
        _y = np.polyval(params, _x)
        ax.plot(_x, 
                y1 - _y,
                c=cpal[j],
                lw=1.5,
                ls='--',
                zorder=1
        )
        
        ax.scatter(x,
                   y1-y,
                   color=cpal[j],
                   s=20,
                   alpha=0.8,
                   zorder=2
        )

    ax.tick_params(direction='in', which='both')
    ax.set_yticks(np.linspace(0,2,21), [], minor=True)

    ax.set_xlim(0,1.02)
    ax.set_ylim(-5e-2,1.75)
    
    ax.set_xlabel(r'Cooldown fraction $c$')
    ax.set_ylabel(r'$\ln{\gamma^\star(1)} - \ln{\gamma^\star(c)}$')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    ax.set_title('linear' if decay_type=='linear' else '1-sqrt')

    do_fancy_legend(ax,
                    labels=[r"$T\in \{ %s \}$" % ','.join([str(T) for T in all_T]),],
                    color_list=[cpal[:len(all_T)],],
                    loc='upper right',
                    fontsize=10,
                    lw=2,
    )

    fig.subplots_adjust(top=0.92,
                        bottom=0.15,
                        left=0.13,
                        right=0.975,
    )
    
    fig.savefig(os.path.join(plot_dir, f'cooldown_fit_gamma_star_{decay_type}.pdf'))
