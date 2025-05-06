"""
Computes the schedule that minimizes the bound from Theorem 1.
This is done using autodiff in Pytorch and (projected) gradient descent on the bound mapping.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scheduled import WSDSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11, FIGSIZE12

#%% Setup

plot_dir = '../plots/optimize_schedule/'

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%% Problem definition

T = 100
S = WSDSchedule(steps=T+1, final_lr=0.0, cooldown_len=0.0)

D = 1.0
grad_norms = 1.0

"""
Hypothesis: the optimized schedule is linear-decay.

1) Running PEPit for T=100, G=1, D=1, and the linear-decay schedule,
the lower bound value is 0.10003814933044465.
2) Below we compute the bound for linear-decay as baseline. 
"""
pepit_linear_decay = 0.10003814933044465

# tuned linear decay baseline
S0 = WSDSchedule(steps=T+1, final_lr=0.0, cooldown_len=1.0)
linear_decay_lr, linear_decay_rate = compute_optimal_base(S0, G=grad_norms, D=D, T=T)

#%% Pytorch function for bound

def compute_bound(etas: torch.Tensor,
                  grad_norms: torch.Tensor,
                  D: float,
                  T: int
        ):
        _sum_etas = etas[0:T].sum()
        _cumsum_etas = etas[0:T].cumsum(dim=-1)
        _eta_grad = (etas**2 * grad_norms**2)[0:T]
        _sum_eta_grad = _eta_grad.sum()
        _cumsum_eta_grad = _eta_grad.cumsum(dim=-1)

        term1 = (D**2)/(2*_sum_etas)
        term2 = _sum_eta_grad/(2*_sum_etas)

        _alpha = (_sum_etas - _cumsum_etas)[:T-1]
        _betas = _alpha + etas[0:T-1]
        _gammas = (_sum_eta_grad - _cumsum_eta_grad + _eta_grad)[:T-1]
        if torch.any(_gammas < 0):
            _gammas = np.maximum(_gammas, torch.zeros_like(_gammas))

        term3 = (1/2)*((etas[0:T-1]/_alpha)*((_gammas/_betas))).sum()

        return term1+term2+term3

#%% Run optimization

etas = torch.from_numpy(S.schedule)
etas.requires_grad_(True)

opt = torch.optim.SGD([etas],
                      lr=0.1,
                      momentum=0.9,
                      dampening=0.9,
                      nesterov=False
)

n_steps = 800
log_every = 10
res = {'iter': [],
       'loss': [],
       'etas': []
}

for k in range(n_steps):
    opt.zero_grad()
    L = compute_bound(etas, grad_norms=grad_norms, D=D, T=T)
    # Log
    if (k%log_every==0) or (k == n_steps-1):
        res['iter'].append(k)
        res['loss'].append(L.item())
        res['etas'].append(etas.detach().clone().numpy())
    
    L.backward()
    opt.step()
    # project
    with torch.no_grad():
        etas.clamp_(1e-18)

#%% Plot

def plot_single_schedule(i, ax, rescale=True):
    if i >= K:
        return
    cmap = plt.cm.viridis_r
    this_sched = res['etas'][i][:-1]  # last one is not optimized
    
    priority = (i==0) or (i==K-1)
    ax.plot(1+np.arange(T),
            this_sched/(this_sched[0] if rescale else 1.0),
            color=cmap((i/K)**0.8),
            lw=2.5 if priority else 0.7,
            alpha=1.0,
            zorder=5 if priority else 1,
            label="Optimized schedule"
    )
    return
     
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE12)

ax = axs[0]
axins = ax.inset_axes([0.7, 0.7, 0.25, 0.25])  # [x, y, width, height] in relative coordinates
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

for _ax in [ax, axins]:
    _ax.plot(res['iter'], res['loss'], lw=2, zorder=3)

ax.set_xlabel('Optimization step')
ax.set_ylabel(r'Bound $\Omega_T$')
ax.set_ylim(0.08, 1.0)
ax.set_yscale('log')

# 2*PEPit linear-decay
for _ax in [ax, axins]:
    _ax.hlines(y=pepit_linear_decay,
            xmin=0,
            xmax=n_steps,
            ls='--',
            lw=1.5,
            color='grey',
            label='PEPit linear-decay')
    _ax.hlines(y=linear_decay_rate,
            xmin=0,
            xmax=n_steps,
            ls='dashdot',
            lw=1.5,
            color='silver',
            label='Bound for linear-decay')

# Legend
ax.legend(loc="upper left", fontsize=9)

###############################################
# Inset Axis
# axins.set_yscale('log')
axins.set_xticklabels([])
axins.set_yticklabels([])
ret = mark_inset(ax, axins, loc1=1, loc2=2, ec="k", fc="none")
axins.set_xlim(500, 800)
axins.set_ylim(0.18, 0.24)

# zorder does apply only on axis level
# to get the inset markers in background, use this fix
# https://stackoverflow.com/questions/56932448/how-to-control-zorder-and-clipping-with-mpl-toolkits-axes-grid1-inset-locator-ma
for bc in ret[1:]:
    bc.remove()
    ax.add_patch(bc)
    bc.set_zorder(0)

###############################################
# Plot optimized schedule
K = len(res['etas'])
ax = axs[1]
for i in range(K):
    plot_single_schedule(i, ax, rescale=True)
    
ax.set_ylim(0, 1.02)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'Rescaled Schedule $\eta_t/\eta_1$')

fig.subplots_adjust(top=0.971,
bottom=0.159,
left=0.084,
right=0.993,
hspace=0.2,
wspace=0.224)

fig.savefig(os.path.join(plot_dir, f"minimize_bound.pdf"))

# %% Animate

import matplotlib.animation as animation

fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.set_ylim(0, 1.02)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'Rescaled Schedule $\eta_t/\eta_1$')
fig.subplots_adjust(top=0.971,
bottom=0.154,
left=0.137,
right=0.989,)

rescale = True
ani = animation.FuncAnimation(fig=fig,
                              func=plot_single_schedule,
                              fargs=(ax, rescale),
                              frames=int(2*K),
                              interval=75,
                              repeat=False
)

ani.save(filename=os.path.join(plot_dir, f"animated_schedule.gif"), dpi=400, writer="imagemagick")

# %%
