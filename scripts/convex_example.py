import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from scheduled import WSDSchedule, ConstantSchedule, CosineSchedule
from scheduled.utils import set_plot_aesthetics, FIGSIZE11

#%%

plot_dir = '../plots/convex_example/'
save = True

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%% Setup

torch.random.manual_seed(0)

d = 2
n = 20
eps = 0
norm_sol = 5.0

X = 2*torch.rand(n, d) - 1
w0 = torch.randn(1, d)
w0 = w0.mul(norm_sol/torch.linalg.norm(w0))
y = X @ w0.T + eps*torch.randn(n, 1)


def loss_fn(out, targets):
    return torch.max(torch.abs(out-targets))
    
out_star = X @ w0.T 
ub_fstar = loss_fn(out_star, y)

#%% Train

T = 400
lr = 0.02
beta = 0        # momentum
eval_freq = 10

# double LR for cosine
schedules = [WSDSchedule(final_lr=0, steps=T, cooldown_len=0.2, base_lr=lr),
             ConstantSchedule(steps=T, base_lr=lr),
             CosineSchedule(final_lr=0, steps=T, base_lr=2*lr)
]

res = dict((S.name, {'iter': list(),
                     'obj': list(),
                     'grad_norm': list(),
                     'dist': list(),
                     'iterate': list() # careful: dont make d too large
            }) for S in schedules)

for S in schedules:
    w = torch.zeros(1, d, requires_grad=True)
    
    if S.name == 'constant':
        w = torch.ones(1, d)*0.001  # to make it visually distinguishable
        w.requires_grad = True

    m = torch.zeros_like(w, requires_grad=False)

    for t in range(1, T+1):
        eta = S.schedule[t-1]
        out = X @ w.T

        loss = loss_fn(out, y)
        loss.backward()

        # logging (before step)
        if (eval_freq == 1) or (t % eval_freq == 1) or (t == T-1):
            res[S.name]['iter'].append(t)
            res[S.name]['obj'].append(loss.item())
            res[S.name]['grad_norm'].append(torch.linalg.norm(w.grad.data).item())
            res[S.name]['dist'].append(torch.linalg.norm(w.data - w0).item())
            res[S.name]['iterate'].append(w.clone().data.numpy().squeeze())

        m.mul_(beta).add_(other=w.grad.data, alpha=1-beta)
        w.data.add_(other=m, alpha=-eta)
        w.grad.zero_()
        

#%% Plot convergence
ma_window = 5

colors = {'wsd': "#7587A0",
          'constant': "#CFB294",
          'cosine': "#6B5756"}

ylabels = {'obj': rf'$f(x_t)$',
           'grad_norm': rf'$\|g_t\|$',
           'dist': rf'$\|x_t-x^\star\|$'}

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_metric(ax, metric='obj', ylim=None, log_scale=False):
    for j, S in enumerate(schedules):
        ax.plot(res[S.name]['iter'],
                res[S.name][metric],
                lw=1.0,
                c=colors[S.name],
                marker='o',
                markersize=6,
                markevery=(4, len(res[S.name][metric])//10),
                alpha=0.99,
                label=S.name,
                zorder=2+j
        )

    ax.set_xlim(ma_window, T+1)
    
    if log_scale:
        ax.set_yscale('log')

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel(ylabels[metric])
    #ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    ax.tick_params(direction='in')
    return 

fig, ax = plt.subplots(figsize=FIGSIZE11)
plot_metric(ax,
            metric='obj',
            ylim=(-1e-2, 0.1)
)
ax.axvline(x=schedules[0]._cooldown_start_iter, ymin=0, ymax=2, color='lightgrey', ls='--', zorder=1)
ax.legend(loc='upper right', ncol=1, framealpha=0.95)
ax.set_xlim(1, T+2)
fig.subplots_adjust(top=0.955,
                    bottom=0.16,
                    left=0.155,
                    right=0.97)
if save:
    fig.savefig(os.path.join(plot_dir, 'obj.pdf'))

#%% Plot contour and path (if d=2)
from matplotlib.collections import LineCollection

xlims = (1.21, 1.25)
ylims = (4.75, 4.9)

X1, X2 = np.meshgrid(np.linspace(xlims[0], xlims[1], 100),
            np.linspace(ylims[0], ylims[1], 100))

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
         out = X @ torch.tensor([X1[i,j], X2[i,j]]).to(torch.float).reshape(d,1)
         Z[i,j] = loss_fn(out, y).item()


fig, ax = plt.subplots(figsize=FIGSIZE11)

for j, S in enumerate(schedules):
    if S.name == 'wsd':
        ixx = np.array([t >= S._cooldown_start_iter for t in res['wsd']['iter']])
        path1 = np.stack(res[S.name]['iterate'])[~ixx, :]
        path2 = np.stack(res[S.name]['iterate'])[ixx, :]
        ls = ['-', ':']
        col = [path1, path2]

    else:
        ls = '-'
        path = np.stack(res[S.name]['iterate'])
        col = [path]

    line_segments = LineCollection(col,
                                   linewidths=2.2,
                                   colors=colors[S.name],
                                   linestyle=ls,
                                   capstyle='round',
                                   joinstyle='round',
                                   alpha=0.99,
                                   
    )
    ax.add_collection(line_segments)

    # plot final point
    ax.scatter([res[S.name]['iterate'][-1][0]], 
               [res[S.name]['iterate'][-1][1]],
               c=colors[S.name],
               s=40,
               edgecolors='k',
               zorder=5
    )

# plot xstar
ax.scatter([w0[0,0]], [w0[0,1]], marker='*', s=120, edgecolors='k', c='gold', zorder=10)

# plot contour
ax.contourf(X1, X2, Z,
           cmap=plt.cm.Greys,
           levels=50,
           alpha=0.8)

ax.set_xticks([1.22, 1.23, 1.24, 1.25])
ax.set_yticks([4.75, 4.8, 4.85, 4.9])

ax.set_xlim(*xlims)
ax.set_ylim(*ylims)

fig.subplots_adjust(top=0.955,
                    bottom=0.110,
                    left=0.145,
                    right=0.96)

if save:
    fig.savefig(os.path.join(plot_dir, 'path.pdf'))