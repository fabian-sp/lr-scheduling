"""
Analysis of transfering schedule to a larger horizon.

We expand by either 

* adapting cooldown for args.schedule in ['wsd', 'sqrt']
* adapting base lr (for second stage) if args.schedule == 'wssd'

The candidates and picked parameters for the transfer are called ``trials`` and ``picked``.

Some remarks:

* for wssd, it can happen that for T1 not too much larger than T0, it is optimal
the LR a lot, as the second stage overlaps largely with the cooldown.
* for sqrt, large cooldown fractions result in similar base_lrs. But here we are
interested in reusing as much as possible, so we exclude large cooldowns.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import argparse
import warnings

from scheduled import WSDSchedule, PiecewiseConstantSchedule, SqrtSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11, FIGSIZE12

parser = argparse.ArgumentParser(description='Transfer schedule to longer horizons with same lr.')
parser.add_argument('--schedule', default='sqrt', help="Which base schedule to analyze.")
parser.add_argument('-d', '--decay_type', type=str, default='linear', help="The type of decay, choose linear (default) or sqrt.")
args = parser.parse_args()

# args = argparse.Namespace(schedule='wssd')

# %%

plot_dir = '../plots/horizon_transfer/'

if args.schedule == 'wsd':
    S = WSDSchedule
    S0 = WSDSchedule
    base_lrs = np.logspace(-8, -1, num=50, base=2)
    trials = np.linspace(0.1, 1, 33)
if args.schedule == 'wssd':
    S = PiecewiseConstantSchedule
    S0 = WSDSchedule
    base_lrs = np.logspace(-10, 0, num=300, base=2)
    trials = np.linspace(0.2, 1, 33)
elif args.schedule == 'sqrt':
    S = SqrtSchedule
    S0 = SqrtSchedule
    base_lrs = np.logspace(-6, 1, num=50, base=2)
    trials = np.linspace(0.1, 0.7, 33)

decay_type = args.decay_type
# decay_type = 'linear'

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

# %% Compute the extended schedule

G = 1
D = 1

T0 = 4000
timepoints = [T0, int(1.5*T0), int(2*T0), int(2.5*T0), int(3*T0), int(3.5*T0), int(4*T0)]
picked = dict()

initial_cd = 0.2

initial_schedule = S0(final_lr=0.0, steps=T0+1, cooldown_len=initial_cd, base_lr=1.0, decay_type=decay_type)
# compute optimal gamma for first horizon
initial_lr, initial_rate = compute_optimal_base(initial_schedule, G=G, D=D, T=T0)
initial_schedule.set_base_lr(initial_lr)

res = dict([(T, dict([(c, dict())
                      for c in trials]))
                      for T in timepoints[1:]
                      ]
)

for j in np.arange(len(timepoints)-1):
    T1 = timepoints[j+1]
    
    for c in trials:
        # define new schedule
        if args.schedule == 'wssd':
            cooldown_kwargs = {"steps": int(initial_cd*(T1+1)),
                               "final_lr": 0.0,
                               "type": decay_type}
            this_schedule = S(
                steps=T1+1,
                base_lr=1.0,
                milestones=[int((1-initial_cd)*T0)],
                factors=[c],
                cooldown_kwargs=cooldown_kwargs
            )
        else:
            this_schedule = S(final_lr=0.0,
                              steps=T1+1,
                              cooldown_len=c,
                              base_lr=1.0,
                              decay_type=decay_type
            )
        # store (optimal_lr, optimal_bound)
        res[T1][c]['optimal'] = compute_optimal_base(this_schedule, G=G, D=D, T=T1)

        rates = list()
        for lr in base_lrs:
            this_schedule.set_base_lr(lr)
            rates.append(this_schedule.compute_rate(grad_norms=G, D=D, T=T1))

        res[T1][c]['rates'] = dict(zip(base_lrs, rates))

    # compute candidate where optimal LR matches initial LR best
    lr_diff = np.abs([res[T1][c]['optimal'][0] - initial_lr for c in trials])/initial_lr
    if not np.any(lr_diff <= 1e-2):
        warnings.warn(
            f"Have not found an accurate candidate for T={T1}."+
            f"Best candidate matches LR realtively up to {lr_diff.min()}."
        )
    c_ix = np.argmin(lr_diff)

    picked[T1] = trials[c_ix]

print("Picked transfer factors:", picked)

# %% Plot the extended schedule with rate
# Baselines: linear-decay schedule, extended with same cooldown and gamma

fig, axs = plt.subplots(1,2, figsize=FIGSIZE12)
greens = sns.color_palette("Greens", 8)[-len(timepoints):]
blues = sns.color_palette("Blues", 8)[-len(timepoints):]
extended_schedules = dict()

for j, T in enumerate(timepoints):
    time = np.arange(1, T+1)

    if j == len(timepoints)-1:
        label1 = 'adapted schedule' if args.schedule=='wssd' else 'adapted cooldown'
        label2 = 'same base lr+cooldown'
        label_linear = 'linear-decay'
    else:
        label1 = label2 = label_linear = None
    
    ############### Linear Schedule ################
    # construct linear decay for this T
    this_linear = WSDSchedule(steps=T+1, cooldown_len=1)
    linear_lr, _ = compute_optimal_base(this_linear, G=G, D=D, T=T)
    this_linear.set_base_lr(linear_lr)
    linear_rate = [this_linear.compute_rate(grad_norms=G, D=D, T=t)
                   for t in time]

    # axs[0].plot(np.arange(1, T+2),
    #             this_linear.schedule,
    #             c=blues[j],
    #             alpha=0.7
    # )
    # axs[1].plot(time,
    #             linear_rate,
    #             c=blues[j],
    #             alpha=0.7
    # )
    _T0 = 0 if j==0 else timepoints[j-1]
    axs[1].hlines(linear_rate[-1], _T0, T, ls='--', color=blues[j], label=label_linear)

    ############### Naive extension ################
    # use same gamma and cooldown fraction as in inital schedule
    _s2 = S0(final_lr=0.0,
             steps=T+1,
             cooldown_len=initial_cd,
             base_lr=initial_lr,
             decay_type=decay_type
    )
    this_rate2 = [_s2.compute_rate(grad_norms=G, D=D, T=t)
                  for t in time]
    axs[1].plot(time,
                this_rate2,
                c='grey',
                alpha=0.7,
                zorder=1,
                label=label2
    )
    axs[0].plot(np.arange(1, T+2),
                _s2.schedule,
                c='grey',
                alpha=0.7,
                zorder=1
    )

    ############### Extended Schedule ################
    # compute optimal gamma for first horizon
    if j == 0:
        _s = initial_schedule
    else:
        this_c = picked[T]
        if args.schedule == "wssd":
            cooldown_kwargs = {"steps": int(initial_cd*(T+1)),
                               "final_lr": 0.0,
                               "type": decay_type
            }
            _s = S(
                steps=T+1,
                base_lr=initial_lr,
                milestones=[int((1-initial_cd)*T0)],
                factors=[this_c],
                cooldown_kwargs=cooldown_kwargs
            )
        else:
            _s = S(
                final_lr=0.0,
                steps=T+1,
                cooldown_len=this_c,
                base_lr=initial_lr,
                decay_type=decay_type,
            )

    axs[0].plot(np.arange(1, T+2),
                _s.schedule,
                c=greens[j],
                lw=2.2,
                zorder=2
    )

    this_rate = [_s.compute_rate(grad_norms=G, D=D, T=t)
                 for t in time]

    axs[1].plot(time,
                this_rate,
                c=greens[j],
                lw=2.2,
                zorder=2,
                label=label1
    )

axs[0].set_ylim(0, 0.02)
axs[0].set_xlabel(r'Iteration $t$')
axs[0].set_ylabel(r'$\gamma^\star \eta_t$')

axs[1].set_ylim(0.01 * np.sqrt(4000/T0), 0.1)
axs[1].set_xlabel(r'Iteration $t$')
axs[1].set_ylabel(r'$\Omega_t$')
axs[1].legend(loc='upper right')


for ax in axs:
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

fig.subplots_adjust(top=0.97,
                    bottom=0.15,
                    left=0.075,
                    right=0.995,)

fig.savefig(os.path.join(plot_dir, f'extended_{args.schedule}_{decay_type}.pdf'), bbox_inches='tight')

# %% Plot
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=FIGSIZE11)
cmap = sns.color_palette("flare", as_cmap=True)

T0 = timepoints[0]
T1 = timepoints[2]

ax.hlines(y=np.log2(initial_lr), xmin=0, xmax=1, ls='--', color='silver', zorder=1)

for c in trials:
    _lrs, _rates = zip(*res[T1][c]['rates'].items())
    _r = np.array(list(_rates))
    x = c + 0.1*(_r - _r.min())
    y = np.log2(np.array(_lrs))

    col = cmap(c)
    ax.plot(x, y, c=col, alpha=0.3)
    ax.scatter(c, np.log2(res[T1][c]['optimal'][0]), color=col)

if args.schedule == 'wssd':
    ax.set_xlabel(r'Decreasing factor $\rho$ for $T=%d$' % T1)
else:
    ax.set_xlabel(r'Cooldown fraction $c$ for $T=%d$' % T1)
ax.set_ylabel(r'$\log_2(\gamma$), for $T=%d$' % T1)
ax.set_xlim(0.2, 0.75)
ax.set_ylim(np.log2((2**-2)*initial_lr), np.log2((2**2)*initial_lr))

handles = [Line2D([], [], ls='--', c='silver', lw=2),
           Line2D([], [], c=cmap(0.5), alpha=0.3, lw=2),
           Line2D([], [], marker='o', lw=0.0, c=cmap(0.5)),
]
labels = [r'$\log_2(\gamma^\star$), for $T=%d$ and $c=%s$' % (T0, initial_cd),
          r'$\Omega_{%d}(\gamma)$' % T1,
          r'$\gamma^\star$ for $T=%d$' % T1,
]

ax.legend(handles, labels, loc='upper right', framealpha=0.95)

fig.subplots_adjust(top=0.985,
                    bottom=0.155,
                    left=0.135,
                    right=0.99
)

# make sure axes and legends are in figure too
fig.savefig(os.path.join(plot_dir, f'transfer_cooldown_{args.schedule}_{decay_type}.pdf'), bbox_inches='tight')


# %% Plot learning rate ratios vs horizon length
if args.schedule == 'wssd':
    fig, ax = plt.subplots(figsize=FIGSIZE11)

    x = np.array(timepoints) / T0
    y = np.array([1] + list(picked.values()))
    # Plot line
    ax.plot(x,
            y,
            '-o',
            color=(0.10557477893118032, 0.41262591311034214, 0.6859669357939254),
            lw=2)

    ax.set_xlabel(r'$T/T_1$')
    ax.set_ylabel(r'Decreasing factor $\rho$')
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    # Adjust layout
    fig.subplots_adjust(top=0.97,
                        bottom=0.15,
                        left=0.15, 
                        right=0.95)

    # Save plot
    fig.savefig(os.path.join(plot_dir, f'lr_ratios_{args.schedule}_{decay_type}.pdf'), bbox_inches='tight')

# %% for testing:
# dependence of gamma* on T for SqrtSchedule

# all_T = [1e2, 1e3, 1e4, 1e5]

# for T in all_T:
#     this_schedule = SqrtSchedule(final_lr=0.0,
#                                  steps=T+1,
#                                  cooldown_len=0.2,
#     )
#     gamma_star, _ = compute_optimal_base(this_schedule, G=G, D=D, T=T)
#     print("(T, lr):", T, gamma_star)
