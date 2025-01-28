"""
Running this script requires the following packages to be installed (which are not part of requirements)
    * PEPit
    * (optional): mosek (requires license)


If MOSEK is not installed, simply set wrapper="cvxpy".

"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothConvexFunction

from scheduled import WSDSchedule, CosineSchedule, ConstantSchedule, SqrtSchedule, compute_optimal_base
from scheduled.utils import set_plot_aesthetics, FIGSIZE11


parser = argparse.ArgumentParser(description='Run lower bound analysis with PEPit.')
parser.add_argument('-s', '--schedule', nargs='?', type=str, default='wsd', help="Ablation for min bound.")
args = parser.parse_args()

schedule = args.schedule
# schedule = 'wsd'

#%%

plot_dir = '../plots/lower_bound/'
save = True

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# %matplotlib qt5
set_plot_aesthetics()

#%%
"""
We want to compute a lower bound for suboptimality for x_T generated from

for t=1,..,T-1
    x_t+1 = x_t - eta_t g_t

Thus, set performance metric to final function gap after T-1 steps. (NOTE: loop ends at T-1)
Border case: T=2 --> eval x_2
"""

def wc_subgradient_method(T, step_sizes, grad_norms, D, smooth=False, wrapper="mosek", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

  
    # Declare a convex lipschitz function
    if smooth:
        func = problem.declare_function(SmoothConvexFunction, L=G)
    else:
        func = problem.declare_function(ConvexFunction)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    xs.set_name("xstar")
    fs = func(xs)


    # Then define the starting point x1 of the algorithm
    x1 = problem.set_initial_point()
    x1.set_name(1)

    # Set the initial constraint that is the distance between x1 and xs
    constraint = ((x1 - xs) **2 == D**2)
    problem.set_initial_condition(constraint)

    # Run T-1 steps of the subgradient method (x1,...,xT)
    # evaluate at last iterate xT
    x = x1
    fun_vals = list()      # function values, will be evald in the end
    
    # to eval at start point, dont loop
    if T > 1:
        for t in range(1, T):
            gt, ft = func.oracle(x)
            fun_vals.append(ft)
            x = x - step_sizes[t] * gt
            x.set_name(t+1)
            func.add_constraint((gt**2 <= grad_norms[t-1]))
            
    # Set the performance metric to the final function value accuracy
    # now x = x_T
    gt, ft = func.oracle(x)
    func.add_constraint((gt**2 <= grad_norms[-1]))  
    fun_vals.append(ft)
    problem.set_performance_metric(ft - fs)

    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)

    pepit_tau = problem.solve(wrapper=wrapper,
                              solver=solver,
                              verbose=pepit_verbose,
                              #dimension_reduction_heuristic="logdet50"
    )

    print("Number of function constraints: ", len(func.list_of_class_constraints))
    print("Number of function points: ", len(func.list_of_points))

    # eval functions
    assert len(fun_vals) == T, f"Something seems to be wrong, we stored {len(fun_vals)} function evaluations."
    fun_vals = np.array([f.eval() for f in fun_vals])
    opt_val = fs.eval()

    return pepit_tau, fun_vals, opt_val

#%% compute lb and ub over t

T = 60
D = 1
G = 1
grad_norms = np.ones(T+1) * G
time = np.arange(1, T+1)

if schedule == 'wsd':
    S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=0.2, base_lr=1.0)
elif schedule == 'cosine':
    S = CosineSchedule(final_lr=0, steps=T+1, base_lr=1.0)
elif schedule == 'constant':
    S = ConstantSchedule(steps=T+1, base_lr=1.0)
elif schedule == 'sqrt':
    S = SqrtSchedule(steps=T+1, base_lr=1.0)
elif schedule == 'linear':
    S = WSDSchedule(final_lr=0.0, steps=T+1, cooldown_len=1.0, base_lr=1.0)

best_base_lr, best_rate = compute_optimal_base(S, G=G, D=D, T=T, type='refined')
S.set_base_lr(best_base_lr)

pepit_freq = 5
LB = dict()
UB = dict()

for t in time:

    t = int(t)  # avoid issues when dumping as json
    if (t % pepit_freq == 1) or (t==T):
        # lower bound with PepIT
        lb, fs, fstar = wc_subgradient_method(T=t, step_sizes=S.schedule, grad_norms=grad_norms, D=D, smooth=False, wrapper="mosek", solver=None, verbose=1)
        LB[t] = lb

    # upper bound with theoretical rate
    ub = S.compute_rate(grad_norms=G,
                        D=D,
                        T=t
    )
    UB[t] = ub

if save:
    with open(f"output/pepit_{schedule}.json", "w") as f: 
        json.dump(LB, f, indent=4, sort_keys=True)

#%% Plot

colors = ["#414455", "#7587A0"]
LW = 2.5

fig, ax = plt.subplots(figsize=FIGSIZE11)

upsampled_lb = np.interp(x=list(UB.keys()), xp=list(LB.keys()), fp=list(LB.values()))

if S.name == 'wsd':
    ax.axvline(x=S._cooldown_start_iter, ymin=0, ymax=2, color='lightgrey', ls='--', zorder=1)

ax.plot(UB.keys(), UB.values(), c=colors[1], lw=LW, label=r"$\Omega_t$ (upper bound)", zorder=5)
ax.plot(LB.keys(), LB.values(), c=colors[0], lw=LW, label=r"PEP lower bound", zorder=5)
ax.fill_between(np.arange(1,T+1),
                upsampled_lb,
                list(UB.values()),
                color='lightgrey',
                alpha=0.4,
                zorder=2
                )

ax.set_xlim(time[0], )
ax.set_ylim(1e-1, 2)
ax.set_yscale('log')

ax.set_xlabel(r'Iteration $t$')
ax.set_ylabel(r'')
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend(loc='upper right')

fig.subplots_adjust(top=0.98, bottom=0.16, left=0.11, right=0.99)

if save:
    fig.savefig(os.path.join(plot_dir, f'{schedule}.pdf'))
