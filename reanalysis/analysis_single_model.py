import os
import json
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from data_utils import load_multiple, schedule_mapping, create_inputs

from scheduled.fit import RateFitter

color_dict = {"wsd": (0.19937337, 0.13719028, 0.27040111), 
              "cos": (0.19937337, 0.13719028, 0.27040111)
}
cmap_dict = {"wsd": "rocket",
             "cos": "mako"
}

palettes = dict((sched, sns.color_palette(cmap_dict[sched], 5)) for sched in cmap_dict.keys())


# Specify here what data you want to load
ALL_LR = [0.0001, 0.0005, 0.001, 0.002]

wsd_config_list = [{"iterations": iter, "lr": lr, "scheduler": 'wsd', "decay_type": "linear", "wsd_fract_decay": 0.2} 
                    for iter in [25_000, 50_000] for lr in ALL_LR]

cos_config_list = [{"iterations": iter, "lr": lr, "scheduler": 'cos'} 
                    for iter in [25_000, 50_000] for lr in ALL_LR]

config_list = cos_config_list + wsd_config_list

df, config_df = load_multiple(config_list, data_dir="../data")

#%% Plot validation loss
# %matplotlib qt5

fig, axs = plt.subplots(1, 2, figsize=(8,3))

metric = "val_loss"

rolling_window = (20 if metric == 'train_loss' else 1)

for id in df.id.unique():
    this = df[df.id == id]
    this_lr = config_df.loc[id, "lr"]
    this_sched = config_df.loc[id, "scheduler"]

    col = palettes[this_sched][ALL_LR.index(this_lr)]

    this_no_nan = ~this[metric].isna()

    ax = axs[0]
    ax.plot(this.iter[this_no_nan], 
            this[metric][this_no_nan].rolling(rolling_window).mean(), 
            c=col
    )

    # add final_metrics, which are computed over full set
    ax.scatter([config_df.loc[id, "iterations"]],
               [config_df.loc[id, f"final_{metric}"]],
               color=col,
    )

    ax = axs[1]
    # at cooldown start we have single nan log, should be correct to just use ffill
    ax.plot(this.iter, this.train_lr.ffill(), c=col)
    

axs[0].grid(which='both', lw=0.2, ls='--')
if metric == 'val_perplexity':
    axs[0].set_ylim(15, 30)
if metric == 'val_loss':
    axs[0].set_ylim(2.9, 4)
axs[0].set_ylabel(metric)

axs[1].grid(which='both', lw=0.2, ls='--')
axs[1].set_ylabel("Learning rate")

fig.tight_layout()

#%% Create a fitter object for each id
"""
We create a mapping that specifies for each id a fit object.
"""

fitter = dict()
ids_for_fitting = list(config_df[config_df.iterations == 25_000].index)

for id in config_df.index:
    fitter[id] = RateFitter()

fit_map = dict()
# for now: fit on 25k, use for 50k
for id in config_df.index:
    fit_map[id] = id.replace("iter50000", "iter25000")

print("This is the mapping to the fitting data>")
print(fit_map)

#%% Fit coefficients 

metric = "val_loss"
method = 'least-squares'
# method = 'huber'

cutoff_iter = (None, None)                                                      # use only the logs within these bound: cuts off warmup if None
metric_ub = None                                                                # use knowledge of best ever metric? can be set to None

for id in ids_for_fitting:
    this_id_list = [id] # for now only one id, but could be multiple
    inputs, targets = create_inputs(df,
                                    config_df=config_df, 
                                    ids=this_id_list, 
                                    metric=metric, 
                                    cutoff_iter=cutoff_iter
    )
    id_map = dict([(_id, schedule_mapping(_id, config_df)) for _id in this_id_list])
    
    f = fitter[id]
    kwargs = {'maxfev': 100_000} if method == 'least-squares' else {'use_log': True, 'huber_mu': 10.0}
    f.fit(inputs,
          targets,
          id_map,
          method=method,
          ub=metric_ub,
          **kwargs
    )
    print(f"Fitted coefficients for {id}:", f.params)
    fitter[id] = f

# Copy params for the non-fitted IDs
for id in config_df.index:
    if id not in ids_for_fitting:
        ref_params = tuple(fitter[fit_map[id]].params.values())
        fitter[id].set_params(ref_params)

#%% Predict 

fig, axs = plt.subplots(1,2,figsize=(9,3))
   
for id in config_df.index:
    
    this_sched = config_df.loc[id, "scheduler"]
    this_lr = config_df.loc[id, "lr"]
    this_iter = config_df.loc[id, "iterations"]
    
    if this_iter > 40_000:
        continue

    inputs, true_rate = create_inputs(df,
                                      config_df=config_df, 
                                      ids=[id], 
                                      metric=metric, 
                                      cutoff_iter=(None, None)
    )
    id_map = {id: schedule_mapping(id, config_df)}
    print(id_map)

    print(f"{id}: ", fitter[id]._params)
    predicted_rate = fitter[id].predict(inputs, id_map)
    
    pal = sns.color_palette(cmap_dict[this_sched], 5)
    col = pal[ALL_LR.index(this_lr)]
    
    ax = axs[0] if this_sched == 'cos' else axs[1]
    ax.plot(inputs['t'], predicted_rate, c=col, ls='--')
    ax.plot(inputs['t'], true_rate, c=col, ls='-')

for ax in axs:
    ax.set_ylim(2.9, 4)

# add label for legend: black dashed for predicted, black solid for true, color for lr
lines_cos = [Line2D([0], [0], color='black', lw=1, ls='--'),
         Line2D([0], [0], color='black', lw=1, ls='-')
]
lines_wsd = [Line2D([0], [0], color='black', lw=1, ls='--'),
            Line2D([0], [0], color='black', lw=1, ls='-')
]

for lr in ALL_LR:
    lines_cos.append(Line2D([0], [0], color=palettes['cos'][ALL_LR.index(lr)], lw=1))
    lines_wsd.append(Line2D([0], [0], color=palettes['wsd'][ALL_LR.index(lr)], lw=1))

axs[0].legend(lines_cos, ["predicted", "true"] + [f"lr={lr}" for lr in ALL_LR], loc='upper right')
axs[0].set_ylabel(metric)
axs[1].legend(lines_wsd, ["predicted", "true"] + [f"lr={lr}" for lr in ALL_LR], loc='upper right')
axs[1].set_ylabel(metric)
# legend
fig.tight_layout()


#%% Plot G values over LR

fig, ax = plt.subplots()
for id in ids_for_fitting:
    _lr = config_df.loc[id, "lr"]
    _sched = config_df.loc[id, "scheduler"]

    params = fitter[id].params
    ax.scatter([_lr],[params["G"]], c='blue' if _sched== 'cos' else 'red')
    ax.annotate(f"{params['G']:.2f}", (_lr, params["G"]))

ax.set_xlabel("base lr")
ax.set_ylabel("G")
# add legend with red and blue dots, similar to lines above but with dots
red_dot = Line2D([0], [0], marker='o', color='w', label='Cos',
                  markerfacecolor='red', markersize=10)
blue_dot = Line2D([0], [0], marker='o', color='w', label='WSD',
                  markerfacecolor='blue', markersize=10)
ax.legend(handles=[red_dot, blue_dot])
ax.set_xscale('log', base=2)

#%% Predict final metric over LR
"""
we dont use the final_ logs here, because they are slightly different from the logged values
due to different validation set size
"""

iterations = 25_000

metric = "val_loss"
final_true = {"wsd": dict(), "cos": dict()}
final_pred = {"wsd": dict(), "cos": dict()}

use_final_log = False

for id in config_df.index:
    if config_df.loc[id, "iterations"] != iterations:
        print(id)
        continue
    
    this = df[df.id == id]
    if use_final_log:
        lr, sched, T, y = config_df.loc[id, ["lr", "scheduler", "iterations", "final_"+metric]]  
    else:
        lr, sched = config_df.loc[id, ["lr", "scheduler"]]
        T = this[~this[metric].isna()].iter.max()
        y = this[this.iter == T][metric].values[0]

    # to be sure that Schedule object is not zero here
    eval_t = T-1
    final_true[sched][lr] = y

    inputs = np.array([[eval_t, lr]]).T
    final_pred[sched][lr] = fitter[id].predict(inputs)[0]                       # predict returns array --> [0]


fig, ax = plt.subplots(figsize=(4,3))

for k, v in final_true.items():
    ax.plot(v.keys(),v.values(), 
            c=palettes[k][2],
            label=k+" (true)"
    )

for k, v in final_pred.items():
    ax.plot(v.keys(),v.values(), 
            ls='--',
            c=palettes[k][2],
            label=k+" (predicted)"
    )

ax.set_xlabel("base lr")
ax.set_ylabel(f"final {metric}")
ax.grid(which='both', lw=0.2, ls='--')

ax.set_ylim([2.9, 3.2])
ax.legend()
ax.set_title(f"T={iterations}")
fig.tight_layout()

# %%
