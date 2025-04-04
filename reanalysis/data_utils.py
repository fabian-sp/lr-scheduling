import os
import json
import pandas as pd
import numpy as np
import warnings

from scheduled import WSDSchedule, CosineSchedule


def schedule_mapping(id, config_df, use_warmup=True):
    """maps id to a Schedule object"""
    
    if use_warmup:
        warmup_kwargs = {"steps": config_df.loc[id, "warmup_steps"],
                         "warmup_lr": 1e-2, #FIXME: what was really used?
                         "warmup_lr_absolute": False
                        }
    else:
        warmup_kwargs = None

    if config_df.loc[id, "scheduler"] == "wsd":
        T = int(config_df.loc[id, "iterations"])
        decay_type = config_df.loc[id, "decay_type"]
        cooldown_len = float(config_df.loc[id, "wsd_fract_decay"])

        # base_lr gets set later in the fit._vec_eval
        S = WSDSchedule(final_lr=0.0,
                        steps=T,
                        cooldown_len=cooldown_len,
                        decay_type=decay_type,
                        base_lr=1.0,
                        warmup_kwargs=warmup_kwargs
        )

    elif config_df.loc[id, "scheduler"] == "cos":
        final_lr = float(config_df.loc[id, "cos_final_lr"])
        final_lr_absolute = True if final_lr > 0 else False
        T = int(config_df.loc[id, "iterations"])

        S = CosineSchedule(final_lr=final_lr,
                           steps=T,
                           base_lr=1.0,
                           final_lr_absolute=final_lr_absolute,
                           warmup_kwargs=warmup_kwargs
        )

    return S

def create_inputs(df, config_df, ids, metric='val_loss', cutoff_iter=(None, None)):
    """Concat data from multiple ids for fitting.

    Input columns: ['id', 't', 'lr']
    """

    input_list = list()
    target_list = list()

    assert len(set([config_df.loc[id, "scheduler"] for id in ids])) == 1, "more than one schedule found"
 
    for id in ids:
        base_lr = config_df.loc[id, "lr"]
        lb_iter = 0 if cutoff_iter[0] is None else cutoff_iter[0]
        ub_iter = cutoff_iter[1]

        this = df[df.id == id]
        this = this[(~this[metric].isna()) &
                     (this.iter >= lb_iter)]
        
        if ub_iter is not None:
            this = this[this.iter <= ub_iter]

        N = len(this)
        
        this_input = pd.DataFrame(np.column_stack((this.iter.values.astype('int'),
                                                   base_lr*np.ones(N))
                                                ),
                                  columns=['t', 'lr']
        )
        this_input['id'] = id

        input_list.append(this_input)
        target_list.append(this[metric].values)

    inputs = pd.concat(input_list).reset_index(drop=True)
    targets = np.hstack(target_list)
    
    # data processing and checks
    inputs['t'] = inputs['t'].astype("int")

    assert len(inputs) == len(targets), "Length of inputs and targets does not match."
    return inputs, targets

#%% Data Loading

# keys that are extracted from the log, stored in config dataframe
CONFIG_KEYS = ["scheduler", "lr", "iterations", "warmup_steps", "decay_type", "cos_final_lr", "wsd_fract_decay"]


def default_config():
    """Fields needed for getting the corresponding file name."""
    return {'dataset': 'slimpajama', 'model': '124m', 'scheduler': 'cos', 'lr': 0.001, 'iterations': 25_000, 'decay_type': 'sqrt', 'wsd_fract_decay': 0.2}

def name_mapping(config):
    sched = config['scheduler']
    model = config['model']
    dataset = config['dataset']
    lr = config['lr']
    iter = config['iterations']
    decay_type = config['decay_type']
    wsd_fract_decay = config['wsd_fract_decay']

    if dataset == 'slimpajama':
        if model.lower() == '124m':
            if sched == 'cos':
                name = f"cos-zero-124m_slimpajama_llama_norm_nlayers12_nhead12_lr{lr}_sched_cos_warmup300_decay_sqrt_0.0_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
            elif sched == 'wsd':
                name = f"ann124-{wsd_fract_decay}_slimpajama_llama_norm_nlayers12_nhead12_lr{lr}_sched_wsd_warmup300_decay_{decay_type}_{wsd_fract_decay}_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '124m_horizon':
            if iter == 50_000:
                name = f"slimpajama_chunk1_llama_norm_nlayers12_nhead12_lr{lr}_sched_{sched}_warmup300_decay_linear_{wsd_fract_decay}_iter{iter}_bs50x2_ws2_seed0_data_seed1337"    
            else:
                name = f"extend-50k_slimpajama_chunk1_llama_norm_nlayers12_nhead12_lr{lr}_sched_{sched}_warmup300_decay_linear_{wsd_fract_decay}_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '210m':
            if sched == 'cos':
                name = f"cos-210m_slimpajama_llama_nlayers24_nhead12_lr{lr}_sched_cos_warmup300_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
            elif sched == 'wsd':
                name = f"ann210-{wsd_fract_decay}_slimpajama_llama_nlayers24_nhead12_lr{lr}_sched_wsd_warmup300_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '210m_horizon':
            if iter == 50_000:
                name = f"slimpajama_chunk1_llama_norm_nlayers24_nhead12_lr{lr}_sched_{sched}_warmup300_decay_linear_{wsd_fract_decay}_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
            else:
                name = f"extend-50k_slimpajama_chunk1_llama_norm_nlayers24_nhead12_lr{lr}_sched_{sched}_warmup300_decay_linear_{wsd_fract_decay}_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
        else:
            raise Exception(f"No name mapping yet for {config}.")
    
    elif dataset == 'openwebtext':
        if model.lower() == '60m':
            nlayers, nhead =  10, 8
        elif model.lower() == '93m':
            nlayers, nhead =  12, 10
        if model.lower() == '166m':
            nlayers, nhead =  12, 14

        if sched == 'cos':
            name = f"openwebtext2_llama_norm_nlayers{nlayers}_nhead{nhead}_lr{lr}_sched_cos_warmup300_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
        elif sched == 'wsd':
            name = f"ann0.2_openwebtext2_llama_norm_nlayers{nlayers}_nhead{nhead}_lr{lr}_sched_wsd_warmup300_iter{iter}_bs50x2_ws2_seed0_data_seed1337"

    else:
        raise Exception(f"No name mapping yet for {config}.")
    return name

def name_mapping_wsd_base(config):
    model = config['model']
    dataset = config['dataset']
    lr = config['lr']
    
    if dataset == 'slimpajama':
        if model.lower() == '124m':
            name = f"stable-124m_slimpajama_llama_norm_nlayers12_nhead12_lr{lr}_sched_wsd_warmup300_decay_linear_0.0_iter50000_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '124m_horizon':
            name = f"slimpajama_chunk1_llama_norm_nlayers12_nhead12_lr{lr}_sched_wsd_warmup300_decay_linear_0.2_iter50000_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '210m':
            name = f"stable-210m_slimpajama_llama_nlayers24_nhead12_lr{lr}_sched_wsd_warmup300_iter40000_bs50x2_ws2_seed0_data_seed1337"
        elif model.lower() == '210m_horizon':
            name = f"slimpajama_chunk1_llama_norm_nlayers24_nhead12_lr{lr}_sched_wsd_warmup300_decay_linear_0.2_iter50000_bs50x2_ws2_seed0_data_seed1337"
        else:
            raise Exception(f"No name mapping yet for {config}.")
    elif dataset == 'openwebtext':
        if model.lower() == '60m':
            nlayers, nhead, iter =  10, 8, 17_500
        elif model.lower() == '93m':
            nlayers, nhead, iter =  12, 10, 25_000
        elif model.lower() == '166m':
            nlayers, nhead, iter =  12, 14, 30_000
        else:
            raise Exception(f"No name mapping yet for {config}.")
        
        name = f"openwebtext2_llama_norm_nlayers{nlayers}_nhead{nhead}_lr{lr}_sched_wsd_warmup300_iter{iter}_bs50x2_ws2_seed0_data_seed1337"
    return name


def merge_wsd_base(df_cooldown, conf_cooldown, df_base, conf_base):
    """For WSD runs, the logs until cooldown come from one long base run.
    So we need to merge the log until cooldown into the log after cooldown."""

    assert conf_base.lr == conf_cooldown.lr, "Lrs of base and cooldown are not matching."

    first_cooldown_iter = df_cooldown.iter.min()
    cooldown_start = conf_cooldown.iterations * (1-conf_cooldown.wsd_fract_decay)

    if cooldown_start > first_cooldown_iter:
        warnings.warn(f"Theoretical first log {cooldown_start} does not match actual first log {first_cooldown_iter}.")
        print("Taking overlap from cooldown log. Overlap has lr (first/last): ", df_cooldown[df_cooldown.iter.between(first_cooldown_iter, cooldown_start)]['train_lr'].values[[1,-1]])

    ixx = (df_base.iter < min(cooldown_start, first_cooldown_iter))
    prepend = df_base.loc[ixx, :]
    prepend.loc[:, "id"] = conf_cooldown.name

    new = pd.concat([prepend, df_cooldown]).sort_values("iter").reset_index(drop=True)

    assert len(new["id"].unique()) == 1
    assert len(new["iter"].unique()) == len(new["iter"]), "Found log for some iteration twice."

    return new

def create_id(res):
    scheduler = res['config']['scheduler']
    lr = res['config']['lr']
    iter = res['config']['iterations']
    if scheduler == 'cos':
        id = f"cos_lr{lr}_iter{iter}"
    elif scheduler == 'wsd' or scheduler == 'wsd_twostage':
        decay_type = res['config']['decay_type'] if 'decay_type' in res['config'] else 'linear'
        id = f"{scheduler}_lr{lr}_iter{iter}_{decay_type}_{res['config']['wsd_fract_decay']}"

    return id

def create_single_frame(res):
    #=============== CREATE ID ========
    id = create_id(res)

    #=============== CONFIG DATA + FINAL METRICS ========
    conf_df = pd.Series(name=id)
    
    for c in CONFIG_KEYS:
        conf_df[c] = res["config"].get(c)

    conf_df["final_train_loss"] = res["summary"]["train/loss"]
    conf_df["final_val_loss"] = res["summary"]["final-val/loss"]
    conf_df["final_val_perplexity"] = res["summary"]["final-val/perplexity"]

    #=============== LOG DATA ========
    train = pd.DataFrame(res["history"]["train"])
    res["history"]["val"].pop("val_acc")  # length of accuracy log is different, we dont need it
    val = pd.DataFrame(res["history"]["val"])

    # we might have multiple loggings for same iter --> for now just take the mean
    # this also sorts by iter
    train = train.groupby("iter").mean().reset_index()
    val = val.groupby("iter").mean().reset_index()

    df = train.merge(val, on="iter", how="outer")    
    df["id"] = id

    return conf_df, df

def load_multiple(config_list, data_dir="data"):
    log_df_list = list()        # for log data
    config_df_list = list()     # for config data

    # Load each
    for c in config_list:
        conf = default_config()
        conf.update(c)
        
        name = name_mapping(conf)
        # runs with decay=0.994 or lr=0.003 are only in grad_norm folder
        # --> temporarily change data_dir
        if (conf['lr'] == 0.003) or (conf.get('wsd_fract_decay')==0.994):
            if "grad_norm" not in data_dir:
                this_data_dir = os.path.join(data_dir, "grad_norm")
        else:
            this_data_dir = data_dir

        try:
            with open(os.path.join(this_data_dir, f"{name}.json"), "r") as f:
                res = json.load(f)                                              # load
        except:
            print("Could not find or load the file:")
            print(name)
            continue

        conf_df, df = create_single_frame(res)                                  # create frame
        
        # Append logs from stable phase for wsd
        if conf['scheduler'] in ['wsd', 'wsd_twostage']:
            name2 = name_mapping_wsd_base(conf)
            with open(os.path.join(this_data_dir, f"{name2}.json"), "r") as f:
                res2 = json.load(f)
            wsd_base_conf, wsd_base_df = create_single_frame(res2)
            df = merge_wsd_base(df, conf_df, wsd_base_df, wsd_base_conf)        # prepend until cooldown
        
        log_df_list.append(df)                                                  # append
        config_df_list.append(conf_df)
        
    # Concat and final modifications
    df = pd.concat(log_df_list)
    df.sort_values(["id", "iter"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    config_df = pd.concat(config_df_list, axis=1).T

    return df, config_df