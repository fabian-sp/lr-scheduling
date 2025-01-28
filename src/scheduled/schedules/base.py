import numpy as np
from typing import Union
from abc import abstractmethod

import copy

ALMOST_ZERO_ALLOWED = -1e-15

DEFAULT_WARMUP_KWARGS = {"steps": 0,
                         "warmup_lr": 1e-10,
                         "warmup_lr_absolute": False
}

DEFAULT_COOLDOWN_KWARGS = {"steps": 0,
                           "type": "linear",
                           "final_lr": 1e-10,
                           "final_lr_absolute": False
}


class ScheduleBase:
    def __init__(self,
                 steps: int,
                 base_lr=1.0,
                 warmup_kwargs: dict=None,
                 cooldown_kwargs: dict=None                 
        ):
        
        # Assertions
        assert base_lr >= 0.0, "Base LR must be non-negative"
        
        # Assignments
        self._steps = steps
        self._base_lr = base_lr
        
        # flag to recompute schedule everytime the base_lr is changed
        # this is necessary, if for example, the final LR is chosen as absolute value, independent of base_lr
        # might be overwritten in subclass init
        self._recompute_sched_each_time = False

        # for no warmup use warmup_kwargs=None
        if warmup_kwargs is not None:
            self._warmup_kwargs = copy.deepcopy(DEFAULT_WARMUP_KWARGS)
            self._warmup_kwargs.update(warmup_kwargs)
            self._recompute_sched_each_time = self._recompute_sched_each_time or self._warmup_kwargs["warmup_lr_absolute"]
        else:
            self._warmup_kwargs = None
        
        # for no cooldown use cooldown_kwargs=None
        if cooldown_kwargs is not None:
            self._cooldown_kwargs = copy.deepcopy(DEFAULT_COOLDOWN_KWARGS)
            self._cooldown_kwargs.update(cooldown_kwargs)
            self._recompute_sched_each_time = self._recompute_sched_each_time or self._cooldown_kwargs["final_lr_absolute"]
        else:
            self._cooldown_kwargs = None

        # we store the unscaled schedule in ._schedule, the scaled one by .schedule()
        self._schedule = self._construct_schedule()
        
        # do some checks
        self._check_schedule()

        
    @abstractmethod
    def _construct_main_schedule(self):
        """Construct main part of schedule (without warmup and cooldown). Override this method."""
        pass
    
    def _check_warmup_cooldown_length(self):
        """If warmup+cooldown length == total length, then modify them. 
        This is relevant for linear-decay (WSD with cooldown length=1.0)."""

        if self._warmup_steps + self._cooldown_steps == self._steps:
            if self._cooldown_kwargs is not None:
                self._cooldown_kwargs["steps"] -= 1
            else:
                self._warmup_kwargs["steps"] -= 1

    def _construct_schedule(self):
        """Combines warmup + main schedule + cooldown."""

        self._check_warmup_cooldown_length()
        main_sched = self._construct_main_schedule()
        warmup = self._construct_warmup()
        cooldown = self._construct_cooldown(last_main_lr=main_sched[-1])
        
        sched = np.concatenate((warmup,         # Warmup    
                                main_sched,     # Main
                                cooldown)       # Cooldown
        )

        return sched    
    
    @property
    def schedule(self):
        return self.get_base_lr() * self._schedule

    def _check_schedule(self):
        assert np.all(self._schedule >= ALMOST_ZERO_ALLOWED), "Schedule must be non-negative"
        assert len(self.schedule) == self._steps, "Steps are not equal to schedule length."
        #TODO: add check that max of schedule is one?

    def get_base_lr(self):
        return self._base_lr
    
    def set_base_lr(self, base_lr):
        assert base_lr >= 0.0, "Base LR must be non-negative"
        self._base_lr = base_lr

        if self._recompute_sched_each_time:
            self._schedule = self._construct_schedule()
            self._check_schedule()

    def _construct_warmup(self):
        """Generates a linear warmup.
        Such that 1.0 would be reached in step after to avoid a one-step plateau.
        """
        if self._warmup_kwargs is None:
            return np.array([])
        else:
            _warmup_lr_absolute = self._warmup_kwargs["warmup_lr_absolute"]
            _warmup_lr = self._warmup_kwargs["warmup_lr"]
            _warmup_steps = self._warmup_kwargs["steps"]

            assert _warmup_lr > 0 and _warmup_lr <= 1.0, f"{_warmup_lr} is not a valid warmup_lr, must be in (0,1]."
        
            _wup_lr = _warmup_lr if not _warmup_lr_absolute else _warmup_lr/self._base_lr
            _warmup = lambda t : _wup_lr + (t/_warmup_steps) * (1.0-_warmup_lr)
            
            return np.array([_warmup(t) for t in range(_warmup_steps)])
    
    @property
    def _warmup_steps(self):
        return 0 if self._warmup_kwargs is None else self._warmup_kwargs["steps"]

    def _construct_cooldown(self, last_main_lr: float):
        """Generates a linear or 1-sqrt cooldown.
        """

        if self._cooldown_kwargs is None:
            return np.array([])
        else:
            _decay_steps = self._cooldown_kwargs["steps"]
            _decay_type = self._cooldown_kwargs["type"]
            _final_lr = self._cooldown_kwargs["final_lr"]
            _final_lr_absolute = self._cooldown_kwargs["final_lr_absolute"]

            # this is equal to T_0 in the draft
            self._cooldown_start_iter = self._steps - _decay_steps

            _target_lr = _final_lr if not _final_lr_absolute else _final_lr/self._base_lr
            assert _target_lr <= last_main_lr, f"Final LR of main schedule is smaller than cooldown final LR ({last_main_lr} and {_target_lr})."
            if _decay_type == 'linear':
                _decay = lambda t: last_main_lr - (t+1)*(last_main_lr-_target_lr)/_decay_steps
            elif _decay_type == 'sqrt':
                _decay = lambda t: _target_lr + (last_main_lr-_target_lr) * (1.0 - np.sqrt((t+1)/_decay_steps))
            else:
                raise NotImplementedError(f"Unknown decay type {_decay_type}.")

            return np.array([_decay(t) for t in range(_decay_steps)])
    
    @property
    def _cooldown_steps(self):
        return 0 if self._cooldown_kwargs is None else self._cooldown_kwargs["steps"]
    
    def __len__(self):
        return len(self._schedule)
    
    @property
    def name(self):
        return None

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except:
            print("Can not plot as matplotlib is not installed")
            return

        time = np.arange(1, len(self.schedule)+1)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(time, self.schedule)
        ax.set_xlabel("iteration t")
        ax.set_ylabel("learning rate")
        ax.grid(which='both', lw=0.2, ls='--')
        ax.set_ylim(0,)
        fig.tight_layout()

        return 

    def compute_rate(self,
                     grad_norms: Union[float, np.ndarray],
                     D: float,
                     T: int=None,
                     type: str='refined',
                     **kwargs
    ):
        """ General interface for computing the rate.
        """
        # input checks
        if T is None:
            T = len(etas)
        else:
            assert T <= len(self.schedule),     "T must be less than the length of the schedule"
            assert T >= 1,                      "T must be positive"
        
        if isinstance(T, float):
            T = int(T)

        if type == 'refined':
            return self._compute_refined_rate(grad_norms=grad_norms,
                                              D=D,
                                              T=T,
                                              **kwargs
                    )
        elif type == 'standard':
            return self._compute_standard_rate(grad_norms=grad_norms,
                                               D=D,
                                               T=T
                    )
        else:
            raise KeyError(f"Unknown rate computation type {type}.")
        
    def _compute_refined_rate(self,
                              grad_norms: Union[float, np.ndarray],
                              D: float,
                              T: int=None,
                              return_split: bool=False,
                              eps: float=0.0
    ): 
        """Convergence rate from Theorem 10 in https://arxiv.org/pdf/2310.07831"""
        etas = self.schedule.copy()

        _sum_etas = etas[0:T].sum()
        _cumsum_etas = etas[0:T].cumsum()
        _eta_grad = (etas**2 * grad_norms**2)[0:T]
        _sum_eta_grad = _eta_grad.sum()
        _cumsum_eta_grad = _eta_grad.cumsum()

        term1 = (D**2)/(2*_sum_etas)
        term2 = _sum_eta_grad/(2*_sum_etas)
        
        #========== Notation ===============
        # alpha_k = sum_{k+1}_{T} eta_t
        # beta_k = sum_{k}_{T} eta_t
        # gamma_k = sum_{k}_{T} eta_t**2 grad_norm_t**2
        # compute: 0.5 * sum_{k=1}_{T-1} eta_k/alpha_k * (gamma_k/beta_k)

        _alpha = _sum_etas - _cumsum_etas
        _alpha[-1] += 1e-10 # avoid warning for divide by zero, has no effect as we have [:T-1]
        _betas = _alpha + etas[0:T]
        _gammas = _sum_eta_grad - _cumsum_eta_grad + _eta_grad
        term3 = (1/2)*((etas[0:T]/_alpha)*((_gammas/_betas)+eps))[:T-1].sum()  

        if return_split:
            return (term1, term2, term3)
        else:
            return term1+term2+term3
        
    def _compute_standard_rate(self,
                               grad_norms: Union[float, np.ndarray],
                               D: float,
                               T: int=None
    ):
        """Standard convergence rate for convex, evaluated at T"""
        etas = self.schedule.copy()
        return 1/(2*etas[0:T].sum()) * (D**2 + ((etas**2 * grad_norms**2)[0:T]).sum())



def compute_optimal_base(schedule: ScheduleBase,
                         G: Union[float, np.ndarray],
                         D: float,
                         T: int=None,
                         type: str='refined'
):
    """for given schedule and hparams, compute optimal base_lr"""
    
    # get identical schedule with base lr of one
    _s = copy.copy(schedule)
    _s.set_base_lr(1.0)

    if type == 'refined':
        # compute individual rate terms --> return_split=True
        t1, t2, t3 = _s.compute_rate(grad_norms=G,
                                     D=D,
                                     T=T,
                                     type='refined',
                                     return_split=True
        )
        opt_base_lr = np.sqrt(t1/(t2+t3))
        opt_rate = 2*np.sqrt(t1*(t2+t3))
    
    elif type == 'standard':
        etas = _s.schedule.copy()
        a = ((etas*G)[0:T]**2).sum()
        opt_base_lr = D/np.sqrt(a)
        opt_rate = D*np.sqrt(a)/(etas[0:T].sum())

    else:
        raise KeyError(f"Unknown rate computation type {type}.")

    return opt_base_lr, opt_rate
