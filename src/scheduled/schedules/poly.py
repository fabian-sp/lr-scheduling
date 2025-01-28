import numpy as np

from .base import ScheduleBase

class PolynomialBackwardSchedule(ScheduleBase):
    def __init__(self, 
                 steps: int=100,
                 base_lr: float=1.0,
                 alpha: float=1.0,
                 final_lr: float=0.0,
                 warmup_kwargs: dict=None
        ):
        assert (final_lr>=0.0) and (final_lr<=1.0), "Final LR must be in [0, 1]."
        assert alpha > 0, f"alpha must be positive, but was given {alpha}."
        
        self._steps = steps
        self._final_lr = final_lr
        self._alpha = alpha

        super().__init__(steps=steps,
                         base_lr=base_lr,
                         warmup_kwargs=warmup_kwargs,
                         cooldown_kwargs=None
        )

    def _construct_main_schedule(self):
        """
        eta_t = ((T-t)/(T-1))^alpha
        where T = self._steps
        """
        _decay = lambda t: ((self._steps-(t+1))/(self._steps-1))**self._alpha
        sched = np.array([self._final_lr+(1-self._final_lr)*_decay(t) for t in range(self._steps)])
        return sched

    @property
    def name(self):
        return 'polynomial-backward'
    
class PolynomialCooldownSchedule(ScheduleBase):
    def __init__(self,
                 alpha: float=1.0,
                 final_lr: float=0.0,
                 steps: int=100,
                 base_lr: float=1.0,
                 cooldown_len=0.0,
                 decay_type='linear',
                 final_lr_absolute: bool=False,
                 warmup_kwargs: dict=None
        ):
        assert alpha > 0, f"alpha must be positive, but was given {alpha}"
        assert (cooldown_len>=0.0) and (cooldown_len<=1.0), "Cooldown length must be in [0,1] (its a percentage of steps)."
        
        cooldown_kwargs = {"steps": int(cooldown_len * steps),
                           "type": decay_type,
                           "final_lr": final_lr,
                           "final_lr_absolute": final_lr_absolute
        }

        self._alpha = alpha
        # self._final_lr = final_lr
        # self._cooldown_len = cooldown_len
        # self._decay_type = decay_type
        # self._final_lr_absolute = final_lr_absolute
        
        super().__init__(steps=steps,
                         base_lr=base_lr,
                         warmup_kwargs=warmup_kwargs,
                         cooldown_kwargs=cooldown_kwargs
        )

    def _construct_main_schedule(self):
        """
        eta_t = 1/(t^alpha)
        """
        poly_steps = self._steps - self._cooldown_steps - self._warmup_steps
        assert poly_steps > 0, f"Resulted in {poly_steps} polynomial steps, not possible."

        # polynomial decay part
        time = np.arange(poly_steps) + 1
        sched = 1/(time**self._alpha) 
        return sched

    @property
    def name(self):
        return 'polynomial-cooldown'