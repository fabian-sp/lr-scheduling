import numpy as np

from .base import ScheduleBase

class WSDSchedule(ScheduleBase):
    def __init__(self,
                 final_lr: float=0.0,
                 steps: int=100,
                 base_lr: float=1.0,
                 cooldown_len: float=0.2,
                 decay_type: str='linear',
                 final_lr_absolute: bool=False,
                 warmup_kwargs: dict=None
    ):
        assert (final_lr>=0.0) and (final_lr<=base_lr), "Final LR must be in [0, base_lr]."
        assert (cooldown_len>=0.0) and (cooldown_len<=1.0), "Cooldown length must be in [0,1] (its a percentage of steps)."
        
        cooldown_kwargs = {"steps": int(cooldown_len * steps),
                           "type": decay_type,
                           "final_lr": final_lr,
                           "final_lr_absolute": final_lr_absolute
        }

        super().__init__(steps=steps,
                         base_lr=base_lr,
                         cooldown_kwargs=cooldown_kwargs,
                         warmup_kwargs=warmup_kwargs
        )

    def _construct_main_schedule(self):
        stable_steps = self._steps - self._cooldown_steps - self._warmup_steps
        assert stable_steps > 0, f"Resulted in {stable_steps} constant steps, not possible."    
        return np.ones(stable_steps)
    
    @property
    def name(self):
        return 'wsd'