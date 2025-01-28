import numpy as np

from .base import ScheduleBase

class CosineSchedule(ScheduleBase):
    def __init__(self, 
                 final_lr: float=0.0, 
                 steps: int=100,
                 base_lr: float=1.0,
                 final_lr_absolute: bool=False,
                 warmup_kwargs: dict=None
        ):
        """ Cosine decay schedule
        """
        assert (final_lr>=0.0) and (final_lr<=base_lr), "Final LR must be in [0, base_lr]."
        self._final_lr = final_lr
        self._steps = steps
        self._final_lr_absolute = final_lr_absolute
        
        super().__init__(base_lr=base_lr,
                         steps=steps,
                         cooldown_kwargs=None,
                         warmup_kwargs=warmup_kwargs,
                         )

        # if final lr is independent of base lr, recompute schedule every time
        if self._final_lr_absolute:
            self._recompute_sched_each_time = True


    def _construct_main_schedule(self):
        cosine_steps = self._steps - self._warmup_steps
        half_cos = np.cos(np.arange(cosine_steps)/(cosine_steps-1) * np.pi)
        _target_lr = self._final_lr if not self._final_lr_absolute else self._final_lr/self._base_lr
        return _target_lr + 0.5*(1.0-_target_lr)*(1+half_cos)
    
    @property
    def name(self):
        return 'cosine'
