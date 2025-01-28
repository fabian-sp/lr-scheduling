import numpy as np

from .poly import PolynomialCooldownSchedule

class SqrtSchedule(PolynomialCooldownSchedule):
    def __init__(self, 
                 final_lr: float=0.0,
                 steps: int=100,
                 base_lr: float=1.0,
                 cooldown_len=0,
                 decay_type='linear',
                 final_lr_absolute: bool=False,
                 warmup_kwargs: dict=None
        ):
        super().__init__(alpha=0.5,
                         final_lr=final_lr,
                         steps=steps,
                         cooldown_len=cooldown_len,
                         decay_type=decay_type,
                         base_lr=base_lr,
                         final_lr_absolute=final_lr_absolute,
                         warmup_kwargs=warmup_kwargs
        )

    @property
    def name(self):
        return 'sqrt'