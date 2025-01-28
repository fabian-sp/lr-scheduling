import numpy as np

from .wsd import WSDSchedule

# constant by having no cooldown step at all
class ConstantSchedule(WSDSchedule):
    def __init__(self,
                 steps=100, 
                 base_lr: float=1.0,
                 **kwargs
    ):
        
        self._steps = steps
        super().__init__(base_lr=base_lr,
                         steps=steps,
                         cooldown_len=0.0,
                         **kwargs
                         )

    @property
    def name(self):
        return 'constant'
