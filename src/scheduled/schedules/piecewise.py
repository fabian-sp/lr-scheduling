import numpy as np

from .base import ScheduleBase

class PiecewiseConstantSchedule(ScheduleBase):
    """
    If milestones = [m1, m2, ...], and factors = [f1, f2, ...]
    then 
    eta_t = prod(f_i) * base_lr with m_i <= t

    Warmup and cooldown periods are simply cutted out.
    """
    def __init__(self,
                 steps: int,
                 base_lr: float=1.0,
                 milestones: list=[],
                 factors: list=[],
                 cooldown_kwargs: dict=None,
                 warmup_kwargs: dict=None
    ):
        assert len(factors) == len(milestones), "Length of milestones and factors are different."
        assert np.all([m >= 1 for m in milestones]), "Milestones must be all integers."
        assert np.all([f > 0 for f in factors]), "Factors must be all positive."
        
        self._milestones = milestones
        self._factors = factors

        super().__init__(steps=steps,
                         base_lr=base_lr,
                         cooldown_kwargs=cooldown_kwargs,
                         warmup_kwargs=warmup_kwargs
        )

    def _helper_piecewise_multiply(self, t):
        v = 1.0
        for m, f in sorted(zip(self._milestones, self._factors)):
            ind = int(t >= m)
            v = ind * f * v + (1-ind) * v
        return v

    def _construct_main_schedule(self):
        time = np.arange(self._steps)
        full = np.array([self._helper_piecewise_multiply(t) for t in time])
        return full[self._warmup_steps: self._steps-self._cooldown_steps]
    
    @property
    def name(self):
        return 'piecewise-constant'
