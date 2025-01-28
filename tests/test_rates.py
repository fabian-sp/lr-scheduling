import numpy as np
import pytest
from scheduled import ConstantSchedule, compute_optimal_base
from scheduled.utils import harmonic_number

ASSERT_TOL = 1e-10


@pytest.mark.parametrize('D', [1,5,10])
@pytest.mark.parametrize('G', [1,5,10])
@pytest.mark.parametrize('T', [10, 100, 1000])
def test_rate_constant_schedule(D, G, T):
    gamma = 0.5

    S = ConstantSchedule(steps=T,
                         base_lr=gamma
    )

    for t in np.arange(1,T+1):
        terms = S.compute_rate(grad_norms=G,
                               D=D,
                               T=t,
                               return_split=True
        )

        expected1 = (D**2)/(2*gamma*t)
        expected2 = gamma*(G**2)/2
        expected3 = (gamma/2) * G**2 * harmonic_number(t-1)

        np.testing.assert_array_almost_equal(np.array(terms), 
                                             np.array([expected1, expected2, expected3])
                                             )
        

        _, best_rate = compute_optimal_base(S, G, D, t)
        exp_rate = 2*np.sqrt(terms[0]*(terms[1]+terms[2]))
        assert np.abs(best_rate - exp_rate) <= ASSERT_TOL

    return