import numpy as np
import pytest
from scheduled import WSDSchedule, compute_optimal_base
from scheduled.utils import harmonic_number

    
def theorem_bound(T, T0, D, G):

    assert T0 < T

    t1 = 2/3
    t2 = (T+2*T0)/(3*(T+T0))
    t3 = harmonic_number(T+T0-2) - harmonic_number(T-T0+1)
    t4 = - ((T-T0)*(T0-1))/(3*(T-T0+2)*(T+T0))
    t5 = 1/(T-T0)**2
    t6 = harmonic_number(T-T0-1)/(T-T0+1)
    a = 4*(t1+t2+t3+t4+t5+t6)/(T+T0)
    return D*G*np.sqrt(a)

@pytest.mark.parametrize('D', [1,5,10])
@pytest.mark.parametrize('G', [1,5,10])
def test_wsd_theorem(D, G, all_T=np.arange(100, 5000, 200), result=False):
    """tests the bound derived for WSD schedule."""
    all_omega = list()
    all_thm_bound = list()
    for T in all_T:
        S = WSDSchedule(final_lr=0, steps=T+1, cooldown_len=0.2, decay_type='linear')

        best_base_lr, best_rate = compute_optimal_base(S,
                                                    G=G,
                                                    D=D,
                                                    T=T
        )

        T0 = S._cooldown_start_iter
        S.set_base_lr(best_base_lr)

        all_thm_bound.append(theorem_bound(T, T0, D, G))
        all_omega.append(best_rate)

    assert np.all(all_omega <= all_thm_bound)

    if result:
        return all_omega, all_thm_bound

#=============================================================
## If you want to plot Omega_T vs the bound from the theorem
#=============================================================

# import matplotlib.pyplot as plt
# from scheduled.utils import set_plot_aesthetics
# set_plot_aesthetics()
# %matplotlib qt5


# all_T = np.arange(100, 20_000, 200)
# all_omega, all_thm_bound = test_wsd_theorem(D=1, G=1, all_T=all_T, result=True)
# fig, ax = plt.subplots(figsize=(4.5,3))
# ax.tick_params(direction='in')
# ax.plot(all_T, all_omega, c='k', lw=2, label=r"$\Omega_T$")
# ax.plot(all_T, all_thm_bound, c='silver', lw=2, ls='--', label=r"Bound Thm. 9")
# ax.set_xlabel(r"$T$")
# ax.legend()
# fig.tight_layout()
