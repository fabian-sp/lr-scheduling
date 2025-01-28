import numpy as np

from scheduled import WSDSchedule, CosineSchedule, SqrtSchedule

ASSERT_TOL = 1e-10

def test_wsd_basic():
    T = 100
    S = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
    )
    assert len(S.schedule) == T
    assert S.schedule[-1] == 0.0
    
    # in 1-indexing S._cooldown_start_iter is the cooldown start (T_0)
    # --> adjust to 0-indexing by substracting one
    assert S.schedule[S._cooldown_start_iter-1] == S.get_base_lr()  # at t=T_0
    assert S.schedule[S._cooldown_start_iter] < S.get_base_lr()     # at t=T_0+1

    return

def test_linear_basic():
    T = 100
    warmup_kwargs = {"steps": 0}
    S = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=1.0,
                    base_lr=1.0,
                    warmup_kwargs=warmup_kwargs
    )

    assert S.schedule[0] == S.get_base_lr()
    assert S.schedule[-1] == 0.0
    assert S._cooldown_start_iter == 1
    S.set_base_lr(2.1)
    assert np.abs(S.schedule[int(T/2)]/S.get_base_lr() - 0.5) <= 1e-2

    return

def test_cosine_basic():
    T = 100
    S = CosineSchedule(final_lr=0.0,
                       steps=T,
                       base_lr=1.0
    )

    assert len(S.schedule) == T
    assert S.schedule[-1] == 0.0

def test_final_absolute_lr_wsd():
    T = 115
    S = WSDSchedule(final_lr=3e-3,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
                    final_lr_absolute=True
    )
    
    
    assert np.abs(S.schedule[-1] - 3e-3) <= ASSERT_TOL
    S.set_base_lr(2.1)
    assert np.abs(S.schedule[-1] - 3e-3) <= ASSERT_TOL

    return


def test_final_absolute_lr_cosine():
    T = 115
    S = CosineSchedule(final_lr=3e-3,
                       steps=T,
                       base_lr=1.0,
                       final_lr_absolute=True
    )
    assert np.abs(S.schedule[-1] - 3e-3) <= ASSERT_TOL
    S.set_base_lr(2.1)
    assert np.abs(S.schedule[-1] - 3e-3) <= ASSERT_TOL

    return


def test_warmup_wsd():
    T = 100
    warmup_kwargs = {"steps": 5,
                     "warmup_lr": 1e-7
    }
    S = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
                    warmup_kwargs=warmup_kwargs
    )

    # in 1-indexing, we reach base lr at t=warmup_steps+1
    assert S.schedule[S._warmup_steps] == S.get_base_lr()
    assert S.schedule[S._warmup_steps-1] < S.get_base_lr()
    
    return

def test_warmup_absolute_lr_wsd():
    T = 100
    warmup_kwargs= {"steps" : 5,
                    "warmup_lr": 3e-7,
                    "warmup_lr_absolute": True
    }
    S = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
                    warmup_kwargs=warmup_kwargs
    )

    assert np.abs(S.schedule[0] - S._warmup_kwargs["warmup_lr"]) <= ASSERT_TOL
    S.set_base_lr(2.1)
    assert np.abs(S.schedule[0] - S._warmup_kwargs["warmup_lr"]) <= ASSERT_TOL

    return



def test_warmup_cosine():
    T = 100
    warmup_kwargs= {"steps" : 5,
                    "warmup_lr": 1e-7,
    }
    S = CosineSchedule(final_lr=0.0,
                       steps=T,
                       base_lr=1.0,
                       warmup_kwargs=warmup_kwargs
    )

    # in 1-indexing, we reach base lr at t=warmup_steps+1
    assert S.schedule[S._warmup_steps] == S.get_base_lr()
    assert S.schedule[S._warmup_steps-1] < S.get_base_lr()
    
    return

def test_warmup_absolute_lr_cosine():
    T = 100
    warmup_kwargs= {"steps" : 5,
                    "warmup_lr": 3e-7,
                    "warmup_lr_absolute": True
    }
    S = CosineSchedule(final_lr=0.0,
                       steps=T,
                       base_lr=1.0,
                       warmup_kwargs=warmup_kwargs
    )

    assert np.abs(S.schedule[0] - S._warmup_kwargs["warmup_lr"]) <= ASSERT_TOL
    S.set_base_lr(2.1)
    assert np.abs(S.schedule[0] - S._warmup_kwargs["warmup_lr"]) <= ASSERT_TOL

    return

def test_sqrt_basic():
    T = 100
    S = SqrtSchedule(final_lr=0.0,
                     steps=T,
                     cooldown_len=0.2,
                     base_lr=1.0,
    )
    assert len(S.schedule) == T
    assert np.abs(S.schedule[-1]) <= ASSERT_TOL
    
    # in 1-indexing S._cooldown_start_iter is the cooldown start (T_0)
    # --> adjust to 0-indexing by substracting one
    last_lr = 1/np.sqrt(80)
    np.testing.assert_almost_equal(S.schedule[S._cooldown_start_iter-1], last_lr)   # at t=T_0
    
    return

def test_redundant_warmup():
    """should have 0 warmup steps --> same as no warmup"""
    T = 100
    warmup_kwargs = {"warmup_lr": 1e-1
    }
    S = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
                    warmup_kwargs=warmup_kwargs
    )
    S2 = WSDSchedule(final_lr=0.0,
                    steps=T,
                    cooldown_len=0.2,
                    base_lr=1.0,
                    warmup_kwargs=None
    )

    np.testing.assert_array_almost_equal(S.schedule, S2.schedule)
    return

