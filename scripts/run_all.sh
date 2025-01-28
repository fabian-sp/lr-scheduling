python multiple_horizon.py
python cooldown_length.py
python grad_norm_shape.py
python schedule_comparison.py
python cosine_cycle.py
python horizon_transfer.py --schedule wsd
python horizon_transfer.py --schedule wssd
python horizon_transfer.py --schedule sqrt
python lr_transfer.py
#
python multiple_horizon.py --ablation
python cooldown_length.py --ablation
python grad_norm_shape.py --ablation
python schedule_comparison.py --ablation
python cosine_cycle.py --ablation
#
python cooldown_length.py --decay_type sqrt
python convex_example.py
#
# python lower_bound.py --schedule wsd
# python lower_bound.py --schedule cosine
# python lower_bound.py --schedule linear
# python lower_bound.py --schedule sqrt
# python lower_bound.py --schedule constant