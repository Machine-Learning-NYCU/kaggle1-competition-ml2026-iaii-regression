import numpy as np
from mae_models import SCALE, encode_target, decode_target, learn_mae_blend
from finalize_mae import optimal_price_scale


def test_all_training_modes_equal_price_absolute_error():
    y = np.array([1e6, 9e6, 8e7, 3e9])
    anchor = np.array([2e6, 7e6, 5e7, 2e9])
    for mode in ['price', 'ratio', 'additive']:
        t, w = encode_target(y, anchor, mode)
        z = t + np.array([.1, -.2, .5, -.1])
        np.testing.assert_allclose(w * np.abs(t - z), np.abs(y - decode_target(z, anchor, mode)) / SCALE)
        np.testing.assert_allclose(decode_target(t, anchor, mode), y)


def test_mae_convex_blend_and_median_behavior():
    # L1 must follow the majority, unlike squared error which favors the rare giant error.
    y = np.r_[np.full(99, 10.), 1000.]
    matrix = np.column_stack([np.full(100, 10.), np.full(100, 19.9)])
    w, result = learn_mae_blend(matrix, y, penalty=0)
    np.testing.assert_allclose(w, [1, 0], atol=1e-7)
    np.testing.assert_allclose(result['calibration_mae'], 9.9)
    assert (w >= 0).all() and abs(w.sum()-1) < 1e-8


def test_mae_blend_is_invariant_to_monetary_unit():
    y = np.arange(20.) + 50
    p = np.column_stack([y*.9, y*1.1, y+4])
    w1, _ = learn_mae_blend(p, y)
    w2, _ = learn_mae_blend(p*1e6, y*1e6)
    np.testing.assert_allclose(w1, w2, atol=1e-6)


def test_scale_is_weighted_median_and_respects_bound():
    y, p = np.array([101., 105., 200.]), np.array([100., 100., 10.])
    a = optimal_price_scale(y, p)
    np.testing.assert_allclose(a, 1.05)
    loss = lambda s: np.mean(np.abs(y-s*p))
    assert loss(a) <= loss(1.049)
    assert optimal_price_scale(y*10, p) == 1.05
