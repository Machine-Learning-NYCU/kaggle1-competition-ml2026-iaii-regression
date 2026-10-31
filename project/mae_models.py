"""Price-scale L1 models and convex, regularized MAE blending.

For positive anchor B, target y/B with weight B/S gives exactly
(B/S)*abs(y/B - f(x)) = abs(y - B*f(x))/S. No log-loss surrogate.
"""
import numpy as np
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye

from features import CategoryEncoder

SCALE = 1e7
SPECS = {
    'cat_price_mae': {'kind': 'cat', 'mode': 'price', 'depth': 8},
    'cat_ratio_mae': {'kind': 'cat', 'mode': 'ratio', 'depth': 8},
    'lgb_ratio_mae': {'kind': 'lgb', 'mode': 'ratio', 'leaves': 63},
    'lgb_additive_mae': {'kind': 'lgb', 'mode': 'additive', 'leaves': 63},
}


def anchor_price(x, offset):
    return np.exp(np.clip(np.asarray(offset, float) + x['market_project'].to_numpy(float), 0, np.log(1e12)))


def encode_target(y, anchor, mode):
    y, anchor = np.asarray(y, float), np.asarray(anchor, float)
    if mode == 'ratio':
        return y / anchor, anchor / SCALE
    if mode == 'additive':
        return (y - anchor) / SCALE, np.ones(len(y))
    if mode == 'price':
        return y / SCALE, np.ones(len(y))
    raise ValueError(mode)


def decode_target(z, anchor, mode):
    if mode == 'ratio':
        return np.asarray(z) * anchor
    if mode == 'additive':
        return np.asarray(z) * SCALE + anchor
    if mode == 'price':
        return np.asarray(z) * SCALE
    raise ValueError(mode)


def fit_model(spec, x, y, offset, cats, *, iterations=3500, threads=16,
              device='GPU', validation=None, seed=2026):
    target, weight = encode_target(y, anchor_price(x, offset), spec['mode'])
    encoder = None
    if validation is not None:
        vx, vy, vo = validation
        vt, vw = encode_target(vy, anchor_price(vx, vo), spec['mode'])
    if spec['kind'] == 'cat':
        model = CatBoostRegressor(
            iterations=iterations, learning_rate=.045, depth=spec['depth'],
            loss_function='MAE', eval_metric='MAE', l2_leaf_reg=12,
            random_strength=.5, bootstrap_type='Bayesian', bagging_temperature=.5,
            max_ctr_complexity=1, one_hot_max_size=10, has_time=True,
            task_type=device, devices='0' if device == 'GPU' else None,
            thread_count=threads, random_seed=seed, verbose=500, allow_writing_files=False)
        params = {}
        if validation is not None:
            params.update(eval_set=Pool(vx, vt, cat_features=cats, weight=vw), early_stopping_rounds=200)
        model.fit(Pool(x, target, cat_features=cats, weight=weight), **params)
        best = model.tree_count_
    else:
        encoder = CategoryEncoder().fit(x, cats)
        model = lgb.LGBMRegressor(
            n_estimators=iterations, learning_rate=.04, num_leaves=spec['leaves'],
            min_child_samples=80, reg_lambda=20, reg_alpha=.1,
            colsample_bytree=.85, subsample=.85, subsample_freq=1,
            cat_smooth=30, cat_l2=20, max_cat_threshold=64,
            objective='regression_l1', n_jobs=threads, random_state=seed,
            verbosity=-1, deterministic=True, force_col_wise=True)
        params = {}
        if validation is not None:
            params.update(eval_X=encoder.transform(vx), eval_y=vt, eval_sample_weight=[vw],
                          eval_metric='l1', callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        model.fit(encoder.transform(x), target, sample_weight=weight, **params)
        best = model.best_iteration_ or model.n_estimators_
    return {'model': model, 'encoder': encoder, 'spec': spec, 'iterations': int(best)}


def predict(bundle, x, offset):
    xx = bundle['encoder'].transform(x) if bundle['encoder'] is not None else x
    z = bundle['model'].predict(xx)
    result = decode_target(z, anchor_price(x, offset), bundle['spec']['mode'])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite model predictions')
    return np.maximum(result, 1)


def learn_mae_blend(matrix, y, penalty=.02):
    """Globally solve MAE + lambda*||w-uniform||_1 on the probability simplex."""
    matrix, y = np.asarray(matrix, float), np.asarray(y, float)
    n, k = matrix.shape
    if len(y) != n or n == 0 or not np.isfinite(matrix).all() or not np.isfinite(y).all():
        raise ValueError('Invalid blend inputs')
    prior = np.full(k, 1. / k)
    scale = max(float(np.mean(np.abs(matrix @ prior - y))), 1.)
    a, b = csr_matrix(matrix / scale), y / scale
    # Variables [w (k), absolute errors (n), absolute deviations from prior (k)].
    identity = eye(n, format='csr')
    small = eye(k, format='csr')
    constraints = bmat([[a, -identity, None], [-a, -identity, None],
                        [small, csr_matrix((k, n)), -small],
                        [-small, csr_matrix((k, n)), -small]], format='csr')
    rhs = np.r_[b, -b, prior, -prior]
    objective = np.r_[np.zeros(k), np.full(n, 1. / n), np.full(k, penalty)]
    equality = csr_matrix((np.ones(k), (np.zeros(k, dtype=int), np.arange(k))), shape=(1, n + 2*k))
    solution = linprog(objective, A_ub=constraints, b_ub=rhs, A_eq=equality, b_eq=[1.],
                       bounds=[(0, 1)] * k + [(0, None)] * (n+k), method='highs-ipm',
                       options={'time_limit': 180})
    if not solution.success:
        raise RuntimeError(f'MAE linear program failed: {solution.message}')
    weights = np.maximum(solution.x[:k], 0)
    weights /= weights.sum()
    return weights, {'optimizer': 'HiGHS interior-point linear program', 'penalty': penalty,
                     'objective': float(solution.fun), 'optimality_status': solution.message,
                     'calibration_mae': float(np.mean(np.abs(matrix @ weights - y)))}
