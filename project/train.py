"""Reproducible temporal backtest, frozen blending, full refit, and submission."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize

from features import CategoryEncoder, ID, TARGET, duplicate_groups, make_features, roc_date
from market import CausalMarket

ROOT = Path(__file__).resolve().parent
SPECS = {
    'cat_total': {'kind': 'cat', 'target': 'log_total', 'depth': 8},
    'cat_market': {'kind': 'cat', 'target': 'market_residual', 'depth': 8},
    'lgb_market': {'kind': 'lgb', 'target': 'market_residual', 'leaves': 63},
    'lgb_price': {'kind': 'lgb', 'target': 'price', 'leaves': 31},
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def scores(y, prediction):
    y, p = np.asarray(y, float), np.maximum(np.asarray(prediction, float), 1)
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite predictions or labels')
    return {'rmsle': float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
            'rmse': float(np.sqrt(np.mean((p - y) ** 2))),
            'mae': float(np.mean(np.abs(p - y))),
            'mape': float(np.mean(np.abs(p - y) / y)),
            'within_10pct': float(np.mean(np.abs(p - y) / y <= .1))}


def target_transform(y, x, offset, target):
    if target == 'price':
        return y / 1e7
    if target == 'market_residual':
        return np.log(y) - offset - x['market_project'].to_numpy()
    return np.log(y)


def inverse(prediction, x, offset, target):
    if target == 'price':
        return np.maximum(prediction * 1e7, 1)
    if target == 'market_residual':
        prediction = prediction + offset + x['market_project'].to_numpy()
    return np.exp(np.clip(prediction, 0, 30))


def fit_model(spec, x, y, offset, cats, args, validation=None, iterations=None, seed=2026):
    transformed_y = target_transform(y, x, offset, spec['target'])
    encoder = None
    if spec['kind'] == 'cat':
        model = CatBoostRegressor(
            iterations=iterations or args.iterations, depth=spec['depth'], learning_rate=.055,
            loss_function='RMSE', l2_leaf_reg=12, random_strength=.5,
            bootstrap_type='Bayesian', bagging_temperature=.5,
            max_ctr_complexity=1, one_hot_max_size=10, has_time=True,
            task_type=args.device, devices='0' if args.device == 'GPU' else None,
            thread_count=args.threads, random_seed=seed, verbose=250,
            allow_writing_files=False)
        params = {'cat_features': cats}
        if validation is not None:
            vx, vy, vo = validation
            params.update(eval_set=(vx, target_transform(vy, vx, vo, spec['target'])), early_stopping_rounds=150)
        model.fit(x, transformed_y, **params)
        best = model.tree_count_
    else:
        encoder = CategoryEncoder().fit(x, cats)
        model = lgb.LGBMRegressor(
            n_estimators=iterations or args.iterations, learning_rate=.045,
            num_leaves=spec['leaves'], max_depth=-1, min_child_samples=80,
            reg_lambda=20, reg_alpha=.1, colsample_bytree=.85, subsample=.85,
            subsample_freq=1, cat_smooth=30, cat_l2=20, max_cat_threshold=64,
            objective='regression', n_jobs=args.threads, random_state=seed,
            verbosity=-1, deterministic=True, force_col_wise=True)
        params = {}
        if validation is not None:
            vx, vy, vo = validation
            params.update(eval_set=[(encoder.transform(vx), target_transform(vy, vx, vo, spec['target']))],
                          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
        model.fit(encoder.transform(x), transformed_y, **params)
        best = model.best_iteration_ or model.n_estimators_
    return {'model': model, 'encoder': encoder, 'spec': spec, 'iterations': int(best)}


def predict(bundle, x, offset):
    xx = bundle['encoder'].transform(x) if bundle['encoder'] is not None else x
    p = bundle['model'].predict(xx)
    return inverse(p, x, offset, bundle['spec']['target'])


def blend(matrix, weights, metric):
    if metric == 'rmsle':
        return np.expm1(np.log1p(matrix) @ weights)
    return matrix @ weights


def learn_blend(matrix, y, metric):
    n = matrix.shape[1]
    # Three log branches form the prior. Raw-price head can enter if calibration supports it.
    prior = np.array([.25, .4, .35, 0.])[:n]
    prior /= prior.sum()
    scale = scores(y, blend(matrix, prior, metric))[metric] ** 2
    def objective(w):
        loss = scores(y, blend(matrix, w, metric))[metric] ** 2 / max(scale, 1e-12)
        return loss + .025 * np.sum((w - prior) ** 2)
    fit = minimize(objective, prior, method='SLSQP', bounds=[(0, 1)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                   options={'maxiter': 300, 'ftol': 1e-10})
    if not fit.success:
        raise RuntimeError(f'Blend optimizer failed: {fit.message}')
    weights = np.maximum(fit.x, 0)
    return weights / weights.sum()


def subset(x, indices):
    return x.iloc[indices].reset_index(drop=True)


def bootstrap_interval(y, pred, groups, metric, seed=2026):
    # Resample buildings/projects, not correlated apartment rows.
    errors = (np.log1p(pred) - np.log1p(y)) ** 2 if metric == 'rmsle' else (pred - y) ** 2
    stats = pd.DataFrame({'g': groups, 'error': errors}).groupby('g').error.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(400):
        ix = rng.integers(0, len(stats), len(stats))
        values.append(np.sqrt(stats['sum'].to_numpy()[ix].sum() / stats['count'].to_numpy()[ix].sum()))
    return list(np.quantile(values, [.025, .975]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['develop', 'final', 'evaluate', 'all'], default='all')
    parser.add_argument('--out', default='artifacts/run_v2')
    parser.add_argument('--iterations', type=int, default=2200)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    config = {'iterations': args.iterations, 'threads': args.threads, 'device': args.device,
              'specs': SPECS, 'fit_end': '2023-09-30', 'tune_end': '2023-12-31',
              'calibration_end': '2024-04-30', 'train_end': '2024-08-10',
              'code_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                              for f in ['features.py', 'market.py', 'train.py']}}
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Run configuration/code changed: choose a new --out directory')
    dump(out / 'config.json', config)
    raw = pd.read_parquet(ROOT / 'data/train.parquet')
    y0 = pd.to_numeric(raw[TARGET], errors='coerce')
    dates = roc_date(raw['交易年月日'])
    keep = y0.notna() & (y0 > 0) & dates.between('2000-01-01', '2024-08-10')
    excluded = int((~keep).sum())
    raw = raw.loc[keep].copy()
    raw['_date'] = dates.loc[keep]
    raw = raw.sort_values('_date', kind='stable').drop(columns=['_date']).reset_index(drop=True)
    y = pd.to_numeric(raw[TARGET]).to_numpy(float)
    dates = roc_date(raw['交易年月日'])
    cache = out / 'features.joblib'
    if cache.exists():
        x, cats, groups, offset = joblib.load(cache)
    else:
        x, cats, groups, offset = make_features(raw)
        joblib.dump((x, cats, groups, offset), cache, compress=3)
    fit_ix = np.where(dates <= '2023-09-30')[0]
    tune_ix = np.where((dates > '2023-09-30') & (dates <= '2023-12-31'))[0]
    pre_ix = np.where(dates <= '2023-12-31')[0]
    cal_ix = np.where((dates > '2023-12-31') & (dates <= '2024-04-30'))[0]
    audit_ix = np.where(dates > '2024-04-30')[0]
    dup = duplicate_groups(raw)
    partitions = {'fit': fit_ix, 'tune': tune_ix, 'calibration': cal_ix, 'temporal_audit': audit_ix}
    for a, ia in partitions.items():
        for b, ib in partitions.items():
            if a < b and np.intersect1d(dup[ia], dup[ib]).size:
                raise ValueError(f'Duplicate feature rows cross {a}/{b}')
    dump(out / 'split_manifest.json', {'excluded_rows': excluded, 'rows': len(raw), 'features': list(x),
        'partitions': {k: {'rows': len(v), 'min_date': str(dates.iloc[v].min()), 'max_date': str(dates.iloc[v].max())}
                       for k, v in partitions.items()}, 'duplicate_groups_disjoint': True})
    print('SPLITS', {k: len(v) for k, v in partitions.items()}, 'excluded', excluded, flush=True)
    names = list(SPECS)

    if args.stage in ['develop', 'all'] and not (out / 'frozen_blend.json').exists():
        market_fit = CausalMarket().fit(subset(x, fit_ix), np.log(y[fit_ix]) - offset[fit_ix])
        xf = market_fit.transform(subset(x, fit_ix))
        xt = market_fit.transform(subset(x, tune_ix))
        market_dev = CausalMarket().fit(subset(x, pre_ix), np.log(y[pre_ix]) - offset[pre_ix])
        joblib.dump(market_dev, out / 'market_dev.joblib', compress=3)
        xp = market_dev.transform(subset(x, pre_ix))
        xc = market_dev.transform(subset(x, cal_ix))
        matrix, model_records = [], {}
        for name, spec in SPECS.items():
            path = out / f'dev_{name}.joblib'
            start = time.time()
            if path.exists():
                model = joblib.load(path)
            else:
                print('TUNE', name, flush=True)
                tuned = fit_model(spec, xf, y[fit_ix], offset[fit_ix], cats, args,
                                  validation=(xt, y[tune_ix], offset[tune_ix]))
                print('REFIT DEVELOPMENT', name, tuned['iterations'], flush=True)
                model = fit_model(spec, xp, y[pre_ix], offset[pre_ix], cats, args, iterations=tuned['iterations'])
                joblib.dump(model, path, compress=3)
            prediction = predict(model, xc, offset[cal_ix])
            matrix.append(prediction)
            model_records[name] = {'iterations': model['iterations'], 'calibration': scores(y[cal_ix], prediction)}
            print('CALIBRATION', name, model_records[name], 'seconds', round(time.time() - start), flush=True)
        matrix = np.column_stack(matrix)
        weights = {metric: learn_blend(matrix, y[cal_ix], metric).tolist() for metric in ['rmsle', 'rmse']}
        frozen = {'models': names, 'records': model_records, 'weights': weights,
                  'metric_status': 'Official scoring rule unconfirmed; separate RMSLE/RMSE candidates.',
                  'calibration_is_selection_data': True,
                  'calibration_blends': {k: scores(y[cal_ix], blend(matrix, w, k)) for k, w in weights.items()},
                  'frozen_before_holdout_evaluation': True}
        dump(out / 'frozen_blend.json', frozen)
        np.savez_compressed(out / 'calibration_predictions.npz', predictions=matrix, y=y[cal_ix], indices=cal_ix)
        print('FROZEN', frozen, flush=True)

    if args.stage in ['final', 'all']:
        frozen = json.loads((out / 'frozen_blend.json').read_text())
        market = CausalMarket().fit(x, np.log(y) - offset)
        joblib.dump(market, out / 'market_final.joblib', compress=3)
        xx = market.transform(x)
        test = pd.read_parquet(ROOT / 'data/test.parquet')
        tx, _, tg, to = make_features(test)
        tx = market.transform(tx)
        predictions = []
        for name, spec in SPECS.items():
            path = out / f'final_{name}.joblib'
            if path.exists():
                model = joblib.load(path)
            else:
                print('FINAL FIT', name, flush=True)
                model = fit_model(spec, xx, y, offset, cats, args,
                                  iterations=frozen['records'][name]['iterations'])
                joblib.dump(model, path, compress=3)
            predictions.append(predict(model, tx, to))
        matrix = np.column_stack(predictions)
        sample = pd.read_csv(ROOT / 'data/test-reindex-test-sample-submission-v2.csv')
        if list(sample) != [ID, TARGET] or len(sample) != len(test) or sample[ID].astype(str).tolist() != test[ID].astype(str).tolist():
            raise ValueError('Submission template ID/order/schema mismatch')
        for metric, weights in frozen['weights'].items():
            prediction = blend(matrix, weights, metric)
            if not (np.isfinite(prediction).all() and (prediction > 0).all()):
                raise ValueError('Submission values must be finite and positive')
            submission = sample.copy()
            submission[TARGET] = prediction
            submission.to_csv(out / f'submission_{metric}.csv', index=False, encoding='utf-8-sig')
        np.savez_compressed(out / 'test_base_predictions.npz', predictions=matrix)
        dump(out / 'submission_manifest.json', {'rows': len(test), 'columns': list(sample),
            'test_new_group_fraction': float(np.mean(~pd.Series(tg).isin(groups).to_numpy())),
            'official_valid_used_for_training': False,
            'files': {f'submission_{k}.csv': hashlib.sha256((out / f'submission_{k}.csv').read_bytes()).hexdigest()
                      for k in frozen['weights']}})

    if args.stage in ['evaluate', 'all']:
        if (out / 'evaluation.json').exists():
            print('Locked evaluation already exists; reporting without retuning.', flush=True)
            print((out / 'evaluation.json').read_text(), flush=True)
            return
        frozen = json.loads((out / 'frozen_blend.json').read_text())
        report = {'selection_frozen': True, 'evaluations': {}}
        # Temporal audit is untouched until the architecture, rounds and weights are fixed.
        valid = pd.read_parquet(ROOT / 'data/valid.parquet')
        vy = pd.to_numeric(valid[TARGET], errors='raise').to_numpy(float)
        vx, _, vg, vo = make_features(valid)
        datasets = [('future_2024_05_to_08', subset(x, audit_ix), y[audit_ix], groups[audit_ix], offset[audit_ix], 'dev', groups[pre_ix]),
                    ('official_valid', vx, vy, vg, vo, 'final', groups)]
        for label, ex, ey, eg, eo, stage, fitting_groups in datasets:
            market = joblib.load(out / f'market_{stage}.joblib')
            ex = market.transform(ex)
            matrix = np.column_stack([predict(joblib.load(out / f'{stage}_{n}.joblib'), ex, eo) for n in names])
            seen = pd.Series(eg).isin(fitting_groups).to_numpy()
            # Official validation can contain exact duplicates of supplied training transactions.
            clean = ~np.isin(duplicate_groups(valid), dup) if label == 'official_valid' else np.ones(len(ey), bool)
            entry = {'rows': len(ey), 'unseen_group_fraction': float(np.mean(~seen)),
                     'exact_duplicate_train_rows': int((~clean).sum()),
                     'base_models': {n: scores(ey, matrix[:, i]) for i, n in enumerate(names)}, 'blends': {}}
            for metric, w in frozen['weights'].items():
                p = blend(matrix, w, metric)
                entry['blends'][metric] = {'all': scores(ey, p),
                    'cluster_bootstrap_95pct': bootstrap_interval(ey, p, eg, metric),
                    'slices': {k: {'n': int(mask.sum()), **scores(ey[mask], p[mask])}
                              for k, mask in {'seen_group': seen, 'unseen_group': ~seen,
                                              'no_exact_train_duplicate': clean}.items() if mask.any()}}
            # A simple median baseline supplies a real, falsifiable improvement reference.
            baseline_y = y[pre_ix] if stage == 'dev' else y
            entry['median_baseline'] = scores(ey, np.full(len(ey), np.median(baseline_y)))
            report['evaluations'][label] = entry
            np.savez_compressed(out / f'{label}_predictions.npz', y=ey, predictions=matrix, groups=eg)
            print('EVALUATION', label, entry, flush=True)
        dump(out / 'evaluation.json', report)


if __name__ == '__main__':
    main()
