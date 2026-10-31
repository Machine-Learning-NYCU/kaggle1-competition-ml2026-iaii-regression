"""Controlled v2 extensions; no external data or evaluation-label features."""
import argparse
import json
import time

import joblib
import numpy as np
from catboost import CatBoostRegressor, Pool

from train_robust import ROOT, OUT as V2
from train_spatial import OUT as SPATIAL, base_data
from train_mae import checksum, dump, save_model
from train import scores

OUT = ROOT / 'artifacts/hedonic_v1'
SPECS = {
    'net_log': dict(area='net', spatial=False, depth=8, loss='RMSE'),
    'deep_log': dict(area='total', spatial=False, depth=10, loss='RMSE'),
    'context_log': dict(area='total', spatial=True, depth=8, loss='RMSE'),
    'net_huber': dict(area='net', spatial=False, depth=8, loss='Huber:delta=0.10'),
}


def freeze_config():
    OUT.mkdir(exist_ok=True, parents=True)
    config = dict(specs=SPECS, learning_rate=.045, max_iterations=5000,
                  weight='sqrt(price) for RMSE, price for log-Huber; bounded; no recency decay',
                  source_sha256={f: checksum(ROOT/f) for f in
                      ['train_hedonic.py', 'spatial_prior.py', 'train_spatial.py',
                       'robust_features.py', 'train_robust.py', 'features.py']},
                  parent_selection_sha256=checksum(V2/'selection.json'),
                  spatial_config_sha256=checksum(SPATIAL/'config.json'),
                  early_stop='First four months of 2022 development only',
                  selection_windows=['stress2021', 'development'],
                  confirmation_used_for_selection=False)
    path = OUT/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Experiment changed; preserve versioned configuration')
    dump(path, config)


def inputs(fold, name):
    spec = SPECS[name]
    if spec['area'] == 'net' or spec['spatial']:
        d = joblib.load(SPATIAL/fold/f"{spec['area']}.joblib")
    else:
        d = base_data(fold)
    if spec['spatial']:
        cols = list(d['x'])
    elif spec['area'] == 'net':
        # Column order of the fixed v2 representation, plus physical effective area.
        cols = [c for c in d['x'] if not c.startswith('spatial_')]
    else:
        cols = list(d['x'])
    return d, cols


def weights(y, spec):
    return np.clip(np.sqrt(y/1e7), .2, 6) if spec['loss']=='RMSE' else np.clip(y/1e7, .2, 20)


def fit(name, d, cols, iterations=None):
    spec = SPECS[name]
    cats = [c for c in d['cats'] if c in cols]
    model = CatBoostRegressor(iterations=iterations or 5000,
        depth=spec['depth'], learning_rate=.045, loss_function=spec['loss'],
        l2_leaf_reg=5, random_strength=.5, bootstrap_type='Bayesian',
        bagging_temperature=.5, max_ctr_complexity=2, one_hot_max_size=10,
        has_time=False, task_type='GPU', devices='0', thread_count=16,
        random_seed=3017, verbose=500, allow_writing_files=False)
    kw = {}
    if iterations is None:
        mask = d['tune']
        kw = dict(eval_set=Pool(d['vx'].loc[mask, cols],
                    np.log(d['vy'][mask]/d['vbase'][mask]),
                    cat_features=cats, weight=weights(d['vy'][mask], spec)),
                  early_stopping_rounds=300)
    model.fit(Pool(d['x'][cols], np.log(d['y']/d['base']),
                   cat_features=cats, weight=weights(d['y'], spec)), **kw)
    return dict(model=model, columns=cols, name=name, spec=spec, iterations=model.tree_count_)


def predict(model, x, base):
    z = model['model'].predict(x[model['columns']])
    return np.maximum(base*np.exp(np.clip(z, -10, 20)), 1)


def train(fold, names):
    freeze_config()
    path = OUT/fold
    path.mkdir(exist_ok=True)
    for name in names:
        if (path/f'{name}.json').exists():
            continue
        d, cols = inputs(fold, name)
        iterations = None if fold=='development' else json.loads(
            (OUT/'development'/f'{name}.json').read_text())['iterations']
        start = time.time()
        model_path = path/f'{name}.joblib'
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = fit(name, d, cols, iterations)
            save_model(model_path, model)
        p = predict(model, d['vx'], d['vbase'])
        np.save(path/f'{name}_predictions.npy', p)
        record = dict(iterations=model['iterations'], seconds=time.time()-start)
        if d['vy'] is not None:
            record['all'] = scores(d['vy'], p)
            for key, mask in [('seen',d['seen']), ('unseen',~d['seen']),
                              ('early',d['tune']), ('later',~d['tune'])]:
                record[key] = scores(d['vy'][mask], p[mask])
        dump(path/f'{name}.json', record)
        print('DONE', fold, name, record, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default='development')
    parser.add_argument('--models', nargs='+', choices=list(SPECS), default=list(SPECS))
    args = parser.parse_args()
    train(args.fold, args.models)
