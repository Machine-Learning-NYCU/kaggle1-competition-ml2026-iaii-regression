"""Frozen non-price geography ablation, with time and whole-project validation."""
import argparse
import json
import shutil
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import CategoryEncoder
from geography_features import ROOT, OUT
from train_generalization import (BASE_SHA, SEED, SECOND_SEED, WINDOWS,
    inputs as original_inputs, baseline, cold_mask, metrics, grouped_improvement, project_keys)
from train_mae import checksum, dump, save_model, export_submission

SPECS = {'cat_control':('cat', False), 'cat_geo':('cat', True),
         'lgb_control':('lgb', False), 'lgb_geo':('lgb', True)}
CANDIDATES = ['cat_geo', 'lgb_geo']
DROP = ['project', 'canonical_project', 'project_building_type', 'address_base',
        'tower', 'house_number', 'parcel_number']
ROUNDS = {'cat':3000, 'lgb':2000}


def freeze_protocol():
    sources = ['train_geography.py', 'geography_features.py', 'test_geography.py',
        'train_generalization.py', 'train_shift.py', 'features.py', 'train_mae.py']
    files = [ROOT/'submission1017_v6.csv', ROOT/'artifacts/external_data_policy.json',
        OUT/'features.joblib', OUT/'geocoder.joblib', OUT/'geocoder_audit.json']
    files += [ROOT/f'artifacts/shift_v1/{f}/data.joblib' for f in WINDOWS+['confirmation', 'final']]
    files += [ROOT/f'artifacts/generalization_v1/{f}/identity_log_{SEED}_predictions.npy' for f in WINDOWS]
    files += sorted((OUT/'sources').glob('*.csv'))
    config = dict(specs=SPECS, rounds=ROUNDS, seeds=[SEED, SECOND_SEED],
        cat_params=dict(depth=8, learning_rate=.045, l2_leaf_reg=10, loss='RMSE'),
        lgb_params=dict(num_leaves=63, min_child_samples=80, learning_rate=.04,
            reg_lambda=10, reg_alpha=.05, objective='regression_l1'),
        blend_fractions=[.25, .5], trend_damping=.5, early_stopping=False,
        primary_control='Reuse frozen generalization_v1 identity_log predictions for cat_control; matching parameters and seed',
        external_features='Only 2020 NPA road coordinates and reference density; no external house-price samples or per-unit offers',
        scope='Canonical and original-name unseen projects, excluding relocation housing',
        selection='Both years cold MAE improves >=0.5%; late cold never worsens; must beat matching same-alpha control in both years. Worst-year ratio; prefer .25 if within 0.5% of best.',
        seed_check='Freeze candidate first; two-seed mean passes original gates, every seed improves cold MAE in both years; matching second-seed control also fitted; no runner-up.',
        confirmation='Same frozen candidate and alpha; >=0.5% cold improvement, late non-regression; project bootstrap lower95>0 versus baseline and versus matching control.',
        history_reused=True, public_scores_used_for_selection=False,
        source_sha256={s:checksum(ROOT/s) for s in sources},
        input_sha256={str(p.relative_to(ROOT)):checksum(p) for p in files})
    config = json.loads(json.dumps(config))
    path = OUT/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Frozen experiment changed; use a new version')
    assert checksum(ROOT/'submission1017_v6.csv') == BASE_SHA
    dump(path, config)
    (OUT/'source').mkdir(exist_ok=True)
    for name in sources:
        shutil.copy2(ROOT/name, OUT/'source'/name)


def inputs(fold):
    d = original_inputs(fold)
    geo = joblib.load(OUT/'features.joblib')
    x = d['x'].copy(); vx = d['vx'].copy()
    # This constant matches the frozen CatBoost control's feature set exactly.
    x['identity_masked'] = np.float32(0); vx['identity_masked'] = np.float32(0)
    xx = geo['train'].iloc[d['fit_ix']].reset_index(drop=True)
    vv = geo['test'] if fold=='final' else geo['train'].iloc[d['valid_ix']].reset_index(drop=True)
    return dict(d, x=pd.concat([x, xx], axis=1), vx=pd.concat([vx, vv], axis=1))


def pred_path(fold, name, seed):
    return OUT/fold/f'{name}_{seed}_predictions.npy'


def model_path(fold, name, seed):
    return OUT/fold/f'{name}_{seed}.joblib'


def predict(model, x, base):
    xx = x[model['columns']]
    if model['encoder'] is not None:
        xx = model['encoder'].transform(xx)
    z = model['model'].predict(xx)
    p = np.exp(np.clip(z, -10, 20))*base if model['kind']=='cat' else z*base*1e5
    return np.maximum(p*np.exp(-.5*x.time_index.to_numpy(float)), 1.)


def fit_fold(fold, names, seed=SEED):
    freeze_protocol()
    d = inputs(fold); cold = cold_mask(d); old = baseline(fold)
    directory = OUT/fold; directory.mkdir(exist_ok=True)
    for name in names:
        result = directory/f'{name}_{seed}.json'
        if result.exists():
            continue
        start = time.time(); kind, geo = SPECS[name]
        reuse = name=='cat_control' and seed==SEED and fold in WINDOWS
        if reuse:
            path = ROOT/f'artifacts/generalization_v1/{fold}/identity_log_{SEED}_predictions.npy'
            p = np.load(path)
        else:
            path = model_path(fold, name, seed)
            if path.exists():
                model = joblib.load(path)
            else:
                cols = [c for c in d['x'] if (geo or not c.startswith('ext_')) and (kind=='cat' or c not in DROP)]
                cats = [c for c in d['cats'] if c in cols]
                x = d['x'][cols]
                if kind=='cat':
                    encoder = None
                    estimator = CatBoostRegressor(iterations=ROUNDS[kind], depth=8, learning_rate=.045,
                        loss_function='RMSE', l2_leaf_reg=10, random_strength=.5,
                        bootstrap_type='Bayesian', bagging_temperature=.5, max_ctr_complexity=2,
                        one_hot_max_size=10, has_time=False, task_type='GPU', devices='0', thread_count=12,
                        random_seed=seed, verbose=500, allow_writing_files=False)
                    estimator.fit(Pool(x, np.log(d['y']/d['base']), cat_features=cats,
                        weight=np.clip(np.sqrt(d['y']/1e7), .2, 6)))
                else:
                    encoder = CategoryEncoder().fit(x, cats)
                    estimator = lgb.LGBMRegressor(n_estimators=ROUNDS[kind], learning_rate=.04,
                        num_leaves=63, min_child_samples=80, reg_lambda=10, reg_alpha=.05,
                        colsample_bytree=.85, subsample=.85, subsample_freq=1,
                        cat_smooth=20, cat_l2=10, max_cat_threshold=64,
                        objective='regression_l1', n_jobs=12, random_state=seed, verbosity=-1,
                        deterministic=True, force_col_wise=True)
                    estimator.fit(encoder.transform(x), d['y']/d['base']/1e5,
                        sample_weight=d['base']/100)
                model = dict(model=estimator, encoder=encoder, kind=kind, name=name, seed=seed,
                    columns=cols, iterations=ROUNDS[kind], external_geography=geo)
                save_model(path, model)
            p = predict(model, d['vx'], d['vbase'])
        np.save(pred_path(fold, name, seed), p)
        record = dict(seconds=time.time()-start, fit_rows=len(d['y']), cold_rows=int(cold.sum()),
            control_predictions_reused=reuse, iterations=ROUNDS[kind])
        if d['vy'] is not None:
            record['standalone'] = metrics(d['vy'], old, p, cold, ~d['tune'])
            record['blends'] = {str(alpha):metrics(d['vy'], old,
                np.where(cold, (1-alpha)*old+alpha*p, old), cold, ~d['tune']) for alpha in [.25, .5]}
        dump(result, record)
        print('DONE', fold, name, seed, json.dumps(record, ensure_ascii=False), flush=True)


def choose(records):
    eligible = [r for r in records if max(r['ratios'])<=.995 and max(r['late_ratios'])<=1 and max(r['control_ratios'])<1]
    if not eligible:
        return None
    best = min(r['worst_ratio'] for r in eligible)
    return min([r for r in eligible if r['worst_ratio']<=best*1.005], key=lambda r:(r['alpha'], r['worst_ratio'], r['name']))


def freeze_selection():
    path = OUT/'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    records = []
    for name in CANDIDATES:
        for alpha in [.25, .5]:
            yearly = []; control_ratios = []
            for fold in WINDOWS:
                m = json.loads((OUT/fold/f'{name}_{SEED}.json').read_text())['blends'][str(alpha)]
                control = json.loads((OUT/fold/f'{name.replace("_geo", "_control")}_{SEED}.json').read_text())['blends'][str(alpha)]
                yearly.append(m)
                control_ratios.append(m['cold']['candidate_mae']/control['cold']['candidate_mae'])
            ratios = [m['cold']['ratio'] for m in yearly]
            records.append(dict(name=name, alpha=alpha, ratios=ratios,
                late_ratios=[m['cold_late']['ratio'] for m in yearly], control_ratios=control_ratios,
                worst_ratio=max(ratios), yearly_metrics=yearly))
    selected = choose(records)
    result = dict(selected=selected, grid=records,
        status='primary_qualified' if selected else 'rejected_no_cross_year_gain',
        config_sha256=checksum(OUT/'config.json'), stage='Frozen before second seed and confirmation')
    dump(path, result); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def seed_check():
    selection = freeze_selection()
    if selection['selected'] is None:
        return False
    choice = selection['selected']; name = choice['name']; alpha = choice['alpha']
    control = name.replace('_geo', '_control')
    reports = {}; accepted = True
    for fold in WINDOWS:
        fit_fold(fold, [control, name], SECOND_SEED)
        d = inputs(fold); old = baseline(fold); cold = cold_mask(d)
        seeds = [np.load(pred_path(fold, name, s)) for s in [SEED, SECOND_SEED]]
        controls = [np.load(pred_path(fold, control, s)) for s in [SEED, SECOND_SEED]]
        p = np.where(cold, (1-alpha)*old+alpha*np.mean(seeds, axis=0), old)
        cp = np.where(cold, (1-alpha)*old+alpha*np.mean(controls, axis=0), old)
        m = metrics(d['vy'], old, p, cold, ~d['tune'])
        control_mae = np.mean(abs(d['vy'][cold]-cp[cold]))
        individual = [float(np.mean(abs(d['vy'][cold]-((1-alpha)*old+alpha*q)[cold]))/m['cold']['baseline_mae']) for q in seeds]
        m['individual_seed_cold_ratios'] = individual
        m['matching_control_cold_ratio'] = float(m['cold']['candidate_mae']/control_mae)
        reports[fold] = m
        accepted &= (m['cold']['ratio']<=.995 and m['cold_late']['ratio']<=1
            and max(individual)<1 and m['matching_control_cold_ratio']<1)
    report = dict(accepted=bool(accepted), folds=reports, selection_sha256=checksum(OUT/'selection.json'))
    dump(OUT/'seed_check.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return bool(accepted)


def confirm_export():
    selection = freeze_selection()
    if selection['selected'] is None:
        return
    seed = OUT/'seed_check.json'
    if not seed.exists() or not json.loads(seed.read_text())['accepted']:
        return
    choice = selection['selected']; name = choice['name']; alpha = choice['alpha']
    control = name.replace('_geo', '_control')
    for s in [SEED, SECOND_SEED]:
        fit_fold('confirmation', [control, name], s)
    d = inputs('confirmation'); old = baseline('confirmation'); cold = cold_mask(d)
    q = np.mean([np.load(pred_path('confirmation', name, s)) for s in [SEED, SECOND_SEED]], axis=0)
    cq = np.mean([np.load(pred_path('confirmation', control, s)) for s in [SEED, SECOND_SEED]], axis=0)
    p = np.where(cold, (1-alpha)*old+alpha*q, old)
    cp = np.where(cold, (1-alpha)*old+alpha*cq, old)
    m = metrics(d['vy'], old, p, cold, ~d['tune'])
    groups = project_keys(d['vx']).to_numpy()[cold]
    gain = grouped_improvement(d['vy'][cold], p[cold], old[cold], groups)
    control_gain = grouped_improvement(d['vy'][cold], p[cold], cp[cold], groups)
    accepted = m['cold']['ratio']<=.995 and m['cold_late']['ratio']<=1 and gain[0]>0 and control_gain[0]>0
    report = dict(accepted=bool(accepted), metrics=m, cold_cluster_gain_95=gain,
        matching_control_cluster_gain_95=control_gain, history_reused=True,
        frozen_choice_unchanged=True, public_score_unknown=True)
    dump(OUT/'confirmation.json', report)
    np.savez_compressed(OUT/'confirmation/predictions.npz', y=d['vy'], baseline=old,
        candidate=p, matching_control=cp, cold=cold)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not accepted:
        return
    for s in [SEED, SECOND_SEED]:
        fit_fold('final', [name], s)
    d = inputs('final'); old = baseline('final'); cold = cold_mask(d)
    q = np.mean([np.load(pred_path('final', name, s)) for s in [SEED, SECOND_SEED]], axis=0)
    p = np.where(cold, (1-alpha)*old+alpha*q, old)
    raw = pd.read_parquet(ROOT/'data/test.parquet')
    candidate = OUT/'submission1017_v7.csv'
    frame = export_submission(raw, p, candidate)
    np.testing.assert_array_equal(frame.loc[~cold, '總價元'], old[~cold])
    destination = ROOT/candidate.name
    if destination.exists() and checksum(destination) != checksum(candidate):
        raise ValueError('Existing v7 differs; preserving it and leaving candidate in artifact directory')
    shutil.copy2(candidate, destination)
    dump(OUT/'verification.json', dict(rows=len(frame), eligible_rows=int(cold.sum()),
        changed_rows=int((frame['總價元'].to_numpy()!=old).sum()),
        protected_rows_unchanged=True, public_mae=None, submission_sha256=checksum(candidate),
        base_v6_sha256=BASE_SHA, new_branch_uses_external_prices=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'freeze', 'seed_check', 'confirm_export', 'all'])
    parser.add_argument('--fold', default='development')
    parser.add_argument('--models', nargs='+', default=list(SPECS))
    args = parser.parse_args()
    if args.stage=='train':
        fit_fold(args.fold, args.models)
    elif args.stage=='freeze':
        freeze_protocol(); freeze_selection()
    elif args.stage=='seed_check':
        seed_check()
    elif args.stage=='confirm_export':
        confirm_export()
    else:
        for fold in ['development', 'stress2021']:
            fit_fold(fold, list(SPECS))
        if seed_check():
            confirm_export()


if __name__ == '__main__':
    main()
