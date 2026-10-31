"""MAE-aligned training, time-separated calibration and explicit reused evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from features import ID, TARGET, make_features, roc_date, duplicate_groups
from market import CausalMarket
from mae_models import SPECS, fit_model, predict, learn_mae_blend
from train import scores, subset, blend as legacy_blend

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'artifacts/run_v2'


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def save_model(path, model):
    temporary = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_prediction(name, expected_y=None):
    frozen = json.loads((LEGACY / 'frozen_blend.json').read_text())
    archive = np.load(LEGACY / name, allow_pickle=False)
    if expected_y is not None:
        np.testing.assert_array_equal(archive['y'], expected_y)
    return legacy_blend(archive['predictions'], frozen['weights']['rmsle'], 'rmsle')


def clustered_comparison(y, new, old, groups):
    frame = pd.DataFrame({'group': groups, 'new': np.abs(y-new), 'old': np.abs(y-old)})
    stats = frame.groupby('group').agg(new=('new', 'sum'), old=('old', 'sum'), n=('new', 'size'))
    values = stats.to_numpy()
    rng = np.random.default_rng(2026)
    sampled = []
    for _ in range(500):
        sums = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        sampled.append([sums[0]/sums[2], (sums[1]-sums[0])/sums[2]])
    sampled = np.asarray(sampled)
    return {'mae_95pct': np.quantile(sampled[:, 0], [.025, .975]).tolist(),
            'improvement_over_legacy_95pct': np.quantile(sampled[:, 1], [.025, .975]).tolist()}


def export_submission(test, prediction, output):
    template = pd.read_csv(ROOT / 'data/test-reindex-test-sample-submission-v2.csv')
    if list(template) != [ID, TARGET] or template[ID].astype(str).tolist() != test[ID].astype(str).tolist():
        raise ValueError('Template schema or ID order mismatch')
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError('Invalid predictions')
    template[TARGET] = np.rint(prediction).astype(np.int64)
    template.to_csv(output, index=False, encoding='utf-8')
    return template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['develop', 'final', 'evaluate', 'all'], default='all')
    parser.add_argument('--out', default='artifacts/run_mae_v1')
    parser.add_argument('--iterations', type=int, default=3500)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    source = ['features.py', 'market.py', 'mae_models.py', 'train_mae.py', 'train.py']
    config = {'official_metric': 'MAE in original NTD', 'specs': SPECS,
              'iterations': args.iterations, 'threads': args.threads, 'device': args.device,
              'blend_penalty': .02, 'legacy_model': 'run_v2 frozen RMSLE ensemble',
              'evaluation_status': 'Historical evaluation reused; not a new untouched test set.',
              'source_sha256': {name: checksum(ROOT/name) for name in source},
              'data_sha256': {name: checksum(ROOT/'data'/f'{name}.parquet') for name in ['train', 'valid', 'test']},
              'legacy_sha256': {name: checksum(LEGACY/name) for name in
                               ['frozen_blend.json', 'calibration_predictions.npz', 'test_base_predictions.npz',
                                'future_2024_05_to_08_predictions.npz', 'official_valid_predictions.npz']}}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Source/data/config changed: use a new --out')
    dump(out/'config.json', config)
    source_dir = out/'source'
    source_dir.mkdir(exist_ok=True)
    for name in source:
        shutil.copy2(ROOT/name, source_dir/name)
    raw = pd.read_parquet(ROOT/'data/train.parquet')
    y0 = pd.to_numeric(raw[TARGET], errors='coerce')
    dates = roc_date(raw['交易年月日'])
    keep = y0.notna() & (y0 > 0) & dates.between('2000-01-01', '2024-08-10')
    raw = raw.loc[keep].assign(_date=dates.loc[keep]).sort_values('_date', kind='stable').drop(columns=['_date']).reset_index(drop=True)
    y = pd.to_numeric(raw[TARGET]).to_numpy(float)
    dates = roc_date(raw['交易年月日'])
    cache = out/'features.joblib'
    if cache.exists():
        x, cats, groups, offset = joblib.load(cache)
    else:
        x, cats, groups, offset = make_features(raw)
        save_model(cache, (x, cats, groups, offset))
    fit = np.where(dates <= '2023-09-30')[0]
    tune = np.where((dates > '2023-09-30') & (dates <= '2023-12-31'))[0]
    pre = np.where(dates <= '2023-12-31')[0]
    cal = np.where((dates > '2023-12-31') & (dates <= '2024-04-30'))[0]
    audit = np.where(dates > '2024-04-30')[0]
    partitions = {'fit': fit, 'early_stopping': tune, 'calibration': cal, 'reused_temporal_evaluation': audit}
    fingerprints = duplicate_groups(raw)
    for a, ia in partitions.items():
        for b, ib in partitions.items():
            if a < b and np.intersect1d(fingerprints[ia], fingerprints[ib]).size:
                raise ValueError('Duplicate rows cross partitions')
    dump(out/'split_manifest.json', {k: {'rows': len(ix), 'start': str(dates.iloc[ix].min()), 'end': str(dates.iloc[ix].max())}
                                   for k, ix in partitions.items()})
    print('SPLITS', {k: len(v) for k, v in partitions.items()}, flush=True)
    names = list(SPECS) + ['legacy_rmsle']
    options = {'threads': args.threads, 'device': args.device}
    if args.stage in ['develop', 'all'] and not (out/'frozen_blend.json').exists():
        market_fit = CausalMarket().fit(subset(x, fit), np.log(y[fit])-offset[fit])
        xf = market_fit.transform(subset(x, fit))
        xt = market_fit.transform(subset(x, tune))
        market_dev = CausalMarket().fit(subset(x, pre), np.log(y[pre])-offset[pre])
        save_model(out/'market_dev.joblib', market_dev)
        xp = market_dev.transform(subset(x, pre))
        xc = market_dev.transform(subset(x, cal))
        matrix, records = [], {}
        for name, spec in SPECS.items():
            path = out/f'dev_{name}.joblib'
            start = time.time()
            if path.exists():
                model = joblib.load(path)
            else:
                tuned_path = out/f'tuned_{name}.joblib'
                if tuned_path.exists():
                    tuned = joblib.load(tuned_path)
                else:
                    print('TUNE', name, flush=True)
                    tuned = fit_model(spec, xf, y[fit], offset[fit], cats, iterations=args.iterations,
                                      validation=(xt, y[tune], offset[tune]), **options)
                    save_model(tuned_path, tuned)
                print('DEVELOPMENT REFIT', name, tuned['iterations'], flush=True)
                model = fit_model(spec, xp, y[pre], offset[pre], cats, iterations=tuned['iterations'], **options)
                save_model(path, model)
            p = predict(model, xc, offset[cal])
            matrix.append(p)
            records[name] = {'iterations': model['iterations'], 'calibration': scores(y[cal], p)}
            print('CALIBRATION', name, records[name], 'seconds', round(time.time()-start), flush=True)
        matrix.append(legacy_prediction('calibration_predictions.npz', y[cal]))
        matrix = np.column_stack(matrix)
        np.savez_compressed(out/'calibration_predictions.npz', predictions=matrix, y=y[cal], indices=cal)
        print('FIT MAE BLEND', flush=True)
        weights, optimization = learn_mae_blend(matrix, y[cal], penalty=.02)
        frozen = {'models': names, 'records': records, 'weights': weights.tolist(),
                  'optimization': optimization, 'calibration': scores(y[cal], matrix@weights),
                  'legacy_calibration': scores(y[cal], matrix[:, -1]),
                  'calibration_is_selection_data': True,
                  'historical_evaluation_previously_seen': True,
                  'official_valid_used_for_training_or_weights': False}
        dump(out/'frozen_blend.json', frozen)
        print('FROZEN', frozen, flush=True)
    frozen = json.loads((out/'frozen_blend.json').read_text())
    weights = np.array(frozen['weights'])
    if args.stage in ['final', 'all']:
        market = CausalMarket().fit(x, np.log(y)-offset)
        save_model(out/'market_final.joblib', market)
        xx = market.transform(x)
        test = pd.read_parquet(ROOT/'data/test.parquet')
        tx, _, test_groups, to = make_features(test)
        tx = market.transform(tx)
        matrix = []
        for name, spec in SPECS.items():
            path = out/f'final_{name}.joblib'
            if path.exists():
                model = joblib.load(path)
            else:
                print('FINAL FIT', name, flush=True)
                model = fit_model(spec, xx, y, offset, cats,
                                  iterations=frozen['records'][name]['iterations'], **options)
                save_model(path, model)
            matrix.append(predict(model, tx, to))
        matrix.append(legacy_prediction('test_base_predictions.npz'))
        matrix = np.column_stack(matrix)
        prediction = matrix@weights
        np.savez_compressed(out/'test_predictions.npz', predictions=matrix, blended=prediction)
        submission = export_submission(test, prediction, out/'submission_mae.csv')
        shutil.copy2(out/'submission_mae.csv', ROOT/'submission_mae.csv')
        dump(out/'submission_manifest.json', {'rows': len(submission), 'columns': list(submission),
            'sha256': checksum(out/'submission_mae.csv'), 'rounding': 'nearest integer NTD',
            'all_positive_finite': True, 'ordered_ids_match_template': True,
            'unseen_test_group_fraction': float((~pd.Series(test_groups).isin(groups)).mean()),
            'submitted_to_kaggle': False})
    if args.stage in ['evaluate', 'all'] and not (out/'evaluation.json').exists():
        valid = pd.read_parquet(ROOT/'data/valid.parquet')
        vy = pd.to_numeric(valid[TARGET], errors='raise').to_numpy(float)
        vx, _, vg, vo = make_features(valid)
        report = {'metric': 'MAE', 'evaluation_reused_from_previous_work': True, 'evaluations': {}}
        datasets = [('temporal_2024_05_to_08', subset(x, audit), y[audit], groups[audit], offset[audit], 'dev', groups[pre], 'future_2024_05_to_08_predictions.npz'),
                    ('official_valid', vx, vy, vg, vo, 'final', groups, 'official_valid_predictions.npz')]
        for label, ex, ey, eg, eo, stage, fit_groups, legacy_file in datasets:
            market = joblib.load(out/f'market_{stage}.joblib')
            ex = market.transform(ex)
            matrix = [predict(joblib.load(out/f'{stage}_{name}.joblib'), ex, eo) for name in SPECS]
            old = legacy_prediction(legacy_file, ey)
            matrix.append(old)
            matrix = np.column_stack(matrix)
            p = matrix@weights
            seen = pd.Series(eg).isin(fit_groups).to_numpy()
            clean = ~np.isin(duplicate_groups(valid), fingerprints) if label == 'official_valid' else np.ones(len(ey), bool)
            entry = {'rows': len(ey), 'legacy': scores(ey, old), 'mae_ensemble': scores(ey, p),
                     'mae_improvement_fraction': float(1-np.mean(np.abs(ey-p))/np.mean(np.abs(ey-old))),
                     'uncertainty': clustered_comparison(ey, p, old, eg),
                     'base_models': {name: scores(ey, matrix[:, i]) for i, name in enumerate(names)},
                     'slices': {name: {'rows': int(mask.sum()), 'legacy': scores(ey[mask], old[mask]),
                                      'mae_ensemble': scores(ey[mask], p[mask])}
                                for name, mask in {'seen_group': seen, 'unseen_group': ~seen, 'no_exact_train_duplicate': clean}.items() if mask.any()}}
            report['evaluations'][label] = entry
            np.savez_compressed(out/f'{label}_predictions.npz', y=ey, predictions=matrix, blended=p, groups=eg.astype(str))
            print('EVALUATION', label, entry, flush=True)
        dump(out/'evaluation.json', report)


if __name__ == '__main__':
    main()
