"""Independent inference and integrity checks; never fit a house-price model."""
import json

import joblib
import numpy as np
import pandas as pd

from geography_features import ROOT, OUT, RoadGeography
from train_generalization import SEED, SECOND_SEED, baseline, cold_mask, project_keys
from train_geography import inputs, predict, pred_path, choose
from train_mae import checksum, dump


def reconstruct_test(raw, geocoder):
    from robust_features import enhanced_features
    from spatial_prior import effective_area
    from train_shift import semantic_features
    base, _, _, _ = enhanced_features(raw)
    prep = joblib.load(ROOT/'artifacts/robust_v1/final/preprocess.joblib')
    x = prep['embedding'].transform(base)
    net = joblib.load(ROOT/'artifacts/spatial_v1/final/net_preprocess.joblib')
    area = effective_area(x, 'net'); factor = net['index'].factor(x)
    x['effective_area'] = area.astype(np.float32)
    x['time_index'] = np.log(factor).astype(np.float32)
    x = pd.concat([x, semantic_features(raw)], axis=1)
    x['identity_masked'] = np.float32(0)
    return pd.concat([x, geocoder.transform(raw)], axis=1), area*factor


def main():
    config = json.loads((OUT/'config.json').read_text())
    for filename, expected in config['source_sha256'].items():
        assert checksum(ROOT/filename) == expected, filename
        assert checksum(OUT/'source'/filename) == expected, filename
    for filename, expected in config['input_sha256'].items():
        assert checksum(ROOT/filename) == expected, filename
    selection = json.loads((OUT/'selection.json').read_text())
    assert choose(selection['grid']) == selection['selected']
    assert selection['config_sha256'] == checksum(OUT/'config.json')
    dates = []
    for path in sorted((OUT/'sources').glob('*.csv')):
        a = pd.read_csv(path, usecols=['發生時間'])
        footer = a['發生時間'].dropna().loc[lambda v:v.str.startswith('資料提供日期：')].tolist()
        assert footer == ['資料提供日期：110年07月01日'], path
        dates.append(dict(file=path.name, declared_provision_date='2021-07-01'))
    geocoder = joblib.load(OUT/'geocoder.joblib')
    raw = pd.read_parquet(ROOT/'data/test.parquet')
    cached = joblib.load(OUT/'features.joblib')['test']
    ix = np.random.default_rng(SEED).choice(len(raw), 512, replace=False)
    fresh = geocoder.transform(raw.iloc[ix])
    pd.testing.assert_frame_equal(fresh, cached.iloc[ix].reset_index(drop=True), check_exact=True)
    changed = raw.iloc[ix].assign(總價元=-999, 編號='ignored', 建案名稱='ignored', 棟及號='ignored')
    pd.testing.assert_frame_equal(fresh, geocoder.transform(changed), check_exact=True)
    checks = []
    for fold in ['development', 'stress2021', 'confirmation', 'final']:
        directory = OUT/fold
        models = sorted(directory.glob('*.joblib'))
        if not models:
            continue
        d = inputs(fold); cold = cold_mask(d)
        assert set(project_keys(d['x'])).isdisjoint(set(project_keys(d['vx'].loc[cold])))
        assert (d['vx'].loc[cold, 'sem_relocation']==0).all()
        ix = np.random.default_rng(SEED).choice(len(d['vx']), min(512, len(d['vx'])), replace=False)
        for path in models:
            model = joblib.load(path)
            expected = np.load(pred_path(fold, model['name'], model['seed']))
            actual = predict(model, d['vx'].iloc[ix], d['vbase'][ix])
            np.testing.assert_allclose(actual, expected[ix], rtol=1e-12, atol=1e-7)
            reordered = predict(model, d['vx'].iloc[ix[::-1]], d['vbase'][ix[::-1]])
            np.testing.assert_allclose(reordered, actual[::-1], rtol=1e-12, atol=1e-7)
            checks.append(dict(fold=fold, model=path.name, sample_rows=len(ix),
                max_difference=float(abs(actual-expected[ix]).max()),
                model_sha256=checksum(path)))
    submission = OUT/'submission1017_v7.csv'
    if submission.exists():
        x, base = reconstruct_test(raw, geocoder)
        choice = selection['selected']; name = choice['name']; alpha = choice['alpha']
        ps = []
        for seed in [SEED, SECOND_SEED]:
            model = joblib.load(OUT/'final'/f'{name}_{seed}.joblib')
            p = predict(model, x, base)
            np.testing.assert_allclose(p, np.load(pred_path('final', name, seed)), rtol=1e-12, atol=1e-6)
            ps.append(p)
        d = inputs('final'); old = baseline('final'); cold = cold_mask(d)
        pred = np.where(cold, (1-alpha)*old+alpha*np.mean(ps, axis=0), old)
        saved = pd.read_csv(submission)
        assert list(saved) == ['編號', '總價元']
        assert saved['編號'].astype(str).tolist() == raw['編號'].astype(str).tolist()
        assert saved['編號'].is_unique and (saved['總價元']>0).all()
        np.testing.assert_array_equal(saved['總價元'], np.rint(pred).astype(np.int64))
        np.testing.assert_array_equal(saved.loc[~cold, '總價元'], old[~cold])
        assert checksum(ROOT/submission.name) == checksum(submission)
        dump(OUT/'reproduction.json', dict(raw_test_reconstructed=True,
            geocoder_features_recomputed=True, saved_price_models_reproduce_all_rows=True,
            submission_sha256=checksum(submission), protected_rows_unchanged=True,
            policy_rows=int((x.sem_relocation>0).sum()), refitting_performed=False,
            base_v6_used_for_unchanged_branch=True))
        new = saved['總價元'].to_numpy(float)
        policy = x.sem_relocation.to_numpy()>0
        quantiles = [.01, .05, .5, .95, .99]
        dump(OUT/'test_prediction_summary.json', dict(
            test_targets_used=False, eligible_new_project_rows=int(cold.sum()),
            protected_rows=int((~cold).sum()), policy_rows=int(policy.sum()),
            old_cold_mean=float(old[cold].mean()), new_cold_mean=float(new[cold].mean()),
            old_policy_mean=float(old[policy].mean()), new_policy_mean=float(new[policy].mean()),
            new_over_old_ratio_quantiles=dict(zip(map(str, quantiles), np.quantile(new[cold]/old[cold], quantiles).tolist())),
            absolute_change_quantiles=dict(zip(map(str, quantiles), np.quantile(abs(new[cold]-old[cold]), quantiles).tolist()))))
    dump(OUT/'audit_verification.json', dict(source_and_input_hashes_unchanged=True,
        source_provision_dates=dates, geography_ignores_target_id_project_and_unit=True,
        geography_raw_test_sample_reproduced=True, model_checks=checks,
        strict_cold_project_groups_disjoint=True, selection_recomputed_identically=True,
        v7_reproduced=submission.exists(), fitting_performed=False))
    print(json.dumps(dict(models_verified=len(checks), v7_reproduced=submission.exists()), ensure_ascii=False))


if __name__ == '__main__':
    main()
