"""Read-only model diagnostics for frozen v7; never train or export predictions.

All diagnostics are post-selection. A checkpoint minimum is descriptive, not
an authorized replacement for the frozen 3,000-tree submission.
"""
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from features import MISSING, normalize, duplicate_groups
from train_generalization import baseline, cold_mask, project_keys, ProjectIdentityMasker
from train_mae import checksum, dump
from train_robust import data as raw_training

ROOT = Path(__file__).resolve().parent
GEO = ROOT / 'artifacts/geography_v1'
OUT = ROOT / 'artifacts/overfitting_v7'
FOLDS = ['stress2021', 'development', 'confirmation']
SEEDS = [20261017, 20261018]
STEPS = [500, 1000, 1500, 2000, 2500, 3000]
AUDIT_SEED = 20261018


def gain_metrics(y, old, new):
    a = np.abs(y-old); b = np.abs(y-new)
    return dict(rows=len(y), baseline_mae=float(a.mean()), candidate_mae=float(b.mean()),
                gain_ntd=float((a-b).mean()), gain_pct=float(100*(a.mean()-b.mean())/a.mean()))


def cluster_ci(frame, group, before='old_error', after='new_error'):
    a = frame.assign(gain=frame[before]-frame[after]).groupby(group).agg(
        gain=('gain', 'sum'), n=('gain', 'size')).to_numpy()
    rng = np.random.default_rng(AUDIT_SEED)
    samples = np.empty(2000)
    for i in range(len(samples)):
        s = a[rng.integers(0, len(a), len(a))].sum(axis=0)
        samples[i] = s[0]/s[1]
    return np.quantile(samples, [.025, .975]).tolist()


def model_file(fold, name, seed):
    p = GEO/fold/f'{name}_{seed}.joblib'
    if not p.exists():
        assert name == 'cat_control' and seed == SEEDS[0] and fold != 'confirmation'
        p = ROOT/f'artifacts/generalization_v1/{fold}/identity_log_{seed}.joblib'
    return p


def frozen_inventory():
    files = set(GEO.rglob('*'))
    files |= set(ROOT.glob('*.py')) | set(ROOT.glob('submission*.csv'))
    files |= set((ROOT/'artifacts/policy_learned_v1').rglob('*'))
    config = json.loads((GEO/'config.json').read_text())
    for name, expected in {**config['source_sha256'], **config['input_sha256']}.items():
        p = ROOT/name
        assert checksum(p) == expected, name
        files.add(p)
    for fold in FOLDS:
        for name in ['cat_control', 'cat_geo']:
            for seed in SEEDS:
                files.add(model_file(fold, name, seed))
    files |= {ROOT/'data/train.parquet', ROOT/'data/test.parquet',
              ROOT/'artifacts/leaderboard_history.json'}
    return {str(p.relative_to(ROOT)): checksum(p) for p in sorted(files) if p.is_file()}


def inputs(fold, geo):
    # Load caches directly: no prepare/fitting functions or fallback writes.
    d = joblib.load(ROOT/f'artifacts/shift_v1/{fold}/data.joblib')
    for key, ix in [('x', d['fit_ix']), ('vx', d['valid_ix'])]:
        x = d[key].copy()
        x['identity_masked'] = np.float32(0)
        d[key] = pd.concat([x, geo.iloc[ix].reset_index(drop=True)], axis=1)
    return d


def stages(fold, name, seed, d, train_ix):
    model = joblib.load(model_file(fold, name, seed))
    estimator = model['model']
    assert estimator.tree_count_ == 3000
    assert not {'編號', '總價元', 'ID', 'target', 'price'} & set(model['columns'])
    result = {}
    rows = []
    learned = estimator.get_evals_result()['learn']['RMSE']
    for split, x, y, base in [
        ('fit_sample', d['x'].iloc[train_ix], d['y'][train_ix], d['base'][train_ix]),
        ('future', d['vx'], d['vy'], d['vbase'])
    ]:
        xx = x[model['columns']]
        pool = Pool(xx, cat_features=estimator.get_cat_feature_indices())
        predictions = []
        weights = np.clip(np.sqrt(y/1e7), .2, 6)
        target = np.log(y/base)
        for step, z in zip(STEPS, estimator.staged_predict(pool, eval_period=500, thread_count=8)):
            raw = np.exp(np.clip(z, -10, 20))*base
            deployed = np.maximum(raw*np.exp(-.5*x.time_index.to_numpy(float)), 1.)
            predictions.append(deployed)
            rows.append(dict(fold=fold, model=name, seed=seed, split=split, trees=step,
                rows=len(y), raw_total_mae=float(np.mean(abs(y-raw))),
                deployed_total_mae=float(np.mean(abs(y-deployed))),
                weighted_log_rmse=float(np.sqrt(np.average((target-z)**2, weights=weights))),
                logged_fit_rmse=float(learned[step-1])))
        assert len(predictions) == len(STEPS)
        result[split] = predictions
        del pool
    saved = np.load(GEO/fold/f'{name}_{seed}_predictions.npy')
    np.testing.assert_allclose(result['future'][-1], saved, rtol=1e-12, atol=1e-6)
    print('STAGES VERIFIED', fold, name, seed, flush=True)
    del model, estimator
    gc.collect()
    return result['future'], rows


def fold_diagnostics(fold, d, raw, dates, geo):
    cold = cold_mask(d); y = d['vy']; old = baseline(fold)
    train_keys = set(project_keys(d['x']))
    val_keys = project_keys(d['vx']).to_numpy()
    assert train_keys.isdisjoint(val_keys[cold])
    fit_dates = dates.iloc[d['fit_ix']]; val_dates = dates.iloc[d['valid_ix']].reset_index(drop=True)
    assert fit_dates.max() < val_dates.min()
    np.testing.assert_array_equal(d['y'], pd.to_numeric(raw.iloc[d['fit_ix']]['總價元']).to_numpy(float))
    np.testing.assert_array_equal(y, pd.to_numeric(raw.iloc[d['valid_ix']]['總價元']).to_numpy(float))
    fingerprints = duplicate_groups(raw)
    overlaps = np.isin(fingerprints[d['valid_ix']], fingerprints[d['fit_ix']])
    assert not overlaps.any()
    train_ix = np.sort(np.random.default_rng(AUDIT_SEED).choice(len(d['y']), min(50000, len(d['y'])), replace=False))
    pred = {}; curves = []; ensemble_curves = []
    for name in ['cat_control', 'cat_geo']:
        seed_predictions = []
        for seed in SEEDS:
            p, c = stages(fold, name, seed, d, train_ix)
            curves.extend(c); seed_predictions.append(p)
        pred[name] = np.mean(seed_predictions, axis=0)
        for step, q in zip(STEPS, pred[name]):
            p = np.where(cold, .5*old+.5*q, old)
            for scope, mask in [('all', np.ones(len(y), bool)), ('cold', cold), ('cold_late', cold & ~d['tune'])]:
                ensemble_curves.append(dict(fold=fold, model=name, trees=step, scope=scope,
                    **gain_metrics(y[mask], old[mask], p[mask])))
    new = np.where(cold, .5*old+.5*pred['cat_geo'][-1], old)
    control = np.where(cold, .5*old+.5*pred['cat_control'][-1], old)
    frame = pd.DataFrame(dict(project=val_keys, y=y, old=old, new=new, control=control,
        old_error=abs(y-old), new_error=abs(y-new), control_error=abs(y-control),
        cold=cold, city=d['vx'].geo_city.to_numpy(), month=val_dates.dt.strftime('%Y-%m'),
        location_level=d['vx'].ext_location_level.to_numpy(int)))
    # Use the declared temporal boundary, even when the last fitting day has no transactions.
    cut = {'stress2021': pd.Timestamp('2021-08-10'),
           'development': pd.Timestamp('2022-08-10'),
           'confirmation': pd.Timestamp('2023-08-10')}[fold]
    assert fit_dates.max() <= cut < val_dates.min()
    frame['horizon_block'] = np.select(
        [val_dates <= cut+pd.DateOffset(months=4), val_dates <= cut+pd.DateOffset(months=8)],
        ['months_1_4', 'months_5_8'], default='months_9_12')
    fit_roads = set(d['x'].geo_road1) - {MISSING, ''}
    roads = d['vx'].geo_road1
    frame['road_status'] = np.where(roads.isin([MISSING, '']), 'missing',
        np.where(roads.isin(fit_roads), 'seen_road', 'unseen_road'))
    price_edges = np.quantile(d['y'], [.5, .9, .99])
    frame['price_bin_fit_quantiles'] = np.searchsorted(price_edges, y, side='right')
    a = frame.loc[cold].copy()
    groups = a.groupby('project').agg(rows=('y', 'size'), baseline_mae=('old_error', 'mean'),
        candidate_mae=('new_error', 'mean'), control_mae=('control_error', 'mean'))
    groups['gain_ntd'] = groups.baseline_mae-groups.candidate_mae
    groups['total_gain'] = groups.gain_ntd*groups.rows
    groups['gain_pct'] = groups.gain_ntd/groups.baseline_mae*100
    groups.to_csv(OUT/f'{fold}_project_metrics.csv', encoding='utf-8-sig')
    subgroup = []
    for col in ['month', 'horizon_block', 'city', 'location_level', 'road_status', 'price_bin_fit_quantiles']:
        for value, sub in a.groupby(col):
            subgroup.append(dict(fold=fold, dimension=col, value=str(value), projects=sub.project.nunique(),
                **gain_metrics(sub.y.to_numpy(), sub.old.to_numpy(), sub.new.to_numpy())))
    sensitivity = []
    for n in [1, 5, 10]:
        biggest = groups.nlargest(n, 'total_gain').index
        remain = a.loc[~a.project.isin(biggest)]
        sensitivity.append(dict(remove_top_gain_projects=n,
            **gain_metrics(remain.y.to_numpy(), remain.old.to_numpy(), remain.new.to_numpy())))
    seed_metrics = []
    for seed in SEEDS:
        p = np.load(GEO/fold/f'cat_geo_{seed}_predictions.npy')
        seed_metrics.append(dict(seed=seed, **gain_metrics(y[cold], old[cold], .5*old[cold]+.5*p[cold])))
    baseline_cold = float(abs(y[cold]-old[cold]).mean())
    seed_disagreement = .5*np.abs(np.load(GEO/fold/f'cat_geo_{SEEDS[0]}_predictions.npy')-
        np.load(GEO/fold/f'cat_geo_{SEEDS[1]}_predictions.npy'))
    report = dict(fold=fold, fit_rows=len(d['y']), future_rows=len(y), cold_rows=int(cold.sum()),
        fit_last_date=str(fit_dates.max().date()), nominal_training_cutoff=str(cut.date()),
        future_first_date=str(val_dates.min().date()),
        future_last_date=str(val_dates.max().date()), fit_project_overlap_in_cold=0,
        exact_raw_feature_duplicates_across_split=int(overlaps.sum()),
        all=gain_metrics(y, old, new), cold=gain_metrics(y[cold], old[cold], new[cold]),
        cold_late=gain_metrics(y[cold & ~d['tune']], old[cold & ~d['tune']], new[cold & ~d['tune']]),
        matching_control=gain_metrics(y[cold], control[cold], new[cold]),
        project_cluster_gain_ci95=cluster_ci(a, 'project'),
        city_cluster_gain_ci95=cluster_ci(a, 'city'),
        projects=len(groups), projects_improved=int((groups.gain_ntd > 0).sum()),
        projects_worsened=int((groups.gain_ntd < 0).sum()),
        macro_project_baseline_mae=float(groups.baseline_mae.mean()),
        macro_project_candidate_mae=float(groups.candidate_mae.mean()),
        project_gain_quantiles={str(q):float(groups.gain_pct.quantile(q)) for q in [.1, .25, .5, .75, .9]},
        top10_positive_gain_share=float(groups.total_gain.nlargest(10).sum()/groups.total_gain.clip(lower=0).sum()),
        leave_top_gain_projects_out=sensitivity, individual_seeds=seed_metrics,
        seed_prediction_difference_median_ntd=float(np.median(seed_disagreement[cold])),
        seed_prediction_difference_p95_ntd=float(np.quantile(seed_disagreement[cold], .95)),
        fit_price_quantile_edges=price_edges.tolist(), baseline_cold_mae=baseline_cold)
    dump(OUT/f'{fold}.json', report)
    print('FOLD DONE', fold, json.dumps(report, ensure_ascii=False), flush=True)
    return report, curves, ensemble_curves, subgroup


def policy_diagnostics():
    from train_policy_learned import price_features, house_predict, unit_identity
    folder = ROOT/'artifacts/policy_learned_v1'
    safe = pd.read_csv(folder/'training_units.csv')
    excluded = pd.read_csv(folder/'excluded_test_units.csv')
    test = pd.read_parquet(ROOT/'data/test.parquet')
    keys = unit_identity(safe, 'project', 'unit')
    assert not keys.isin(unit_identity(excluded, 'project', 'unit')).any()
    assert not keys.isin(unit_identity(test, '建案名稱', '棟及號')).any()
    x = price_features(safe.area_sqm, safe.unit.str.extract(r'-(\d+)F', expand=False).astype(float), safe.terrace_ping > 0)
    y = safe.adjusted_house_land_price.to_numpy(float)
    model = joblib.load(folder/'model.joblib')
    pred = house_predict(model['house_model'], x, safe.area_sqm)
    oof = pd.read_csv(folder/'nested_group_oof.csv')
    assert list(zip(safe.project, safe.unit)) == list(zip(oof.project, oof.unit))
    np.testing.assert_allclose(y, oof.target_offer)
    frame = pd.DataFrame(dict(project=safe.project, old_error=abs(y-oof.baseline_prediction),
        new_error=abs(y-oof.outer_prediction)))
    by_project = frame.groupby('project').mean()
    return dict(training_rows=len(safe), projects=safe.project.nunique(), excluded_same_test_unit_offers=len(excluded),
        final_model_resubstitution_offer_mae=float(np.mean(abs(y-pred))),
        nested_selection_outer_offer_mae=float(frame.new_error.mean()),
        nested_baseline_offer_mae=float(frame.old_error.mean()),
        project_cluster_gain_ci95=cluster_ci(frame, 'project'),
        projects_improved=int((by_project.new_error < by_project.old_error).sum()),
        projects_worsened=int((by_project.new_error > by_project.old_error).sum()),
        project_sample_counts=safe.project.value_counts().to_dict(),
        caveat='Only 10 projects, price target is official offer proxy, not actual sale; resubstitution final model and outer selection procedure differ; no real sale error available for relocation test subgroup.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(purpose='Post-selection overfitting diagnosis only', folds=FOLDS, seeds=SEEDS,
        checkpoints=STEPS, training_sample_rows=50000, sample_seed=AUDIT_SEED, bootstrap_replicates=2000,
        model_refitting=False, parameters_or_submission_changed=False, public_score_tuning=False,
        independent_new_test=False, note='All checkpoints/subgroups are diagnostic; no checkpoint or routing selection.',
        source_sha256=checksum(Path(__file__)),
        references=['https://catboost.ai/docs/en/concepts/faq', 'https://www.jmlr.org/papers/v11/cawley10a.html'])
    dump(OUT/'protocol.json', protocol)
    before = frozen_inventory()
    dump(OUT/'input_inventory.json', before)
    raw, _, dates = raw_training()
    geo = joblib.load(GEO/'features.joblib')['train']
    reports=[]; curves=[]; ensemble=[]; subgroup=[]
    for fold in FOLDS:
        d = inputs(fold, geo)
        r, c, e, s = fold_diagnostics(fold, d, raw, dates, geo)
        reports.append(r); curves.extend(c); ensemble.extend(e); subgroup.extend(s)
        pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv', index=False)
        pd.DataFrame(ensemble).to_csv(OUT/'ensemble_checkpoints.csv', index=False)
        pd.DataFrame(subgroup).to_csv(OUT/'subgroup_metrics.csv', index=False, encoding='utf-8-sig')
        del d
        gc.collect()
    policy = policy_diagnostics()
    after = {p:checksum(ROOT/p) for p in before}
    assert before == after, 'An existing source, model, evidence file, or submission changed'
    summary = dict(status='diagnostics_completed', folds=reports, policy=policy,
        checked_files_unchanged=len(before), public_scores_user_reported={'v6':1745458.0575,'v7':1716414.5323},
        no_refitting=True, no_new_submission=True, independent_holdout=False)
    dump(OUT/'audit.json', summary)
    print('AUDIT COMPLETE', json.dumps(dict(checked_files_unchanged=len(before), policy=policy), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
