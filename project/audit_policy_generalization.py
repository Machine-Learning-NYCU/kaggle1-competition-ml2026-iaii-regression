"""Stress-test the frozen learned policy branch without query-unit price labels.

Only the 441 approved other-unit labels may be used for fitting/evaluation.
New fits are diagnostic and never replace the v6/v7 deployment bundle or CSVs.
"""
import builtins
import io
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from features import normalize
from train_mae import checksum, dump
from train_policy_learned import (FEATURES, cv_select, fit, house_predict, predict,
                                  prepare, price_features, query_features, unit_identity)

ROOT = Path(__file__).resolve().parent
OLD = ROOT/'artifacts/policy_learned_v1'
OUT = ROOT/'artifacts/policy_generalization_v1'
SEED = 20261018
AREA_TOLERANCE_SQM = 1.0


def near_feature_pairs(a, b, tolerance=AREA_TOLERANCE_SQM):
    """Geometry only: same floor/terrace and area difference <= 1 square metre."""
    return ((np.abs(a.net_area_100sqm.to_numpy()[:, None]-b.net_area_100sqm.to_numpy())*100 <= tolerance+1e-9)
        & np.isclose(a.floor_10.to_numpy()[:, None], b.floor_10.to_numpy(), atol=1e-12, rtol=0)
        & (a.terrace.to_numpy()[:, None] == b.terrace.to_numpy()))


def families(projects):
    # Adjacent A18 phases share a family; never choose this grouping using prices.
    return pd.Series(projects).str.replace(r'A18-[123]基地$', 'A18整體基地', regex=True).to_numpy()


def statistics(y, p, b, groups):
    frame = pd.DataFrame(dict(group=groups, ae=abs(y-p), be=abs(y-b)))
    by = frame.groupby('group').agg(rows=('ae', 'size'), mae=('ae', 'mean'), baseline_mae=('be', 'mean'))
    g = frame.assign(gain=frame.be-frame.ae).groupby('group').agg(gain=('gain', 'sum'), n=('gain', 'size')).to_numpy()
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(2000):
        z = g[rng.integers(0, len(g), len(g))].sum(axis=0)
        means.append(z[0]/z[1])
    return dict(rows=len(y), row_mae=float(frame.ae.mean()), baseline_mae=float(frame.be.mean()),
        improvement_pct=float(100*(1-frame.ae.mean()/frame.be.mean())),
        macro_project_mae=float(by.mae.mean()), macro_project_baseline_mae=float(by.baseline_mae.mean()),
        groups=len(by), groups_improved=int((by.mae<by.baseline_mae).sum()),
        groups_worsened=int((by.mae>by.baseline_mae).sum()),
        descriptive_group_gain_ci95=np.quantile(means, [.025, .975]).tolist())


def check_inference(q, bundle, safe, excluded):
    original = predict(q, bundle)
    identifiers = ['編號', '建案名稱', '棟及號', '土地位置建物門牌']
    bare = q.drop(columns=identifiers, errors='ignore')
    pd.testing.assert_frame_equal(original, predict(bare, bundle), check_exact=True)
    renamed = q.assign(編號='unseen-id', 建案名稱='完全未見基地', 棟及號='unseen-unit',
        土地位置建物門牌='完全未見地址', 總價元=9e15, published_house_land_price=9e15)
    pd.testing.assert_frame_equal(original, predict(renamed, bundle), check_exact=True)
    order = np.random.default_rng(SEED).permutation(len(q))
    pd.testing.assert_frame_equal(original.iloc[order].reset_index(drop=True),
        predict(q.iloc[order].reset_index(drop=True), bundle), check_exact=True)
    batched = pd.concat([predict(q.iloc[i:i+37], bundle) for i in range(0, len(q), 37)], ignore_index=True)
    pd.testing.assert_frame_equal(original, batched, check_exact=True)
    repeated = pd.concat([q.iloc[:1], q.iloc[:1]], ignore_index=True)
    np.testing.assert_array_equal(predict(repeated, bundle).iloc[0], predict(repeated, bundle).iloc[1])
    # Warm model is already in memory. Any file access during inference is a failure.
    def deny(*args, **kwargs):
        raise AssertionError('Prediction attempted external file access')
    with ExitStack() as stack:
        for obj, name in [(builtins, 'open'), (io, 'open'), (os, 'open'), (Path, 'open'),
                          (pd, 'read_csv'), (pd, 'read_parquet'), (pd, 'read_excel'), (joblib, 'load')]:
            stack.enter_context(patch.object(obj, name, side_effect=deny))
        pd.testing.assert_frame_equal(original, predict(bare, bundle), check_exact=True)

    # Poison synthetic blocked rows. Real excluded-unit prices are never loaded.
    fake = excluded.assign(area_sqm=-999., terrace_ping=-999., adjusted_house_land_price=9e15)
    combined = pd.concat([safe, fake], ignore_index=True)
    selected, x, y, area, keep = prepare(combined, q)
    assert len(selected)==441 and int((~keep).sum())==1404
    np.testing.assert_array_equal(y, safe.adjusted_house_land_price.to_numpy(float))
    fake2 = fake.assign(area_sqm=123456., terrace_ping=1., adjusted_house_land_price=-9e15)
    selected2, x2, y2, area2, keep2 = prepare(pd.concat([safe, fake2], ignore_index=True), q)
    pd.testing.assert_frame_equal(x, x2, check_exact=True)
    np.testing.assert_array_equal(y, y2); np.testing.assert_array_equal(area, area2)
    qx, qa = query_features(q)
    refitted = fit('linear_l1', x, y, area)
    np.testing.assert_allclose(house_predict(refitted, qx, qa), original.house_prediction, rtol=1e-12, atol=1e-6)

    # Only the monetary DIFFERENCE in the supplied note may affect deduction.
    common = q.iloc[:1].copy()
    common['備註'] = '安置住宅 原房屋價款4000000元 差額減收後之房屋價款為3500000元'
    translated = common.assign(備註='安置住宅 原房屋價款9000000元 差額減收後之房屋價款為8500000元')
    bigger_cut = common.assign(備註='安置住宅 原房屋價款4000000元 差額減收後之房屋價款為3400000元')
    first = predict(common, bundle)
    pd.testing.assert_frame_equal(first, predict(translated, bundle), check_exact=True)
    assert abs(float(first.prediction.iloc[0]-predict(bigger_cut, bundle).prediction.iloc[0])-100000)<1e-8
    return dict(rows=len(q), identity_removal_invariant=True, arbitrary_identity_and_price_columns_ignored=True,
        row_order_invariant=True, batch_size_invariant=True, duplicate_query_invariant=True,
        file_access_blocked_during_all_query_predictions=True, excluded_unit_synthetic_poison_has_no_effect=True,
        original_deployment_model_refit_from_safe_labels_reproduced=True,
        note_equal_differences_invariant=True, note_extra_discount_exact=True,
        actual_excluded_unit_offer_labels_loaded=False), original


def evaluate_scheme(name, outer_groups, safe, x, y, area, qx, qa, query_projects, purge=False):
    groups = safe.project.to_numpy(); family = families(groups)
    fixed = np.full(len(y), np.nan); nested = fixed.copy(); reference = fixed.copy()
    folds=[]; qpred=[]; own_project=np.full(len(qx), np.nan)
    for fold, (fit_ix, val_ix) in enumerate(LeaveOneGroupOut().split(x, y, outer_groups)):
        assert set(outer_groups[fit_ix]).isdisjoint(set(outer_groups[val_ix]))
        before = len(fit_ix)
        if purge:
            close = near_feature_pairs(x.iloc[fit_ix], x.iloc[val_ix]).any(axis=1)
            fit_ix = fit_ix[~close]
            assert not near_feature_pairs(x.iloc[fit_ix], x.iloc[val_ix]).any()
        if len(fit_ix)<20 or len(set(family[fit_ix]))<3:
            raise ValueError(f'Insufficient support in {name}/{fold}: {len(fit_ix)} rows')
        # No validation label reaches fit or the inner selector.
        tx = x.iloc[fit_ix].reset_index(drop=True); ty=y[fit_ix]; ta=area[fit_ix]
        model = fit('linear_l1', tx, ty, ta)
        baseline_model = fit('median_unit', tx, ty, ta)
        fixed[val_ix] = house_predict(model, x.iloc[val_ix], area[val_ix])
        reference[val_ix] = house_predict(baseline_model, x.iloc[val_ix], area[val_ix])
        choice, inner_scores, _ = cv_select(tx, ty, ta, family[fit_ix], 3)
        selected = model if choice=='linear_l1' else baseline_model if choice=='median_unit' else fit(choice, tx, ty, ta)
        nested[val_ix] = house_predict(selected, x.iloc[val_ix], area[val_ix])
        pred = house_predict(model, qx, qa); qpred.append(pred)
        if name=='leave_one_project_out':
            qm = np.isin(query_projects, groups[val_ix])
            own_project[qm] = pred[qm]
        record=dict(fold=fold, heldout_group=str(outer_groups[val_ix[0]]),
            train_rows=len(fit_ix), purged_rows=before-len(fit_ix), heldout_rows=len(val_ix),
            train_projects=sorted(set(groups[fit_ix])), heldout_projects=sorted(set(groups[val_ix])),
            train_indices=fit_ix.tolist(), heldout_indices=val_ix.tolist(),
            nested_selected=choice, inner_scores=inner_scores,
            fixed_mae=float(abs(y[val_ix]-fixed[val_ix]).mean()),
            nested_mae=float(abs(y[val_ix]-nested[val_ix]).mean()),
            baseline_mae=float(abs(y[val_ix]-reference[val_ix]).mean()))
        folds.append(record)
    assert np.isfinite(np.c_[fixed, nested, reference]).all()
    frame=safe[['project', 'unit', 'source']].assign(target_offer=y, fixed_prediction=fixed,
        nested_prediction=nested, baseline_prediction=reference)
    frame.to_csv(OUT/f'{name}_oof.csv', index=False, encoding='utf-8-sig')
    summary=dict(folds=len(folds), fixed=statistics(y, fixed, reference, groups),
        nested=statistics(y, nested, reference, groups), outer_group_unit=name,
        minimum_fitting_rows=min(r['train_rows'] for r in folds),
        maximum_purged_rows=max(r['purged_rows'] for r in folds),
        uncertainty='Project-cluster descriptive interval; shared source correlations and prior model selection are not corrected. Only two source domains exist.')
    dump(OUT/f'{name}.json', dict(summary=summary, folds=folds))
    print('SCHEME', name, json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, np.asarray(qpred), own_project, folds


def prediction_change(full, other):
    rel=abs(other/full-1)*100
    return dict(rows=len(full), mean_absolute_change_ntd=float(np.mean(abs(full-other))),
        median_absolute_change_pct=float(np.median(rel)), p95_absolute_change_pct=float(np.quantile(rel, .95)),
        max_absolute_change_pct=float(rel.max()), mean_prediction_ntd=float(other.mean()),
        caution='Prediction sensitivity only, not accuracy: no test prices were evaluated.')


def negative_control(x, y, area, folds):
    scores=[]
    for repeat in range(20):
        rng=np.random.default_rng(SEED+repeat)
        out=np.full(len(y), np.nan)
        for f in folds:
            tr=np.asarray(f['train_indices']); va=np.asarray(f['heldout_indices'])
            # Preserve area scale; destroy feature/unit-price correspondence.
            shuffled=(y[tr]/area[tr])[rng.permutation(len(tr))]*area[tr]
            m=fit('linear_l1', x.iloc[tr], shuffled, area[tr])
            out[va]=house_predict(m, x.iloc[va], area[va])
        scores.append(float(np.mean(abs(out-y))))
    return dict(repeats=20, shuffled_unit_price_oof_mae=scores, median_mae=float(np.median(scores)),
        min_mae=float(min(scores)), max_mae=float(max(scores)),
        interpretation='Negative control, not a formal permutation p-value; unit prices are not exchangeable across policy projects.')


def parking_audit():
    park=pd.read_csv(ROOT/'artifacts/policy_v1/parking_schedules.csv')
    records=[]; all_error=[]
    for group, v in park.groupby('project'):
        fitting=park.loc[park.project!=group, 'price'].to_numpy(float)
        prediction=float(np.median(fitting))
        err=abs(v.price.to_numpy(float)-prediction); all_error.extend(err)
        records.append(dict(project=group, heldout_rows=len(v), learned_other_project_median=prediction,
            offer_mae=float(err.mean())))
    return dict(rows=len(park), projects=park.project.nunique(), leave_project_out_offer_mae=float(np.mean(all_error)),
        folds=records, same_query_parking_identity_screening=False,
        limitation='Inventory-wide constant model; individual test parking spaces are not matched or assigned an external price. This is offer validation, not sale validation.')


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    protocol=dict(purpose='Generalization and no-answer-lookup audit of existing v6/v7 policy branch',
        labels='Only 441 previously screened other-unit house/land offers; no excluded-unit prices or Kaggle test labels',
        schemes=['leave_one_project_out', 'leave_related_family_out', 'leave_source_out', 'purged_leave_project_out'],
        inner_selection='Original three candidates, 3-fold grouped by related project family; no new tuning',
        fixed_model='Original linear_l1 alpha=0.001', near_match='Same floor and terrace; net area within 1.0 sqm',
        near_area_tolerance_sqm=AREA_TOLERANCE_SQM, seed=SEED, negative_control_repeats=20,
        known_before_testing='Unlabeled covariate inspection: 304 of 441 training rows are within near-match tolerance of a query; 137 remain.',
        decision='Report all results; do not replace submission or fit public-score adjustments',
        pristine_holdout=False, source_sha256=checksum(Path(__file__)),
        sources=['https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html',
                 'https://scikit-learn.org/stable/modules/linear_model.html'])
    dump(OUT/'protocol.json', protocol)
    oldconfig=json.loads((OLD/'config.json').read_text())
    for name, sha in {**oldconfig['source_sha256'], **oldconfig['input_sha256']}.items():
        assert checksum(ROOT/name)==sha, name
    files=set(OLD.rglob('*')) | set(ROOT.glob('submission*.csv'))
    files|={ROOT/f for f in oldconfig['source_sha256']} | {ROOT/f for f in oldconfig['input_sha256']}
    frozen={str(p.relative_to(ROOT)):checksum(p) for p in sorted(files) if p.is_file()}
    dump(OUT/'protected_inventory.json', frozen)
    safe=pd.read_csv(OLD/'training_units.csv')
    excluded=pd.read_csv(OLD/'excluded_test_units.csv', usecols=['project', 'unit'])
    raw=pd.read_parquet(ROOT/'data/test.parquet')
    q=raw.loc[raw['備註'].fillna('').str.contains('安置住宅')].copy().reset_index(drop=True)
    assert len(safe)==441 and len(q)==1408 and len(excluded)==1404
    assert not unit_identity(safe,'project','unit').isin(unit_identity(raw,'建案名稱','棟及號')).any()
    x=price_features(safe.area_sqm, safe.unit.str.extract(r'-(\d+)F',expand=False).astype(float),safe.terrace_ping>0)
    y=safe.adjusted_house_land_price.to_numpy(float); area=safe.area_sqm.to_numpy(float)
    qx,qa=query_features(q); bundle=joblib.load(OLD/'model.joblib')
    checks, original=check_inference(q,bundle,safe,excluded)
    saved=pd.read_csv(ROOT/'submission1017_v7.csv',dtype={'編號':str}).set_index('編號')['總價元']
    np.testing.assert_array_equal(np.rint(original.prediction).astype(np.int64),saved.loc[q['編號'].astype(str)])
    dump(OUT/'inference_checks.json',checks)
    print('INFERENCE CHECKS PASSED', flush=True)
    summaries={}; sensitivity={}; qmatrix=None; project_folds=None
    for name, groups, purge in [
        ('leave_one_project_out',safe.project.to_numpy(),False),
        ('leave_related_family_out',families(safe.project),False),
        ('leave_source_out',safe.source.to_numpy(),False),
        ('purged_leave_project_out',safe.project.to_numpy(),True)]:
        summary,pred,own,folds=evaluate_scheme(name,groups,safe,x,y,area,qx,qa,q['建案名稱'].map(normalize).to_numpy(),purge)
        summaries[name]=summary
        full=original.house_prediction.to_numpy()
        sensitivity[name]=dict(per_fold=[prediction_change(full,v) for v in pred],
            across_fold_p95_range_pct=float(np.quantile((pred.max(axis=0)-pred.min(axis=0))/full*100,.95)))
        if name=='leave_one_project_out':
            assert np.isfinite(own).all()
            sensitivity['exclude_query_own_project']=prediction_change(full,own)
            project_folds=folds; qmatrix=pred
    close=near_feature_pairs(x,qx)
    keep=~close.any(axis=1)
    assert keep.sum()>=20
    m=fit('linear_l1',x.loc[keep],y[keep],area[keep])
    purged_pred=house_predict(m,qx,qa)
    sensitivity['exclude_query_similar_features']=dict(training_rows=int(keep.sum()),removed_rows=int((~keep).sum()),
        **prediction_change(original.house_prediction.to_numpy(),purged_pred))
    identical=near_feature_pairs(x,qx,tolerance=1e-6)
    support=dict(training_projects=safe.project.nunique(), query_projects=q['建案名稱'].nunique(),
        query_project_overlap_with_training=q['建案名稱'].map(normalize).isin(set(safe.project.map(normalize))).mean(),
        query_rows_near_training=int(close.any(axis=0).sum()),query_rows_physically_identical=int(identical.any(axis=0).sum()),
        query_rows_outside_any_feature_range=int(((qx<x.min())|(qx>x.max())).any(axis=1).sum()),
        train_net_area_range=[float(area.min()),float(area.max())],query_net_area_range=[float(qa.min()),float(qa.max())],
        all_sources_same_policy_region='Taoyuan airport relocation scheme; no independent other-city policy domain')
    control=negative_control(x,y,area,project_folds)
    observed=summaries['leave_one_project_out']['fixed']['row_mae']
    control['observed_fixed_oof_mae']=observed
    control['shuffles_worse_than_observed']=int(sum(v>observed for v in control['shuffled_unit_price_oof_mae']))
    parking=parking_audit()
    # Check smooth, positive responses over the physical support; this is not an accuracy test.
    mesh=np.array(np.meshgrid(np.linspace(area.min(),area.max(),30),np.arange(1,11),[0.,1.],indexing='ij'))
    grid=mesh.reshape(3,-1).T
    xx=price_features(grid[:,0],grid[:,1],grid[:,2]);pp=house_predict(bundle['house_model'],xx,grid[:,0])
    nx=price_features(grid[:,0]+.1,grid[:,1],grid[:,2]);nn=house_predict(bundle['house_model'],nx,grid[:,0]+.1)
    physical=dict(grid_points=len(xx),positive_predictions=bool((pp>0).all()),
        area_plus_0_1sqm_increases_total_price=bool((nn>pp).all()),
        no_identity_features=True,accuracy_claim=False)
    assert physical['positive_predictions'] and physical['area_plus_0_1sqm_increases_total_price']
    scaler=bundle['house_model'].named_steps['standardscaler'];reg=bundle['house_model'].named_steps['quantileregressor']
    unscaled=reg.coef_/scaler.scale_*1e5
    intercept=(reg.intercept_-np.sum(reg.coef_*scaler.mean_/scaler.scale_))*1e5
    formula=dict(unit_price_intercept_ntd_per_sqm=float(intercept),
        coefficients_for_original_six_features=dict(zip(FEATURES,unscaled.tolist())),
        expression='house_land_prediction = net_area_sqm * (intercept + sum(coefficient * physical_feature))')
    assert all(checksum(ROOT/p)==h for p,h in frozen.items()),'Frozen model or submission changed'
    result=dict(status='completed',inference=checks,validation=summaries,sensitivity=sensitivity,
        support=support,negative_control=control,parking=parking,physical_checks=physical,learned_formula=formula,
        protected_files_unchanged=len(frozen),deployment_model_changed=False,new_submission_created=False,
        limitations=['Post-selection reanalysis, not a pristine independent dataset.',
            'Only official house/land offers are evaluated; actual relocation sales remain unvalidated.',
            'All 10 bases belong to the same regional policy scheme; no claim of generalization to every relocation program.',
            'Project intervals do not account for common-source correlation or all prior model searches.'])
    dump(OUT/'audit.json',result)
    print('AUDIT COMPLETE',json.dumps(dict(support=support,negative_control=control,
        purge_sensitivity=sensitivity['exclude_query_similar_features'],own_project_sensitivity=sensitivity['exclude_query_own_project']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
