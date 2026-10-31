"""Conservative two-window selection, frozen before confirmation inference."""
import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from train_hedonic import ROOT, OUT, V2, SPATIAL, SPECS as BASE_SPECS, predict, inputs as original_inputs
from train_spatial import base_data
from train import SPECS as OLD_SPECS, scores
from train_mae import dump, checksum, clustered_comparison, export_submission
from select_robust import apply_selection
from robust_features import enhanced_features
from spatial_prior import effective_area

WINDOWS = ['stress2021', 'development']
# Both Huber attempts and target-statistic context features failed development
# MAE before stress/confirmation fitting; retain their artifacts as rejected.
SPECS = {n:BASE_SPECS[n] for n in ['net_log','deep_log']}
ENSEMBLES = {'mean_structural': ['net_log', 'deep_log']}


def inputs(fold, name):
    return original_inputs(fold,name)


def members(name):
    return ENSEMBLES.get(name,[name])


def baseline(fold, d):
    path = SPATIAL/fold if fold=='stress2021' else V2/fold
    prefix = 'base_' if fold=='stress2021' else ''
    old = np.column_stack([np.load(path/f'{prefix}{n}_predictions.npy') for n in OLD_SPECS])
    cal = json.loads((V2/'legacy_development_calibration.json').read_text())
    matrix = np.column_stack([old@np.array(cal['weights'])]+[
        np.load(path/f'{prefix}{n}_predictions.npy') for n in ['cat_unit', 'cat_log', 'lgb_unit']])
    sel = json.loads((V2/'selection.json').read_text())
    assert sel['comparable_fraction']==0
    return apply_selection(sel, matrix, d['seen'], np.zeros(len(matrix)),
                           np.zeros(len(matrix)), np.exp(d['vx'].time_index.to_numpy(float)))


def candidate_arrays(fold):
    result = {}
    for name in SPECS:
        d, _ = inputs(fold, name)
        p = np.load(OUT/fold/f'{name}_predictions.npy')
        t = d['vx'].time_index.to_numpy(float)
        for damping in [0., .5, 1.]:
            result[f'{name}@{damping}'] = p*np.exp((damping-1)*t)
    for ensemble,names in ENSEMBLES.items():
        for damping in [0., .5, 1.]:
            result[f'{ensemble}@{damping}'] = np.mean([
                result[f'{name}@{damping}'] for name in names], axis=0)
    return result


def choose_branch(records):
    """Every historical window must gain >=0.5%; retain >=25% of parent."""
    qualified = [r for r in records if max(r['ratios']) <= .995]
    if not qualified:
        return dict(candidate=None, alpha=0., ratios=[1., 1.], worst_ratio=1.)
    minimum = min(r['worst_ratio'] for r in qualified)
    # Relative 0.5% tolerance on the worst normalized loss; favor a smaller change.
    eligible = [r for r in qualified if r['worst_ratio'] <= minimum*1.005]
    return min(eligible, key=lambda r: (r['alpha'], r['worst_ratio'], r['candidate']))


def freeze():
    path = OUT/'selection.json'
    if path.exists():
        print(path.read_text());return
    data = {}
    for fold in WINDOWS:
        d = base_data(fold)
        data[fold] = dict(y=d['vy'], seen=d['seen'], previous=baseline(fold,d),
                          candidates=candidate_arrays(fold))
    records = {}; selected = {}
    for branch in ['seen','unseen']:
        grid = []
        for candidate in data[WINDOWS[0]]['candidates']:
            for alpha in [.25,.5,.75]:
                ratios = [];maes = [];parent_maes = []
                for fold in WINDOWS:
                    d = data[fold];mask=d['seen'] if branch=='seen' else ~d['seen']
                    p = (1-alpha)*d['previous']+alpha*d['candidates'][candidate]
                    old = float(np.mean(abs(d['y'][mask]-d['previous'][mask])))
                    new = float(np.mean(abs(d['y'][mask]-p[mask])))
                    ratios.append(new/old);maes.append(new);parent_maes.append(old)
                grid.append(dict(candidate=candidate, alpha=alpha, ratios=ratios,
                                 worst_ratio=max(ratios), maes=maes, parent_maes=parent_maes))
        records[branch] = grid
        selected[branch] = choose_branch(grid)
    selection = dict(branches=selected, windows=WINDOWS,
        rule='Each window MAE improves >=0.5%; minimize worst relative MAE; within 0.5% prefer smaller alpha',
        grid=records, historical_reused=True, confirmation_not_used_for_weights=True,
        parent_selection_sha256=checksum(V2/'selection.json'),
        config_sha256=checksum(OUT/'config.json'),
        candidate_ensembles=ENSEMBLES,
        source_sha256={f:checksum(ROOT/f) for f in ['select_hedonic.py','train_hedonic.py','spatial_prior.py']})
    for fold in WINDOWS:
        d = data[fold]
        p = apply(selection,d['previous'],d['seen'],d['candidates'])
        np.save(OUT/fold/'selected_predictions.npy',p)
        selection[f'{fold}_mae'] = dict(v2=float(np.mean(abs(d['y']-d['previous']))),
                                       candidate=float(np.mean(abs(d['y']-p))))
    dump(path,selection)
    print(json.dumps({k:v for k,v in selection.items() if k!='grid'},ensure_ascii=False,indent=2))


def apply(selection, previous, seen, candidates):
    p = previous.copy()
    for branch,mask in [('seen',seen),('unseen',~seen)]:
        choice=selection['branches'][branch];alpha=choice['alpha']
        if alpha:
            p[mask] = (1-alpha)*previous[mask]+alpha*candidates[choice['candidate']][mask]
    return p


def selected_names(selection):
    names=set()
    for choice in selection['branches'].values():
        if choice['alpha']:
            name=choice['candidate'].split('@')[0]
            names.update(members(name))
    return sorted(names)


def selected_arrays(fold, selection):
    arrays={};raw={}
    for name in selected_names(selection):
        d,_=inputs(fold,name)
        raw[name]=(np.load(OUT/fold/f'{name}_predictions.npy'), d['vx'].time_index.to_numpy(float))
    for choice in selection['branches'].values():
        if not choice['alpha']:continue
        candidate=choice['candidate'];name,damping=candidate.split('@');damping=float(damping)
        names=members(name)
        arrays[candidate]=np.mean([raw[n][0]*np.exp((damping-1)*raw[n][1]) for n in names],axis=0)
    return arrays


def evaluate():
    selection=json.loads((OUT/'selection.json').read_text())
    d=base_data('confirmation');old=baseline('confirmation',d)
    p=apply(selection,old,d['seen'],selected_arrays('confirmation',selection))
    report=dict(v2=scores(d['vy'],old),candidate=scores(d['vy'],p),groups={},
        uncertainty=clustered_comparison(d['vy'],p,old,d['groups']),
        historical_reused=True,selection_frozen_before_confirmation_inference=True)
    for key,mask in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
        report['groups'][key]=dict(rows=int(mask.sum()),v2=scores(d['vy'][mask],old[mask]),
                                   candidate=scores(d['vy'][mask],p[mask]))
    np.savez_compressed(OUT/'confirmation/selected_predictions.npz',y=d['vy'],previous=old,predictions=p)
    dump(OUT/'confirmation/evaluation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def export():
    selection=json.loads((OUT/'selection.json').read_text())
    for f,h in selection['source_sha256'].items():
        if checksum(ROOT/f)!=h:raise ValueError('Frozen source changed: '+f)
    assert checksum(V2/'selection.json')==selection['parent_selection_sha256']
    assert checksum(OUT/'config.json')==selection['config_sha256']
    config=json.loads((OUT/'config.json').read_text())
    for f,h in config['source_sha256'].items():
        if checksum(ROOT/f)!=h:raise ValueError('Training source changed: '+f)
    assert checksum(SPATIAL/'config.json')==config['spatial_config_sha256']
    parent=json.loads((SPATIAL/'config.json').read_text())
    for name,h in parent['data_sha256'].items():
        if checksum(ROOT/'data'/f'{name}.parquet')!=h:raise ValueError('Data changed: '+name)
    if not selected_names(selection):
        raise ValueError('No expert passed the two-window rule; keep v2 instead of relabeling it')
    d=base_data('final');old=baseline('final',d)
    arrays=selected_arrays('final',selection)
    p=apply(selection,old,d['seen'],arrays)
    # v2 is already independently reproduced from raw test in its archived run.
    saved=pd.read_csv(ROOT/'submission1017_v2.csv')
    np.testing.assert_array_equal(np.rint(old).astype(np.int64),saved['總價元'].to_numpy())
    test=pd.read_parquet(ROOT/'data/test.parquet')
    raw,_,_,area=enhanced_features(test)
    prep=joblib.load(V2/'final/preprocess.joblib')
    xx=prep['embedding'].transform(raw)
    xx['time_index']=np.log(prep['index'].factor(raw)).astype(np.float32)
    reproduced={}
    for name in selected_names(selection):
        spec=SPECS[name]
        if spec['area']=='net' or spec['spatial']:
            spatial=joblib.load(SPATIAL/'final'/f"{spec['area']}_preprocess.joblib")
            xnew=pd.concat([xx,spatial['prior'].transform(xx)],axis=1)
            aa=effective_area(xx,spec['area'])
            factor=spatial['index'].factor(xx)
            xnew['effective_area']=aa.astype(np.float32)
            xnew['time_index']=np.log(factor).astype(np.float32)
            base=aa*factor
        else:
            xnew=xx;base=area*prep['index'].factor(raw)
        model=joblib.load(OUT/'final'/f'{name}.joblib')
        actual=predict(model,xnew,base*model.get('input_base_multiplier',1.))
        np.testing.assert_allclose(actual,np.load(OUT/'final'/f'{name}_predictions.npy'),rtol=1e-12,atol=1e-6)
        reproduced[name]=(actual,xnew.time_index.to_numpy(float))
    checked={}
    for choice in selection['branches'].values():
        if not choice['alpha']:continue
        candidate=choice['candidate'];name,damping=candidate.split('@');damping=float(damping)
        names=members(name)
        checked[candidate]=np.mean([reproduced[n][0]*np.exp((damping-1)*reproduced[n][1]) for n in names],axis=0)
    np.testing.assert_array_equal(np.rint(apply(selection,old,d['seen'],checked)),np.rint(p))
    frame=export_submission(test,p,OUT/'submission1017_v3.csv')
    shutil.copy2(OUT/'submission1017_v3.csv',ROOT/'submission1017_v3.csv')
    verify=dict(rows=len(frame),columns=list(frame),ids_unique=bool(frame['編號'].is_unique),
                positive_integer_prices=bool((frame['總價元']>0).all()),
                saved_model_inference_reproduced_from_raw_test=True,baseline_matches_submitted_v2=True,
                frozen_source_and_training_test_data_hashes_verified=True,
                sha256=checksum(ROOT/'submission1017_v3.csv'),submitted=False,public_mae=None)
    dump(OUT/'verification.json',verify)
    print(json.dumps(verify,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate','export'])
    {'freeze':freeze,'evaluate':evaluate,'export':export}[parser.parse_args().stage]()
