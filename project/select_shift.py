"""Select recency experts against submitted v3 on two historical windows."""
import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from train_shift import ROOT, OUT, SPECS, inputs, semantic_features
from train_hedonic import predict
from train_spatial import base_data
from select_hedonic import choose_branch, apply, baseline as v2_predictions
from select_hedonic import selected_arrays as v3_arrays
from train_mae import checksum, dump, clustered_comparison, export_submission
from train import scores
from robust_features import enhanced_features
from spatial_prior import effective_area
from train_transfer import OUT as TRANSFER, predict as transfer_predict

PARENT=ROOT/'artifacts/hedonic_v1'
WINDOWS=['stress2021','development']


def parent_predictions(fold):
    if fold=='confirmation':
        return np.load(PARENT/fold/'selected_predictions.npz')['predictions']
    if fold!='final':return np.load(PARENT/fold/'selected_predictions.npy')
    d=base_data(fold);sel=json.loads((PARENT/'selection.json').read_text())
    result=apply(sel,v2_predictions(fold,d),d['seen'],v3_arrays(fold,sel))
    saved=pd.read_csv(ROOT/'submission1017_v3.csv')
    np.testing.assert_array_equal(np.rint(result).astype(np.int64),saved['總價元'].to_numpy())
    return result


def candidate_arrays(fold,names=None):
    d=inputs(fold);t=d['vx'].time_index.to_numpy(float)
    return {f'{name}@{damping}':np.load((TRANSFER if name=='transfer' else OUT)/fold/f'{name}_predictions.npy')*np.exp((damping-1)*t)
            for name in (list(SPECS)+['transfer'] if names is None else names) for damping in [0.,.5,1.]}


def selected_names(selection):
    return sorted({c['candidate'].split('@')[0] for c in selection['branches'].values() if c['alpha']})


def freeze():
    path=OUT/'selection.json'
    if path.exists():print(path.read_text());return
    data={}
    for fold in WINDOWS:
        d=inputs(fold)
        data[fold]=dict(y=d['vy'],seen=d['seen'],previous=parent_predictions(fold),candidates=candidate_arrays(fold))
    branches={};grids={}
    for branch in ['seen','unseen']:
        grid=[]
        for candidate in data[WINDOWS[0]]['candidates']:
            for alpha in [.25,.5,.75]:
                ratios=[];maes=[];parent_maes=[]
                for fold in WINDOWS:
                    d=data[fold];m=d['seen'] if branch=='seen' else ~d['seen']
                    p=(1-alpha)*d['previous']+alpha*d['candidates'][candidate]
                    old=float(np.mean(abs(d['y'][m]-d['previous'][m])))
                    new=float(np.mean(abs(d['y'][m]-p[m])))
                    ratios.append(new/old);maes.append(new);parent_maes.append(old)
                grid.append(dict(candidate=candidate,alpha=alpha,ratios=ratios,maes=maes,
                    parent_maes=parent_maes,worst_ratio=max(ratios)))
        grids[branch]=grid;branches[branch]=choose_branch(grid)
    sel=dict(branches=branches,grid=grids,windows=WINDOWS,
        rule='Each window branch MAE improves >=0.5%; minimize worst relative MAE; within 0.5% prefer smaller alpha',
        historical_reused=True,confirmation_not_used_for_weights=True,
        external_data_used=False,config_sha256=checksum(OUT/'config.json'),
        transfer_config_sha256=checksum(TRANSFER/'config.json'),
        parent_selection_sha256=checksum(PARENT/'selection.json'),
        parent_submission_sha256=checksum(ROOT/'submission1017_v3.csv'),
        source_sha256={f:checksum(ROOT/f) for f in ['select_shift.py','train_shift.py','train_transfer.py','select_hedonic.py']})
    for fold,d in data.items():
        p=apply(sel,d['previous'],d['seen'],d['candidates'])
        np.save(OUT/fold/'selected_predictions.npy',p)
        sel[f'{fold}_mae']=dict(v3=float(np.mean(abs(d['y']-d['previous']))),
                                  candidate=float(np.mean(abs(d['y']-p))))
    dump(path,sel)
    print(json.dumps({k:v for k,v in sel.items() if k!='grid'},ensure_ascii=False,indent=2))


def evaluate():
    sel=json.loads((OUT/'selection.json').read_text());d=inputs('confirmation')
    old=parent_predictions('confirmation')
    p=apply(sel,old,d['seen'],candidate_arrays('confirmation',selected_names(sel)))
    report=dict(v3=scores(d['vy'],old),candidate=scores(d['vy'],p),groups={},
        uncertainty=clustered_comparison(d['vy'],p,old,d['groups']),
        historical_reused=True,selection_frozen_before_confirmation_inference=True)
    for key,m in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
        report['groups'][key]=dict(rows=int(m.sum()),v3=scores(d['vy'][m],old[m]),candidate=scores(d['vy'][m],p[m]))
    np.savez_compressed(OUT/'confirmation/selected_predictions.npz',y=d['vy'],previous=old,predictions=p)
    dump(OUT/'confirmation/evaluation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def export():
    sel=json.loads((OUT/'selection.json').read_text())
    if not selected_names(sel):raise ValueError('No branch qualified: keep v3, do not relabel it')
    report=json.loads((OUT/'confirmation/evaluation.json').read_text())
    if report['candidate']['mae']>=report['v3']['mae']:raise ValueError('Confirmation regressed: do not export')
    assert checksum(OUT/'config.json')==sel['config_sha256']
    assert checksum(TRANSFER/'config.json')==sel['transfer_config_sha256']
    assert checksum(PARENT/'selection.json')==sel['parent_selection_sha256']
    assert checksum(ROOT/'submission1017_v3.csv')==sel['parent_submission_sha256']
    config=json.loads((OUT/'config.json').read_text())
    transfer_config=json.loads((TRANSFER/'config.json').read_text())
    for f,h in {**config['source_sha256'],**transfer_config['source_sha256'],**sel['source_sha256']}.items():
        if checksum(ROOT/f)!=h:raise ValueError('Frozen source changed: '+f)
    for f,h in config['data_sha256'].items():
        if checksum(ROOT/'data'/f'{f}.parquet')!=h:raise ValueError('Data changed: '+f)
    d=inputs('final');old=parent_predictions('final');names=selected_names(sel)
    p=apply(sel,old,d['seen'],candidate_arrays('final',names))
    # Independently reconstruct the added experts' inputs from raw test.
    test=pd.read_parquet(ROOT/'data/test.parquet')
    raw,_,_,_=enhanced_features(test)
    prep=joblib.load(ROOT/'artifacts/robust_v1/final/preprocess.joblib')
    xx=prep['embedding'].transform(raw)
    net=joblib.load(ROOT/'artifacts/spatial_v1/final/net_preprocess.joblib')
    area=effective_area(xx,'net');factor=net['index'].factor(xx)
    xx['effective_area']=area.astype(np.float32)
    xx['time_index']=np.log(factor).astype(np.float32)
    xx=pd.concat([xx,semantic_features(test)],axis=1)
    checked={}
    for name in names:
        model_path=TRANSFER if name=='transfer' else OUT
        model=joblib.load(model_path/'final'/f'{name}.joblib')
        actual=(transfer_predict if name=='transfer' else predict)(model,xx,area*factor)
        np.testing.assert_allclose(actual,np.load(model_path/'final'/f'{name}_predictions.npy'),rtol=1e-12,atol=1e-6)
        for damping in [0.,.5,1.]:checked[f'{name}@{damping}']=actual*np.exp((damping-1)*xx.time_index.to_numpy(float))
    np.testing.assert_array_equal(np.rint(apply(sel,old,d['seen'],checked)),np.rint(p))
    frame=export_submission(test,p,OUT/'submission1017_v4.csv')
    shutil.copy2(OUT/'submission1017_v4.csv',ROOT/'submission1017_v4.csv')
    verify=dict(rows=len(frame),columns=list(frame),ids_unique=bool(frame['編號'].is_unique),
        positive_integer_prices=bool((frame['總價元']>0).all()),
        saved_model_inference_reproduced_from_raw_test=True,baseline_matches_submitted_v3=True,
        frozen_source_and_training_test_data_hashes_verified=True,external_data_used=False,
        sha256=checksum(ROOT/'submission1017_v4.csv'),submitted=False,public_mae=None)
    dump(OUT/'verification.json',verify);print(json.dumps(verify,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate','export'])
    {'freeze':freeze,'evaluate':evaluate,'export':export}[parser.parse_args().stage]()
