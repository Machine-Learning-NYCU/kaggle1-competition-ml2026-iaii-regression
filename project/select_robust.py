"""Freeze development-year ensemble; evaluate confirmation; export reproducibly."""
import argparse
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd

from comparable_sales import ComparableSales
from mae_models import learn_mae_blend
from train import scores
from train_mae import dump,save_model,checksum,clustered_comparison,export_submission
from train_robust import ROOT,OUT,NEW_NAMES,OLD_SPECS,predict_new
from robust_features import enhanced_features
from features import make_features
from train import predict as legacy_predict


def inputs(fold):
    path=OUT/fold
    d=joblib.load(path/'data.joblib')
    legacy=json.loads((ROOT/'artifacts/mae_final/calibration.json').read_text())
    old=np.column_stack([np.load(path/f'{name}_predictions.npy') for name in OLD_SPECS])
    # The old submitted blend was calibrated on 2024 labels. Keep it as a
    # historical reference ONLY; do not inherit those weights in this pipeline.
    d['reference']=legacy['scale']*(old@np.array(legacy['weights']))
    calibration_path=OUT/'legacy_development_calibration.json'
    if not calibration_path.exists():
        if fold!='development':raise ValueError('Freeze development calibration first')
        weights,meta=learn_mae_blend(old,d['vy'],penalty=.02)
        dump(calibration_path,{'weights':weights.tolist(),'optimization':meta,
                              'labels_used':'2022-08-11 through 2023-08-10 only'})
    calibration=json.loads(calibration_path.read_text())
    old=old@np.array(calibration['weights'])
    matrix=np.column_stack([old]+[np.load(path/f'{name}_predictions.npy') for name in NEW_NAMES])
    if not (path/'comparable_predictions.npz').exists():
        model=ComparableSales().fit(d['x'],d['y'],d['base'])
        pred,confidence=model.predict(d['vx'],d['vbase'])
        save_model(path/'comparables.joblib',model)
        np.savez_compressed(path/'comparable_predictions.npz',predictions=pred,confidence=confidence)
    comp=np.load(path/'comparable_predictions.npz')
    return d,matrix,comp['predictions'],comp['confidence']


def damped_inputs(selection,matrix,comp,future_factor):
    multiplier=np.asarray(future_factor)**(selection.get('trend_damping',1.)-1.)
    adjusted=matrix.copy()
    adjusted[:,1:]*=multiplier[:,None]
    return adjusted,comp*multiplier


def apply_selection(selection,matrix,seen,comp,confidence,future_factor):
    matrix,comp=damped_inputs(selection,matrix,comp,future_factor)
    prediction=np.empty(len(matrix))
    for key,mask in [('seen',seen),('unseen',~seen)]:
        prediction[mask]=matrix[mask]@np.array(selection['weights'][key])
    weight=selection['comparable_fraction']*confidence
    weight[~np.isfinite(comp)]=0
    return prediction*(1-weight)+np.nan_to_num(comp)*weight


def freeze():
    if (OUT/'selection.json').exists():return
    d,matrix,comp,confidence=inputs('development')
    selection={'models':['historical_experts_dev_calibration']+NEW_NAMES,'weights':{},'penalty':.02,
               'calibration_source':'development: predict 2022-08-11 to 2023-08-10 from 2022-08-10',
               'comparable_fraction':0.,'validation_status':'Historical development; includes early-stopping period',
               'confirmation_not_used_for_weights':True,
               'source_sha256':{f:checksum(ROOT/f) for f in ['robust_features.py','train_robust.py','comparable_sales.py','select_robust.py']}}
    selection['inner_calibration_sha256']=checksum(OUT/'legacy_development_calibration.json')
    future_factor=np.exp(d['vx'].time_index.to_numpy(float))
    records=[];candidates=[]
    for damping in [0.,.5,1.]:
        candidate=dict(selection,trend_damping=damping,weights={})
        adjusted,_=damped_inputs(candidate,matrix,comp,future_factor)
        for key,mask in [('seen',d['seen']),('unseen',~d['seen'])]:
            w,meta=learn_mae_blend(adjusted[mask],d['vy'][mask],penalty=.02)
            candidate['weights'][key]=w.tolist()
        for alpha in [0.,.25,.5,.75]:
            variant=dict(candidate,comparable_fraction=alpha)
            p=apply_selection(variant,matrix,d['seen'],comp,confidence,future_factor)
            records.append({'damping':damping,'alpha':alpha,'mae':float(np.mean(abs(d['vy']-p)))})
            candidates.append(variant)
    minimum=min(row['mae'] for row in records)
    # Prefer no comparable correction when its developmental gain is < 0.5%.
    # This threshold was added from development results before reading confirmation.
    eligible=[j for j,row in enumerate(records) if row['mae']<=minimum*1.005]
    best=min(eligible,key=lambda j:(records[j]['alpha'],records[j]['mae']))
    selection=candidates[best]
    selection['bounded_development_grid']=records
    selection['simplicity_rule']='Within 0.5% of minimum development MAE, prefer lower comparable fraction.'
    p=apply_selection(selection,matrix,d['seen'],comp,confidence,future_factor)
    selection['development_metrics']={name:scores(d['vy'],matrix[:,j]) for j,name in enumerate(selection['models'])}
    selection['development_metrics']['previous_mae']=scores(d['vy'],d['reference'])
    selection['development_metrics']['selected']=scores(d['vy'],p)
    dump(OUT/'selection.json',selection)
    np.save(OUT/'development/selected_predictions.npy',p)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


def evaluate():
    selection=json.loads((OUT/'selection.json').read_text())
    d,matrix,comp,confidence=inputs('confirmation')
    p=apply_selection(selection,matrix,d['seen'],comp,confidence,np.exp(d['vx'].time_index.to_numpy(float)))
    report={'models':{name:scores(d['vy'],matrix[:,j]) for j,name in enumerate(selection['models'])},
            'selected':scores(d['vy'],p),'groups':{},
            'uncertainty':clustered_comparison(d['vy'],p,d['reference'],d['groups']),
            'historical_confirmation_not_pristine':True}
    report['models']['previous_mae']=scores(d['vy'],d['reference'])
    for key,mask in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
        report['groups'][key]={'rows':int(mask.sum()),'previous':scores(d['vy'][mask],d['reference'][mask]),
                               'selected':scores(d['vy'][mask],p[mask])}
    np.savez_compressed(OUT/'confirmation/selected_predictions.npz',y=d['vy'],predictions=p,previous=d['reference'])
    dump(OUT/'confirmation/evaluation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def export():
    selection=json.loads((OUT/'selection.json').read_text())
    for name,expected in selection['source_sha256'].items():
        if checksum(ROOT/name)!=expected:
            raise ValueError(f'Selected source changed: {name}')
    assert checksum(OUT/'legacy_development_calibration.json')==selection['inner_calibration_sha256']
    d,matrix,comp,confidence=inputs('final')
    p=apply_selection(selection,matrix,d['seen'],comp,confidence,np.exp(d['vx'].time_index.to_numpy(float)))
    test=pd.read_parquet(ROOT/'data/test.parquet')
    # Independent inference through the actual saved preprocessing and model objects.
    x,_,groups,area=enhanced_features(test)
    prep=joblib.load(OUT/'final/preprocess.joblib')
    base=area*prep['index'].factor(x)
    xx=prep['embedding'].transform(x)
    xx['time_index']=np.log(prep['index'].factor(x)).astype(np.float32)
    recomputed=[]
    for j,name in enumerate(NEW_NAMES,1):
        actual=predict_new(joblib.load(OUT/'final'/f'{name}.joblib'),xx,base)
        np.testing.assert_allclose(actual,matrix[:,j],rtol=1e-12,atol=1e-6)
        recomputed.append(actual)
    actual_comp,actual_conf=joblib.load(OUT/'final/comparables.joblib').predict(xx,base)
    np.testing.assert_allclose(actual_comp,comp,equal_nan=True)
    np.testing.assert_allclose(actual_conf,confidence)
    legacy_path=ROOT/'artifacts/run_v2'
    oldx,_,_,offset=make_features(test)
    oldx=joblib.load(legacy_path/'market_final.joblib').transform(oldx)
    oldpred=np.column_stack([legacy_predict(joblib.load(legacy_path/f'final_{name}.joblib'),oldx,offset)
                            for name in OLD_SPECS])
    oldcal=json.loads((OUT/'legacy_development_calibration.json').read_text())
    oldpred=oldpred@np.array(oldcal['weights'])
    np.testing.assert_allclose(oldpred,matrix[:,0],rtol=1e-12,atol=1e-6)
    _,_,fit_groups,_,_,_=joblib.load(OUT/'features.joblib')
    actual_seen=pd.Series(groups).isin(set(fit_groups)).to_numpy()
    np.testing.assert_array_equal(actual_seen,d['seen'])
    actual=apply_selection(selection,np.column_stack([oldpred]+recomputed),actual_seen,
                           actual_comp,actual_conf,prep['index'].factor(x))
    # time_index is stored float32; full-precision reconstruction can differ by cents.
    np.testing.assert_allclose(actual,p,rtol=2e-8,atol=.1)
    reproduced=apply_selection(selection,np.column_stack([oldpred]+recomputed),actual_seen,
                           actual_comp,actual_conf,np.exp(xx.time_index.to_numpy(float)))
    np.testing.assert_array_equal(np.rint(reproduced),np.rint(p))
    frame=export_submission(test,p,OUT/'submission1017_v2.csv')
    shutil.copy2(OUT/'submission1017_v2.csv',ROOT/'submission1017_v2.csv')
    verify={'rows':len(frame),'columns':list(frame),'ids_unique':bool(frame['編號'].is_unique),
            'model_reproduction':True,'schema_ids_verified_by_export':True,
            'positive_integer_prices':bool((frame['總價元']>0).all()),
            'sha256':checksum(ROOT/'submission1017_v2.csv'),'submitted':False,
            'previous_reported_public_mae':2393304.6257,'new_public_mae':None}
    dump(OUT/'verification.json',verify)
    print(json.dumps(verify,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate','export'])
    {'freeze':freeze,'evaluate':evaluate,'export':export}[parser.parse_args().stage]()
