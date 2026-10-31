"""Evaluate launch-cohort extrapolation before selecting any new model blend."""
import argparse,json
import joblib,numpy as np
from train_hedonic import ROOT,inputs as old_inputs
from train_spatial import OUT as SPATIAL
from launch_index import LaunchIndex
from spatial_prior import effective_area
from train_mae import checksum,dump,save_model

OUT=ROOT/'artifacts/launch_v1'


def parent_predictions(fold):
    path=ROOT/'artifacts/hedonic_v1'/fold
    if fold=='confirmation':return np.load(path/'selected_predictions.npz')['predictions']
    return np.load(path/'selected_predictions.npy')


def run(folds):
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(source_sha256={f:checksum(ROOT/f) for f in ['launch_index.py','audit_launch.py']},
                months=[24,36],strength=[.25,.5,1.],labels='fitting only; forecast labels only for historical assessment',
                parent_sha256=checksum(ROOT/'artifacts/hedonic_v1/selection.json'))
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:raise ValueError('Config changed')
    dump(OUT/'config.json',config)
    for fold in folds:
        path=OUT/fold;path.mkdir(exist_ok=True)
        d,_=old_inputs(fold,'net_log');area=effective_area(d['x'],'net')
        repeat=joblib.load(SPATIAL/fold/'net_preprocess.joblib')['index']
        p=parent_predictions(fold);y=d['vy'];m=~d['seen'];base=float(np.mean(abs(y-p)));records=[]
        for months in [24,36]:
            index=LaunchIndex(months).fit(d['x'],d['y'],area)
            save_model(path/f'index_{months}.joblib',index)
            for strength in [.25,.5,1.]:
                factor=index.excess_factor(d['vx'],repeat,strength)
                pred=p.copy();pred[m]*=factor[m]
                mae=float(np.mean(abs(y-pred)))
                records.append(dict(months=months,strength=strength,mae=mae,baseline_mae=base,ratio=mae/base,
                                    unseen_mae=float(np.mean(abs(y[m]-pred[m])))))
                np.save(path/f'launch{months}_{strength}_predictions.npy',pred)
            dump(path/f'rates_{months}.json',dict(launch=index.rates,repeat=repeat.rates,projects=int(index.projects),cells=int(index.rows)))
        dump(path/'metrics.json',records);print(fold,json.dumps(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folds',nargs='+',default=['stress2021','development'])
    run(p.parse_args().folds)
