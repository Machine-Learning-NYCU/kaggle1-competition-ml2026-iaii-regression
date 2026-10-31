"""Center log-unit targets so small-delta Huber starts near the data.

This is a separately versioned numerical correction, not a change to frozen
hedonic_v1 models. The original uncentered attempt remains archived.
"""
import argparse
import json
import time
import joblib
import numpy as np
from train_hedonic import ROOT, OUT, inputs as original_inputs, fit, predict
from train_mae import dump, checksum, save_model
from train import scores

CENTER = 150000.


def inputs(fold):
    d,cols=original_inputs(fold,'net_huber')
    d=dict(d,base=d['base']*CENTER,vbase=d['vbase']*CENTER)
    return d,cols


def train(fold):
    config=dict(center=CENTER,source_sha256={'train_centered.py':checksum(ROOT/'train_centered.py')},
                parent_config_sha256=checksum(OUT/'config.json'),
                change='Fixed target origin for Huber, not learned from evaluation labels')
    config_path=OUT/'centered_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Centered experiment changed')
    dump(config_path,config)
    path=OUT/fold;path.mkdir(exist_ok=True);name='centered_huber'
    if (path/f'{name}.json').exists():return
    d,cols=inputs(fold)
    rounds=None if fold=='development' else json.loads((OUT/'development'/f'{name}.json').read_text())['iterations']
    start=time.time();model_path=path/f'{name}.joblib'
    if model_path.exists():model=joblib.load(model_path)
    else:
        model=fit('net_huber',d,cols,rounds)
        model['input_base_multiplier']=CENTER
        save_model(model_path,model)
    p=predict(model,d['vx'],d['vbase']);np.save(path/f'{name}_predictions.npy',p)
    report=dict(iterations=model['iterations'],seconds=time.time()-start)
    if d['vy'] is not None:
        report['all']=scores(d['vy'],p)
        for key,mask in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
            report[key]=scores(d['vy'][mask],p[mask])
    dump(path/f'{name}.json',report);print('DONE',fold,name,report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',default='development')
    train(parser.parse_args().fold)
