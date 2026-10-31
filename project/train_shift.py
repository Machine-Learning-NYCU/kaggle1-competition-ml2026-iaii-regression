"""Recency and project-balance experts, using competition training labels only."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train_hedonic import ROOT, inputs as hedonic_inputs, predict
from train_robust import data as raw_training
from train_spatial import base_data
from features import normalize
from train import scores
from train_mae import dump, checksum, save_model

OUT=ROOT/'artifacts/shift_v1'
SPECS={
    'semantic_full':dict(window=None,decay=None,balance=False),
    'recent24':dict(window=24,decay=None,balance=False),
    'decay18':dict(window=None,decay=18,balance=False),
    'balanced18':dict(window=None,decay=18,balance=True),
}
PATTERNS={
    'leasehold':'地上權', 'relocation':'安置住宅', 'subsidy':'差額減收|補貼|減免',
    'bare_shell':'毛胚|毛坯', 'equipment':'傢俱|家具|家電|設備費',
    'terrace':'露台|露臺', 'canopy':'雨遮', 'farm_extra':'農地',
    'employee':'員工', 'contract_transfer':'讓渡|換約', 'parking_exclusive':'約定專用車位',
}


def semantic_features(raw):
    text=raw['備註'].map(normalize)
    return pd.DataFrame({f'sem_{name}':text.str.contains(pattern).astype(np.float32)
                         for name,pattern in PATTERNS.items()})


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    config=dict(specs=SPECS,patterns=PATTERNS,depth=8,learning_rate=.045,max_iterations=5000,
        source_sha256={f:checksum(ROOT/f) for f in ['train_shift.py','train_hedonic.py','features.py','robust_features.py','spatial_prior.py']},
        data_sha256={f:checksum(ROOT/'data'/f'{f}.parquet') for f in ['train','test']},
        parent_selection_sha256=checksum(ROOT/'artifacts/hedonic_v1/selection.json'),
        external_data_used=False,early_stop='2022 development first four months',
        validation='Reused historical windows; no claim of pristine validation')
    p=OUT/'config.json'
    if p.exists() and json.loads(p.read_text())!=config:raise ValueError('Versioned configuration changed')
    dump(p,config)
    if not (OUT/'semantics.joblib').exists():
        raw,_,_=raw_training();test=pd.read_parquet(ROOT/'data/test.parquet')
        save_model(OUT/'semantics.joblib',dict(train=semantic_features(raw),test=semantic_features(test)))


def inputs(fold):
    path=OUT/fold;path.mkdir(exist_ok=True)
    if (path/'data.joblib').exists():return joblib.load(path/'data.joblib')
    d,cols=hedonic_inputs(fold,'net_log')
    sem=joblib.load(OUT/'semantics.joblib')
    xx=pd.concat([d['x'][cols],sem['train'].iloc[d['fit_ix']].reset_index(drop=True)],axis=1)
    extra=sem['test'] if fold=='final' else sem['train'].iloc[d['valid_ix']].reset_index(drop=True)
    vx=pd.concat([d['vx'][cols],extra],axis=1)
    d=dict(d,x=xx,vx=vx)
    save_model(path/'data.joblib',d)
    return d


def fit(name,d,iterations=None):
    spec=SPECS[name];x=d['x'];y=d['y'];base=d['base']
    month=x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
    age=month.max()-month
    keep=np.ones(len(x),bool) if spec['window'] is None else age<spec['window']
    xx=x.loc[keep];yy=y[keep];b=base[keep]
    weight=np.clip(np.sqrt(yy/1e7),.2,6)
    if spec['decay'] is not None:weight*=np.maximum(.03,np.exp(-age[keep]/spec['decay']))
    if spec['balance']:
        group=xx.canonical_project
        counts=group.map(group.value_counts()).to_numpy(float)
        weight*=np.minimum(1,np.sqrt(100/np.maximum(counts,1)))
    model=CatBoostRegressor(iterations=iterations or 5000,depth=8,learning_rate=.045,
        loss_function='RMSE',l2_leaf_reg=5,random_strength=.5,
        bootstrap_type='Bayesian',bagging_temperature=.5,max_ctr_complexity=2,
        one_hot_max_size=10,has_time=False,task_type='GPU',devices='0',thread_count=16,
        random_seed=3017,verbose=500,allow_writing_files=False)
    kw={}
    if iterations is None:
        m=d['tune'];vy=d['vy'][m]
        kw=dict(eval_set=Pool(d['vx'].loc[m],np.log(vy/d['vbase'][m]),
            cat_features=d['cats'],weight=np.clip(np.sqrt(vy/1e7),.2,6)),early_stopping_rounds=300)
    model.fit(Pool(xx,np.log(yy/b),cat_features=d['cats'],weight=weight),**kw)
    return dict(model=model,columns=list(x),iterations=model.tree_count_,name=name,
                fit_rows=int(keep.sum()),spec=spec)


def train(fold,names):
    prepare();d=inputs(fold);path=OUT/fold
    for name in names:
        if (path/f'{name}.json').exists():continue
        rounds=None if fold=='development' else json.loads((OUT/'development'/f'{name}.json').read_text())['iterations']
        start=time.time();p=path/f'{name}.joblib'
        if p.exists():model=joblib.load(p)
        else:
            model=fit(name,d,rounds);save_model(p,model)
        prediction=predict(model,d['vx'],d['vbase'])
        np.save(path/f'{name}_predictions.npy',prediction)
        record=dict(iterations=model['iterations'],fit_rows=model['fit_rows'],seconds=time.time()-start)
        if d['vy'] is not None:
            record['all']=scores(d['vy'],prediction)
            for key,m in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
                record[key]=scores(d['vy'][m],prediction[m])
        dump(path/f'{name}.json',record);print('DONE',fold,name,record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',default='development')
    p.add_argument('--models',nargs='+',choices=list(SPECS),default=list(SPECS));args=p.parse_args()
    train(args.fold,args.models)
