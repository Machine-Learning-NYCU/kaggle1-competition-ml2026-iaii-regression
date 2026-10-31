"""Year-ahead backtests and frozen full-data refit after leaderboard feedback."""
import argparse
from pathlib import Path
import json
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

from features import roc_date, TARGET, CategoryEncoder
from market import CausalMarket
from robust_features import enhanced_features, AddressEmbedding, RepeatProjectIndex
from train import subset, scores, predict as old_predict, fit_model as old_fit, SPECS as OLD_SPECS, blend
from train_mae import save_model, dump, checksum

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/robust_v1'
FOLDS={'development':('2022-08-10','2023-08-10'),
       'confirmation':('2023-08-10','2024-08-10'),
       'final':('2024-08-10',None)}
NEW_NAMES=['cat_unit','cat_log','lgb_unit']


def data():
    raw=pd.read_parquet(ROOT/'data/train.parquet')
    dates=roc_date(raw['交易年月日']); y=pd.to_numeric(raw[TARGET],errors='coerce')
    keep=(y>0)&dates.between('2000-01-01','2024-08-10')
    raw=raw.loc[keep].assign(_date=dates.loc[keep]).sort_values('_date',kind='stable').drop(columns='_date').reset_index(drop=True)
    return raw,pd.to_numeric(raw[TARGET]).to_numpy(float),roc_date(raw['交易年月日'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    config={'folds':FOLDS,'new_models':NEW_NAMES,'iterations_max':5000,
            'source_sha256':{f:checksum(ROOT/f) for f in ['robust_features.py','train_robust.py','features.py','market.py','train.py']},
            'data_sha256':{f:checksum(ROOT/'data'/f'{f}.parquet') for f in ['train','test']},
            'early_stop':'First four months of development forecast only',
            'selection':'Development year only; confirmation is reused historical data with a new cutoff',
            'feedback_score':2393304.6257}
    config=json.loads(json.dumps(config))
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:
        raise ValueError('Changed experiment; preserve versioned configuration')
    dump(OUT/'config.json',config)
    if not (OUT/'features.joblib').exists():
        raw,y,dates=data()
        base=joblib.load(ROOT/'artifacts/run_v2/features.joblib')
        x,c,g,a=enhanced_features(raw,base)
        save_model(OUT/'features.joblib',(x,c,g,a,y,dates))
    x,c,g,a,y,dates=joblib.load(OUT/'features.joblib')
    for fold,(cutoff,end) in FOLDS.items():
        path=OUT/fold;path.mkdir(exist_ok=True)
        if (path/'data.joblib').exists():
            continue
        fit=np.where(dates<=cutoff)[0]
        xx=subset(x,fit)
        if fold=='final':
            test=pd.read_parquet(ROOT/'data/test.parquet')
            vx,_,vg,va=enhanced_features(test)
            vy=None;vi=None
        else:
            vi=np.where((dates>cutoff)&(dates<=end))[0]
            vx,vg,va,vy=subset(x,vi),g[vi],a[vi],y[vi]
        embedding=AddressEmbedding().fit(xx)
        index=RepeatProjectIndex().fit(xx,y[fit],a[fit])
        bx=a[fit]*index.factor(xx);bv=va*index.factor(vx)
        xt,vt=embedding.transform(xx),embedding.transform(vx)
        xt['time_index']=np.log(index.factor(xx)).astype(np.float32)
        vt['time_index']=np.log(index.factor(vx)).astype(np.float32)
        seen=pd.Series(vg).isin(set(g[fit])).to_numpy()
        tune=np.ones(len(vx),dtype=bool) if fold=='final' else (dates.iloc[vi]<pd.Timestamp(cutoff)+pd.DateOffset(months=4)).to_numpy()
        save_model(path/'preprocess.joblib',{'embedding':embedding,'index':index,'cats':c})
        save_model(path/'data.joblib',{'x':xt,'y':y[fit],'base':bx,'vx':vt,'vy':vy,'vbase':bv,
                                      'cats':c,'fit_ix':fit,'valid_ix':vi,'groups':vg,'seen':seen,'tune':tune})
        dump(path/'manifest.json',{'fit_rows':len(fit),'forecast_rows':len(vx),'seen_group_fraction':float(seen.mean()),
                                  'cutoff':cutoff,'forecast_end':end,'city_monthly_index_rates':index.rates})
        print('PREPARED',fold,len(fit),len(vx),'seen',seen.mean(),'rates',index.rates,flush=True)


def fit_new(name,d,iterations=None):
    x,y,b=d['x'],d['y'],d['base'];cats=d['cats']
    val=iterations is None
    if name=='cat_log':
        target=np.log(y/b)
        weight=np.clip(np.sqrt(y/1e7),.2,6.)
    else:
        target=y/b/1e5
        weight=b/100
    encoder=None
    if name.startswith('cat'):
        model=CatBoostRegressor(iterations=iterations or 5000,depth=8,learning_rate=.055,
            loss_function='RMSE' if name=='cat_log' else 'MAE',l2_leaf_reg=5,
            random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=.5,
            max_ctr_complexity=2,one_hot_max_size=10,has_time=False,
            task_type='GPU',devices='0',thread_count=16,random_seed=1017,
            verbose=500,allow_writing_files=False)
        kw={}
        if val:
            mask=d['tune']; vx=d['vx'].loc[mask]; vy=d['vy'][mask];vb=d['vbase'][mask]
            vt=np.log(vy/vb) if name=='cat_log' else vy/vb/1e5
            vw=np.clip(np.sqrt(vy/1e7),.2,6.) if name=='cat_log' else vb/100
            kw={'eval_set':Pool(vx,vt,cat_features=cats,weight=vw),'early_stopping_rounds':300}
        model.fit(Pool(x,target,cat_features=cats,weight=weight),**kw)
        best=model.tree_count_
    else:
        encoder=CategoryEncoder().fit(x,cats)
        model=lgb.LGBMRegressor(n_estimators=iterations or 5000,learning_rate=.04,
            num_leaves=63,min_child_samples=40,reg_lambda=5,reg_alpha=.05,
            colsample_bytree=.85,subsample=.85,subsample_freq=1,
            cat_smooth=20,cat_l2=10,max_cat_threshold=64,
            objective='regression_l1',n_jobs=16,random_state=1017,
            verbosity=-1,deterministic=True,force_col_wise=True)
        kw={}
        if val:
            mask=d['tune']
            kw={'eval_X':encoder.transform(d['vx'].loc[mask]),'eval_y':d['vy'][mask]/d['vbase'][mask]/1e5,
                'eval_sample_weight':[d['vbase'][mask]/100],
                'callbacks':[lgb.early_stopping(300,verbose=False),lgb.log_evaluation(500)]}
        model.fit(encoder.transform(x),target,sample_weight=weight,**kw)
        best=model.best_iteration_ or model.n_estimators_
    return {'name':name,'model':model,'encoder':encoder,'iterations':int(best)}


def predict_new(model,x,b):
    xx=model['encoder'].transform(x) if model['encoder'] is not None else x
    z=model['model'].predict(xx)
    p=np.exp(np.clip(z,-10,20))*b if model['name']=='cat_log' else z*b*1e5
    return np.maximum(p,1)


def train_new(fold,name):
    path=OUT/fold
    if (path/f'{name}_predictions.npy').exists():
        return
    d=joblib.load(path/'data.joblib')
    iterations=None if fold=='development' else json.loads((OUT/'development'/f'{name}.json').read_text())['iterations']
    start=time.time()
    if (path/f'{name}.joblib').exists():
        model=joblib.load(path/f'{name}.joblib')
    else:
        model=fit_new(name,d,iterations)
        save_model(path/f'{name}.joblib',model)
    p=predict_new(model,d['vx'],d['vbase'])
    np.save(path/f'{name}_predictions.npy',p)
    record={'iterations':model['iterations'],'seconds':time.time()-start}
    if d['vy'] is not None:
        record['all']=scores(d['vy'],p)
        for key,mask in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
            record[key]=scores(d['vy'][mask],p[mask])
    dump(path/f'{name}.json',record)
    print('DONE',fold,name,record,flush=True)


def old_data(fold):
    path=OUT/fold/'legacy_data.joblib'
    if path.exists():return joblib.load(path)
    d=joblib.load(OUT/fold/'data.joblib')
    x,c,g,o=joblib.load(ROOT/'artifacts/run_v2/features.joblib')
    fit,vi=d['fit_ix'],d['valid_ix']
    market=CausalMarket().fit(subset(x,fit),np.log(d['y'])-o[fit])
    record={'x':market.transform(subset(x,fit)),'y':d['y'],'o':o[fit],'c':c,
            'vx':market.transform(subset(x,vi)),'vo':o[vi],'vy':d['vy']}
    save_model(path,record)
    return record


def train_old(fold,name):
    path=OUT/fold
    if (path/f'{name}_predictions.npy').exists():return
    frozen=json.loads((ROOT/'artifacts/run_v2/frozen_blend.json').read_text())
    if fold=='final':
        archive=np.load(ROOT/'artifacts/run_v2/test_base_predictions.npz')['predictions']
        np.save(path/f'{name}_predictions.npy',archive[:,list(OLD_SPECS).index(name)])
        return
    d=old_data(fold)
    if (path/f'{name}.joblib').exists():model=joblib.load(path/f'{name}.joblib')
    else:
        model=old_fit(OLD_SPECS[name],d['x'],d['y'],d['o'],d['c'],
                      SimpleNamespace(iterations=1800,device='GPU',threads=16),
                      iterations=frozen['records'][name]['iterations'])
        save_model(path/f'{name}.joblib',model)
    p=old_predict(model,d['vx'],d['vo'])
    np.save(path/f'{name}_predictions.npy',p)
    dump(path/f'{name}.json',{'all':scores(d['vy'],p),'iterations':model['iterations']})
    print('BASELINE',fold,name,scores(d['vy'],p),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--fold',choices=list(FOLDS),default='development')
    p.add_argument('--models',nargs='+',default=NEW_NAMES)
    args=p.parse_args()
    if args.prepare:prepare();return
    for name in args.models:
        if name in NEW_NAMES:train_new(args.fold,name)
        else:train_old(args.fold,name)


if __name__=='__main__':main()
