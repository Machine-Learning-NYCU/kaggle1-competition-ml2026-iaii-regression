"""A location-transfer expert with project/unit identity features removed."""
import argparse,json,time
import joblib,numpy as np
import lightgbm as lgb
from train_shift import ROOT, inputs
from train_mae import checksum,dump,save_model
from features import CategoryEncoder
from train import scores

OUT=ROOT/'artifacts/transfer_v1'
DROP=['project','canonical_project','project_building_type','address_base','tower']


def predict(model,x,base):
    z=model['model'].predict(model['encoder'].transform(x[model['columns']]))
    return np.maximum(z*base*1e5,1)


def train(fold):
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(drop=DROP,objective='regression_l1',recency_decay_months=18,
        max_iterations=3000,learning_rate=.04,num_leaves=63,min_child_samples=80,
        source_sha256={f:checksum(ROOT/f) for f in ['train_transfer.py','train_shift.py','features.py']},
        parent_config_sha256=checksum(ROOT/'artifacts/shift_v1/config.json'),external_data_used=False)
    p=OUT/'config.json'
    if p.exists() and json.loads(p.read_text())!=config:raise ValueError('Config changed')
    dump(p,config)
    path=OUT/fold;path.mkdir(exist_ok=True)
    if (path/'transfer.json').exists():return
    d=inputs(fold);cols=[c for c in d['x'] if c not in DROP];x=d['x'][cols]
    if (path/'transfer.joblib').exists():model=joblib.load(path/'transfer.joblib')
    else:
        cats=[c for c in d['cats'] if c in cols];enc=CategoryEncoder().fit(x,cats)
        month=x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        w=d['base']/100*np.maximum(.03,np.exp((month-month.max())/18))
        rounds=3000 if fold=='development' else json.loads((OUT/'development/transfer.json').read_text())['iterations']
        estimator=lgb.LGBMRegressor(n_estimators=rounds,learning_rate=.04,num_leaves=63,
            min_child_samples=80,reg_lambda=5,reg_alpha=.05,colsample_bytree=.85,
            subsample=.85,subsample_freq=1,cat_smooth=20,cat_l2=10,max_cat_threshold=64,
            objective='regression_l1',n_jobs=12,random_state=4017,verbosity=-1,
            deterministic=True,force_col_wise=True)
        kw={}
        if fold=='development':
            m=d['tune'];kw=dict(eval_X=enc.transform(d['vx'].loc[m,cols]),
                eval_y=d['vy'][m]/d['vbase'][m]/1e5,eval_sample_weight=[d['vbase'][m]/100],
                callbacks=[lgb.early_stopping(250,verbose=False),lgb.log_evaluation(500)])
        estimator.fit(enc.transform(x),d['y']/d['base']/1e5,sample_weight=w,**kw)
        model=dict(model=estimator,encoder=enc,columns=cols,iterations=estimator.best_iteration_ or estimator.n_estimators_)
        save_model(path/'transfer.joblib',model)
    pred=predict(model,d['vx'],d['vbase']);np.save(path/'transfer_predictions.npy',pred)
    result=dict(iterations=model['iterations'])
    if d['vy'] is not None:
        result['all']=scores(d['vy'],pred)
        result['unseen']=scores(d['vy'][~d['seen']],pred[~d['seen']])
    dump(path/'transfer.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',default='development');train(p.parse_args().fold)
