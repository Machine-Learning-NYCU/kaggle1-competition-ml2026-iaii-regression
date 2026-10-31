"""New-project generalization experiments with group-excluded local priors."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import joblib,numpy as np,pandas as pd
from catboost import CatBoostRegressor,Pool
from robust_features import AddressEmbedding,RepeatProjectIndex
from spatial_prior import CrossProjectPrior,effective_area
from train import subset,scores,fit_model as old_fit,predict as old_predict,SPECS as OLD_SPECS
from train_robust import fit_new as v2_fit,predict_new as v2_predict
from train_mae import dump,save_model,checksum
from market import CausalMarket

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/spatial_v1'
V2=ROOT/'artifacts/robust_v1'
SPECS={
 'cold_total':{'area':'total','identities':False,'loss':'RMSE'},
 'cold_net':{'area':'net','identities':False,'loss':'RMSE'},
 'context_net':{'area':'net','identities':True,'loss':'RMSE'},
 'cold_huber':{'area':'net','identities':False,'loss':'Huber:delta=0.10'},
}
IDENTITIES=['project','canonical_project','address_base','tower','project_building_type']


def stress_base():
    path=OUT/'stress2021';path.mkdir(parents=True,exist_ok=True)
    if (path/'base.joblib').exists():return
    x,c,g,a,y,t=joblib.load(V2/'features.joblib')
    fi=np.where(t<='2021-08-10')[0];vi=np.where((t>'2021-08-10')&(t<='2022-08-10'))[0]
    xx,vx=subset(x,fi),subset(x,vi)
    embedding=AddressEmbedding().fit(xx);index=RepeatProjectIndex().fit(xx,y[fi],a[fi])
    bx=a[fi]*index.factor(xx);bv=a[vi]*index.factor(vx)
    xt,vt=embedding.transform(xx),embedding.transform(vx)
    xt['time_index']=np.log(index.factor(xx)).astype(np.float32)
    vt['time_index']=np.log(index.factor(vx)).astype(np.float32)
    d={'x':xt,'y':y[fi],'base':bx,'vx':vt,'vy':y[vi],'vbase':bv,'cats':c,
       'fit_ix':fi,'valid_ix':vi,'groups':g[vi],
       'seen':pd.Series(g[vi]).isin(set(g[fi])).to_numpy(),
       'tune':(t.iloc[vi]<'2021-12-10').to_numpy()}
    save_model(path/'base.joblib',d)
    save_model(path/'base_preprocess.joblib',{'embedding':embedding,'index':index,'cats':c})
    print('STRESS PREPARED',len(fi),len(vi),flush=True)


def base_data(fold):
    return joblib.load(OUT/'stress2021/base.joblib' if fold=='stress2021' else V2/fold/'data.joblib')


def prepare(folds):
    OUT.mkdir(parents=True,exist_ok=True)
    config={'specs':SPECS,'drop_identities':IDENTITIES,'iterations_max':5000,'recency_decay_months':24,
            'source_sha256':{f:checksum(ROOT/f) for f in ['spatial_prior.py','train_spatial.py','robust_features.py','train_robust.py','features.py','market.py']},
            'parent_selection_sha256':checksum(V2/'selection.json'),
            'data_sha256':{f:checksum(ROOT/'data'/f'{f}.parquet') for f in ['train','test']},
            'validation':'Historical selection reused; extra 2021-2022 stress window is not claimed pristine.'}
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:
        raise ValueError('Changed experiment: use a new version')
    dump(OUT/'config.json',config)
    if 'stress2021' in folds:stress_base()
    for fold in folds:
        path=OUT/fold;path.mkdir(exist_ok=True)
        d=base_data(fold)
        for mode in ['total','net']:
            if (path/f'{mode}.joblib').exists():continue
            area=effective_area(d['x'],mode);va=effective_area(d['vx'],mode)
            index=RepeatProjectIndex().fit(d['x'],d['y'],area)
            prior=CrossProjectPrior().fit(d['x'],d['y'],area)
            # These are fitting-only target statistics with complete project exclusion.
            xf=prior.transform(d['x']);vf=prior.transform(d['vx'])
            xx=pd.concat([d['x'],xf],axis=1);vx=pd.concat([d['vx'],vf],axis=1)
            xx['time_index']=np.log(index.factor(xx)).astype(np.float32)
            vx['time_index']=np.log(index.factor(vx)).astype(np.float32)
            xx['effective_area']=area.astype(np.float32);vx['effective_area']=va.astype(np.float32)
            record=dict(d,x=xx,vx=vx,base=area*index.factor(xx),vbase=va*index.factor(vx))
            save_model(path/f'{mode}_preprocess.joblib',{'index':index,'prior':prior,'area':mode})
            save_model(path/f'{mode}.joblib',record)
            print('PREPARED',fold,mode,xx.shape,vx.shape,flush=True)


def columns(d,spec):
    excluded=[] if spec['identities'] else IDENTITIES
    cols=[c for c in d['x'] if c not in excluded]
    cats=[c for c in d['cats'] if c in cols]
    return cols,cats


def fit_model(name,d,iterations=None):
    spec=SPECS[name];cols,cats=columns(d,spec)
    x,y,b=d['x'],d['y'],d['base']
    anchor=x.spatial_anchor.to_numpy(float)
    target=np.log(y/b)-anchor
    month=x.transaction_year.to_numpy()*12+x.transaction_month.to_numpy()
    recency=np.maximum(.05,np.exp(-(month.max()-month)/24))
    if spec['loss']=='RMSE':weight=np.clip(np.sqrt(y/1e7),.2,6)*recency
    else:weight=np.clip(b*np.exp(anchor)/1e7,.2,8)*recency
    model=CatBoostRegressor(iterations=iterations or 5000,depth=8,learning_rate=.045,
        loss_function=spec['loss'],l2_leaf_reg=8,random_strength=.8,
        bootstrap_type='Bayesian',bagging_temperature=.7,max_ctr_complexity=2,
        one_hot_max_size=10,has_time=False,task_type='GPU',devices='0',thread_count=16,
        random_seed=2017,verbose=500,allow_writing_files=False)
    kw={}
    if iterations is None:
        mask=d['tune'];vx=d['vx'].loc[mask];vy=d['vy'][mask];vb=d['vbase'][mask]
        va=vx.spatial_anchor.to_numpy(float);vt=np.log(vy/vb)-va
        vw=np.clip(np.sqrt(vy/1e7),.2,6) if spec['loss']=='RMSE' else np.clip(vb*np.exp(va)/1e7,.2,8)
        kw={'eval_set':Pool(vx[cols],vt,cat_features=cats,weight=vw),'early_stopping_rounds':300}
    model.fit(Pool(x[cols],target,cat_features=cats,weight=weight),**kw)
    return {'name':name,'model':model,'columns':cols,'iterations':model.tree_count_,'spec':spec}


def predict_model(model,x,base):
    z=model['model'].predict(x[model['columns']])+x.spatial_anchor.to_numpy(float)
    return np.maximum(np.exp(np.clip(z,-10,24))*base,1.)


def train(fold,names):
    path=OUT/fold
    for name in names:
        if (path/f'{name}.json').exists():continue
        d=joblib.load(path/f"{SPECS[name]['area']}.joblib")
        iterations=None if fold=='development' else json.loads((OUT/'development'/f'{name}.json').read_text())['iterations']
        start=time.time();model_path=path/f'{name}.joblib'
        if model_path.exists():model=joblib.load(model_path)
        else:
            model=fit_model(name,d,iterations);save_model(model_path,model)
        prediction=predict_model(model,d['vx'],d['vbase'])
        np.save(path/f'{name}_predictions.npy',prediction)
        report={'iterations':model['iterations'],'seconds':time.time()-start}
        if d['vy'] is not None:
            report['all']=scores(d['vy'],prediction)
            for key,mask in [('seen',d['seen']),('unseen',~d['seen']),('early',d['tune']),('later',~d['tune'])]:
                report[key]=scores(d['vy'][mask],prediction[mask])
        dump(path/f'{name}.json',report)
        print('DONE',fold,name,report,flush=True)


def stress_baseline(names):
    path=OUT/'stress2021';d=base_data('stress2021')
    for name in names:
        if (path/f'base_{name}.json').exists():continue
        start=time.time()
        if name in ['cat_unit','cat_log','lgb_unit']:
            iterations=json.loads((V2/'development'/f'{name}.json').read_text())['iterations']
            model=v2_fit(name,d,iterations)
            p=v2_predict(model,d['vx'],d['vbase'])
        else:
            if not (path/'legacy_data.joblib').exists():
                x,c,g,o=joblib.load(ROOT/'artifacts/run_v2/features.joblib')
                fi,vi=d['fit_ix'],d['valid_ix']
                market=CausalMarket().fit(subset(x,fi),np.log(d['y'])-o[fi])
                dd={'x':market.transform(subset(x,fi)),'vx':market.transform(subset(x,vi)),
                    'y':d['y'],'o':o[fi],'vo':o[vi],'c':c}
                save_model(path/'legacy_data.joblib',dd)
            dd=joblib.load(path/'legacy_data.joblib')
            frozen=json.loads((ROOT/'artifacts/run_v2/frozen_blend.json').read_text())
            model=old_fit(OLD_SPECS[name],dd['x'],dd['y'],dd['o'],dd['c'],SimpleNamespace(iterations=1800,device='GPU',threads=16),
                          iterations=frozen['records'][name]['iterations'])
            p=old_predict(model,dd['vx'],dd['vo'])
        save_model(path/f'base_{name}.joblib',model)
        np.save(path/f'base_{name}_predictions.npy',p)
        dump(path/f'base_{name}.json',{'scores':scores(d['vy'],p),'seconds':time.time()-start})
        print('BASELINE',name,scores(d['vy'],p),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','stress_baseline'])
    parser.add_argument('--fold',default='development');parser.add_argument('--folds',nargs='+',default=['development','stress2021','confirmation','final'])
    parser.add_argument('--models',nargs='+',default=list(SPECS));args=parser.parse_args()
    if args.stage=='prepare':prepare(args.folds)
    elif args.stage=='stress_baseline':stress_baseline(args.models)
    else:train(args.fold,args.models)
