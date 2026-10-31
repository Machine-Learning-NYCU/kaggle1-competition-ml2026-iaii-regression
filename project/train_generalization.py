"""Fixed factorial study: project identity masking x loss, future cold projects.

Competition train labels only. No external unit-price data, query-label joins,
leaderboard-derived factors, or editing of previously frozen experiments.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import MISSING
from train_shift import inputs
from train_mae import checksum, dump, save_model, export_submission

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/generalization_v1'
IDENTITIES=['project','canonical_project','project_building_type','address_base','tower']
IDENTITY_NUMBERS=['house_number','parcel_number']
SPECS={
    'identity_log': {'mask_fraction':0.,'loss':'RMSE'},
    'masked_log': {'mask_fraction':.35,'loss':'RMSE'},
    'identity_l1': {'mask_fraction':0.,'loss':'MAE'},
    'masked_l1': {'mask_fraction':.35,'loss':'MAE'},
}
WINDOWS=['stress2021','development']
SEED=20261017
SECOND_SEED=20261018
ROUNDS=3000
BASE_SHA='53bde5d7fa8c97d95ab8897c905b38cf01e16cda6a6eea48f8605a5b848aadd6'


def project_keys(x):
    return x.canonical_project.where(x.canonical_project.ne(MISSING),x.geo_district+'|'+x.address_base)


def hash_fraction(key,seed):
    digest=hashlib.blake2b((str(seed)+'|'+str(key)).encode(),digest_size=8).digest()
    return int.from_bytes(digest,'big')/2**64


class ProjectIdentityMasker:
    """Mask all aliases of whole fitting projects; no labels used in the mask."""
    def __init__(self,fraction=.35,seed=SEED):
        self.fraction=fraction;self.seed=seed

    def fit(self,x):
        keys=project_keys(x)
        known=set(keys)
        self.retained={k for k in known if hash_fraction(k,self.seed)>=self.fraction}
        self.statistics={'fit_projects':len(known),'masked_projects':len(known-self.retained),
                         'masked_fit_rows':int((~keys.isin(self.retained)).sum())}
        return self

    def transform(self,x):
        result=x.copy()
        masked=~project_keys(x).isin(self.retained) if self.fraction>0 else np.zeros(len(x),bool)
        for col in IDENTITIES:
            result.loc[masked,col]=MISSING
        for col in IDENTITY_NUMBERS:
            result.loc[masked,col]=np.nan
        result['identity_masked']=np.asarray(masked,dtype=np.float32)
        return result


def protocol():
    OUT.mkdir(exist_ok=True,parents=True)
    sources=['train_generalization.py','test_generalization.py','train_shift.py','features.py','train_mae.py']
    files=[ROOT/'submission1017_v6.csv',ROOT/'artifacts/external_data_policy.json',
           ROOT/'artifacts/policy_learned_v1/model.joblib']
    files += [ROOT/f'artifacts/shift_v1/{fold}/data.joblib' for fold in WINDOWS+['confirmation','final']]
    config={'specs':SPECS,'mask_columns':IDENTITIES+IDENTITY_NUMBERS,'iterations':ROUNDS,
        'depth':8,'learning_rate':.045,'l2_leaf_reg':10,'primary_seed':SEED,'second_seed':SECOND_SEED,
        'trend_damping':.5,'blend_fractions':[.25,.5],
        'new_branch':'strictly new canonical projects; seen and relocation predictions unchanged',
        'selection':'Both years cold MAE improves >=0.5%; neither late-eight-month cold MAE worsens; prefer alpha=.25 within 0.5% of best worst-year ratio',
        'seed_check':'Freeze choice before second seed; average seeds must pass same gates, each seed cold MAE must improve in each year; no runner-up on failure',
        'confirmation':'No retuning; cold MAE improvement >=0.5%, late cold non-regression, group bootstrap improvement lower95>0',
        'early_stopping':False,'rounds_source':'Fixed before training; no new forecast labels used to tune rounds',
        'history':'Historical forecast windows reused, not pristine independent tests',
        'external_data_used_in_new_branch':False,'public_scores_used_to_fit':False,
        'source_sha256':{name:checksum(ROOT/name) for name in sources},
        'input_sha256':{str(f.relative_to(ROOT)):checksum(f) for f in files}}
    p=OUT/'config.json'
    if p.exists() and json.loads(p.read_text())!=config:raise ValueError('Frozen study changed: use a new version')
    if checksum(ROOT/'submission1017_v6.csv')!=BASE_SHA:raise ValueError('Frozen v6 changed')
    dump(p,config)
    (OUT/'source').mkdir(exist_ok=True)
    for name in sources:shutil.copy2(ROOT/name,OUT/'source'/name)
    return config


def baseline(fold):
    if fold in WINDOWS:return np.load(ROOT/f'artifacts/shift_v1/{fold}/selected_predictions.npy')
    if fold=='confirmation':return np.load(ROOT/'artifacts/shift_v1/confirmation/selected_predictions.npz')['predictions']
    raw=pd.read_parquet(ROOT/'data/test.parquet')
    saved=pd.read_csv(ROOT/'submission1017_v6.csv',dtype={'編號':str}).set_index('編號')['總價元']
    return raw['編號'].astype(str).map(saved).to_numpy(float)


def cold_mask(d):
    # Stronger than raw-name inequality: canonical aliases must also be absent.
    cold=~project_keys(d['vx']).isin(set(project_keys(d['x']))).to_numpy()
    return cold & ~d['seen'] & (d['vx'].sem_relocation.to_numpy()==0)


def model_path(fold,name,seed):return OUT/fold/f'{name}_{seed}.joblib'
def pred_path(fold,name,seed):return OUT/fold/f'{name}_{seed}_predictions.npy'


def predict(model,x,base):
    xx=model['masker'].transform(x)
    z=model['model'].predict(xx[model['columns']])
    value=np.exp(np.clip(z,-10,20))*base if model['loss']=='RMSE' else z*base*1e5
    # Fixed carry-forward convention inherited from the v2 time index, not tuned here.
    value=value*np.exp(-.5*x.time_index.to_numpy(float))
    return np.maximum(value,1.)


def metrics(y,old,new,cold,late):
    result={}
    for key,mask in [('all',np.ones(len(y),bool)),('cold',cold),('cold_late',cold&late)]:
        before=float(np.mean(abs(y[mask]-old[mask])));after=float(np.mean(abs(y[mask]-new[mask])))
        result[key]={'rows':int(mask.sum()),'baseline_mae':before,'candidate_mae':after,'ratio':after/before}
    return result


def fit_fold(fold,names,seed=SEED):
    protocol();d=inputs(fold);path=OUT/fold;path.mkdir(exist_ok=True)
    cold=cold_mask(d);old=baseline(fold)
    for name in names:
        output=path/f'{name}_{seed}.json'
        if output.exists():continue
        start=time.time();spec=SPECS[name];mp=model_path(fold,name,seed)
        if mp.exists():model=joblib.load(mp)
        else:
            masker=ProjectIdentityMasker(spec['mask_fraction'],seed).fit(d['x'])
            xx=masker.transform(d['x'])
            if spec['loss']=='RMSE':
                target=np.log(d['y']/d['base']);weight=np.clip(np.sqrt(d['y']/1e7),.2,6)
            else:
                target=d['y']/d['base']/1e5;weight=d['base']/100
            estimator=CatBoostRegressor(iterations=ROUNDS,depth=8,learning_rate=.045,
                loss_function=spec['loss'],l2_leaf_reg=10,random_strength=.5,
                bootstrap_type='Bayesian',bagging_temperature=.5,max_ctr_complexity=2,
                one_hot_max_size=10,has_time=False,task_type='GPU',devices='0',thread_count=12,
                random_seed=seed,verbose=500,allow_writing_files=False)
            estimator.fit(Pool(xx,target,cat_features=d['cats'],weight=weight))
            model={'model':estimator,'masker':masker,'columns':list(xx),'loss':spec['loss'],
                'name':name,'seed':seed,'iterations':ROUNDS,'mask_statistics':masker.statistics}
            save_model(mp,model)
        p=predict(model,d['vx'],d['vbase']);np.save(pred_path(fold,name,seed),p)
        record={'seconds':time.time()-start,'fit_rows':len(d['y']),'cold_rows':int(cold.sum()),
                'mask_statistics':model['mask_statistics'],'iterations':ROUNDS}
        if d['vy'] is not None:
            record['standalone']=metrics(d['vy'],old,p,cold,~d['tune'])
            record['blends']={str(a):metrics(d['vy'],old,np.where(cold,(1-a)*old+a*p,old),cold,~d['tune']) for a in [.25,.5]}
        dump(output,record)
        print('DONE',fold,name,seed,json.dumps(record,ensure_ascii=False),flush=True)


def choose(records):
    eligible=[r for r in records if max(r['ratios'])<=.995 and max(r['late_ratios'])<=1.]
    if not eligible:return None
    best=min(r['worst_ratio'] for r in eligible)
    return min([r for r in eligible if r['worst_ratio']<=best*1.005],key=lambda r:(r['alpha'],r['worst_ratio'],r['name']))


def freeze():
    protocol();p=OUT/'selection.json'
    if p.exists():return json.loads(p.read_text())
    records=[]
    for name in SPECS:
        for alpha in [.25,.5]:
            yearly=[]
            for fold in WINDOWS:
                result=json.loads((OUT/fold/f'{name}_{SEED}.json').read_text())
                yearly.append(result['blends'][str(alpha)])
            ratios=[m['cold']['ratio'] for m in yearly]
            records.append({'name':name,'alpha':alpha,'ratios':ratios,
                'late_ratios':[m['cold_late']['ratio'] for m in yearly],
                'worst_ratio':max(ratios),'yearly_metrics':yearly})
    selection={'selected':choose(records),'grid':records,'config_sha256':checksum(OUT/'config.json'),
               'stage':'choice frozen before seed check and confirmation',
               'status':'primary_qualified' if choose(records) else 'rejected_no_cross_year_gain'}
    dump(p,selection);print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    return selection


def seed_check():
    selection=freeze()
    if selection['selected'] is None:return False
    choice=selection['selected'];name=choice['name'];alpha=choice['alpha'];reports={};accepted=True
    for fold in WINDOWS:
        fit_fold(fold,[name],SECOND_SEED)
        d=inputs(fold);old=baseline(fold);cold=cold_mask(d)
        p1=np.load(pred_path(fold,name,SEED));p2=np.load(pred_path(fold,name,SECOND_SEED))
        p=np.where(cold,(1-alpha)*old+alpha*(p1+p2)/2,old)
        m=metrics(d['vy'],old,p,cold,~d['tune']);reports[fold]=m
        individual=[float(np.mean(abs(d['vy'][cold]-((1-alpha)*old+alpha*q)[cold]))/m['cold']['baseline_mae']) for q in [p1,p2]]
        reports[fold]['individual_seed_cold_ratios']=individual
        accepted &= m['cold']['ratio']<=.995 and m['cold_late']['ratio']<=1 and max(individual)<1
    report={'accepted':bool(accepted),'selection_sha256':checksum(OUT/'selection.json'),'folds':reports}
    dump(OUT/'seed_check.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return bool(accepted)


def grouped_improvement(y,new,old,groups):
    a=pd.DataFrame({'group':groups,'gain':abs(y-old)-abs(y-new)})
    b=a.groupby('group').gain.agg(['sum','size']).to_numpy()
    rng=np.random.default_rng(SEED);means=[]
    for _ in range(1000):
        s=b[rng.integers(0,len(b),len(b))].sum(axis=0);means.append(s[0]/s[1])
    return np.quantile(means,[.025,.975]).tolist()


def confirm_and_export():
    selection=freeze()
    if selection['selected'] is None:return
    seed=OUT/'seed_check.json'
    if not seed.exists() or not json.loads(seed.read_text())['accepted']:return
    choice=selection['selected'];name=choice['name'];alpha=choice['alpha']
    for s in [SEED,SECOND_SEED]:fit_fold('confirmation',[name],s)
    d=inputs('confirmation');old=baseline('confirmation');cold=cold_mask(d)
    q=np.mean([np.load(pred_path('confirmation',name,s)) for s in [SEED,SECOND_SEED]],axis=0)
    p=np.where(cold,(1-alpha)*old+alpha*q,old)
    m=metrics(d['vy'],old,p,cold,~d['tune'])
    interval=grouped_improvement(d['vy'][cold],p[cold],old[cold],project_keys(d['vx']).to_numpy()[cold])
    accepted=m['cold']['ratio']<=.995 and m['cold_late']['ratio']<=1 and interval[0]>0
    report={'accepted':bool(accepted),'metrics':m,'cold_cluster_gain_95':interval,
        'history_reused':True,'choice_and_alpha_not_changed_after_confirmation':True}
    dump(OUT/'confirmation.json',report);np.savez_compressed(OUT/'confirmation/predictions.npz',y=d['vy'],previous=old,predictions=p,cold=cold)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not accepted:return
    for s in [SEED,SECOND_SEED]:fit_fold('final',[name],s)
    d=inputs('final');old=baseline('final');cold=cold_mask(d)
    q=np.mean([np.load(pred_path('final',name,s)) for s in [SEED,SECOND_SEED]],axis=0)
    final=np.where(cold,(1-alpha)*old+alpha*q,old)
    raw=pd.read_parquet(ROOT/'data/test.parquet')
    frame=export_submission(raw,final,ROOT/'submission1017_v7.csv')
    np.testing.assert_array_equal(frame.loc[~cold,'總價元'],old[~cold])
    dump(OUT/'verification.json',{'rows':len(frame),'changed_scope_rows':int(cold.sum()),
        'all_other_rows_including_policy_unchanged':True,'submission_sha256':checksum(ROOT/'submission1017_v7.csv'),
        'public_mae':None,'base_v6_sha256':BASE_SHA,'new_branch_uses_external_data':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','freeze','seed_check','confirm_export','all'])
    parser.add_argument('--fold',default='development');parser.add_argument('--models',nargs='+',default=list(SPECS));args=parser.parse_args()
    if args.stage=='train':fit_fold(args.fold,args.models)
    elif args.stage=='freeze':freeze()
    elif args.stage=='seed_check':seed_check()
    elif args.stage=='confirm_export':confirm_and_export()
    else:
        for fold in ['development','stress2021']:fit_fold(fold,list(SPECS))
        if freeze()['selected'] is not None and seed_check():confirm_and_export()
