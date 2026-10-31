"""Learn policy house prices from OTHER units; never look up query-unit prices.

Public offer labels remain an external proxy for actual transaction prices.
Project/unit identifiers are used only for exclusion and grouped validation.
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import ID, TARGET, chinese_number, normalize, roc_date
from policy_tables import unit_key
from policy_expert import contract_discount
from train_mae import checksum, dump, export_submission, save_model

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/policy_learned_v1'
OFFERS=ROOT/'artifacts/policy_v1'
CANDIDATES=['median_unit','linear_l1','shallow_l1']
FEATURES=['net_area_100sqm','floor_10','floor_10_squared','first_floor','second_floor','terrace']
BASE_SHA='723ce5f0c04740fdcdc4ff9cfc3976deddb644a0462fa5ab3e6a8b28aff7e058'


def price_features(area, floor, terrace):
    area=np.asarray(area,dtype=float);floor=np.asarray(floor,dtype=float)
    terrace=np.asarray(terrace,dtype=float)
    if not np.isfinite(np.c_[area,floor,terrace]).all() or (area<=0).any() or (floor<1).any():
        raise ValueError('Invalid physical features')
    return pd.DataFrame(dict(net_area_100sqm=area/100, floor_10=floor/10,
        floor_10_squared=(floor/10)**2, first_floor=(floor==1).astype(float),
        second_floor=(floor==2).astype(float), terrace=terrace))


def query_features(raw):
    # Intentionally never read 編號, 建案名稱, 棟及號, address, or any price table.
    area=pd.to_numeric(raw['建物移轉總面積平方公尺'])-pd.to_numeric(raw['車位移轉總面積平方公尺'])
    floor=raw['移轉層次'].map(chinese_number)
    terrace=raw['備註'].fillna('').str.contains('露台|露臺').astype(float)
    return price_features(area,floor,terrace),area.to_numpy(float)


def unit_identity(frame, project_col, unit_col):
    return pd.MultiIndex.from_arrays([frame[project_col].map(normalize),frame[unit_col].map(unit_key)])


def prepare(offers, test):
    # All test identities excluded before fitting AND before model selection.
    # No label from the excluded 1,404 units enters training or CV.
    blocked=unit_identity(test,'建案名稱','棟及號')
    keep=~unit_identity(offers,'project','unit').isin(blocked)
    safe=offers.loc[keep].copy().reset_index(drop=True)
    if safe.duplicated(['project','unit']).any():raise ValueError('Duplicate training units')
    floor=safe.unit.str.extract(r'-(\d+)F',expand=False).astype(float)
    x=price_features(safe.area_sqm,floor,safe.terrace_ping>0)
    y=safe.adjusted_house_land_price.to_numpy(float)
    area=safe.area_sqm.to_numpy(float)
    if not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid offer labels')
    return safe,x,y,area,keep


def fit(name,x,y,area):
    # Weighted unit L1 is exactly proportional to total-price L1.
    unit=y/(area*100000)
    weight=area/area.mean()
    if name=='median_unit':
        model=DummyRegressor(strategy='quantile',quantile=.5)
        model.fit(x,unit,sample_weight=weight)
    elif name=='linear_l1':
        model=make_pipeline(StandardScaler(),QuantileRegressor(quantile=.5,alpha=.001,solver='highs'))
        model.fit(x,unit,quantileregressor__sample_weight=weight)
    elif name=='shallow_l1':
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,max_leaf_nodes=7,
            min_samples_leaf=20,l2_regularization=10,learning_rate=.05,
            early_stopping=False,random_state=2026)
        model.fit(x,unit,sample_weight=weight)
    else:raise ValueError(name)
    return model


def house_predict(model,x,area):
    result=model.predict(x)*np.asarray(area)*100000
    if not np.isfinite(result).all() or (result<=0).any():raise ValueError('Invalid learned house price')
    return result


def cv_select(x,y,area,groups,n_splits):
    oof={name:np.zeros(len(y)) for name in CANDIDATES}
    for tr,va in GroupKFold(n_splits=n_splits).split(x,y,groups):
        assert not set(groups[tr]) & set(groups[va])
        for name in CANDIDATES:
            model=fit(name,x.iloc[tr],y[tr],area[tr])
            oof[name][va]=house_predict(model,x.iloc[va],area[va])
    scores={}
    for name,p in oof.items():
        by_project=pd.DataFrame({'project':groups,'ae':abs(p-y)}).groupby('project').ae.mean()
        scores[name]={'row_mae':float(abs(p-y).mean()),'macro_project_mae':float(by_project.mean()),
                      'worst_project_mae':float(by_project.max())}
    best=min(s['macro_project_mae'] for s in scores.values())
    # Prefer the simpler model if within 1% of the best project-balanced MAE.
    selected=next(name for name in CANDIDATES if scores[name]['macro_project_mae']<=best*1.01)
    return selected,scores,oof


def predict(raw,bundle):
    """Standalone inference needs fitted estimators and observed fields only."""
    x,area=query_features(raw)
    house=house_predict(bundle['house_model'],x,area)
    counts=pd.to_numeric(raw['交易筆棟數'].str.extract(r'車位(\d+)',expand=False),errors='coerce').to_numpy(float)
    if not np.isfinite(counts).all() or (counts<0).any():raise ValueError('Invalid car count')
    parking=bundle['parking_model'].predict(np.zeros((len(raw),1)))*counts
    # Explicit observed monetary difference is an accounting identity, not a
    # learned coefficient and not a fitted per-ID correction. Keep that explicit.
    discount=np.array([contract_discount(s)[2] for s in raw['備註']])
    total=house+parking-discount
    if (total<=0).any() or not np.isfinite(total).all():raise ValueError('Invalid learned total')
    return pd.DataFrame({'house_prediction':house,'parking_prediction':parking,
                         'observed_discount':discount,'prediction':total})


def main():
    OUT.mkdir(exist_ok=True)
    sources=['train_policy_learned.py','test_policy_learned.py','features.py','policy_tables.py','policy_expert.py','train_mae.py']
    files=[ROOT/'data/test.parquet',ROOT/'submission1017_v4.csv',OFFERS/'house_schedules.csv',OFFERS/'parking_schedules.csv']
    config={'version':'policy_learned_v1','source_sha256':{f:checksum(ROOT/f) for f in sources},
        'input_sha256':{str(f.relative_to(ROOT)):checksum(f) for f in files},
        'features':FEATURES,'excluded_identifiers':['編號','建案名稱','棟及號','address'],
        'exclude_query_unit_labels':True,'outer_folds':5,'inner_folds':3,'group':'whole project',
        'candidates':CANDIDATES,'selection':'macro-project MAE; simplest within 1%',
        'parking':'L1 intercept fitted to official parking inventory; no per-project lookup',
        'discount':'observed original-minus-reduced amount; deterministic accounting, not trained',
        'evaluation':'Offer-price proxy CV; not actual test-sale MAE',
        'source_availability':'Offers before affected query dates, not all before train cutoff',
        'user_requirement':'Trained general prediction; no per-query offer lookup or public-score adjustment'}
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:
        raise ValueError('Frozen learned-policy experiment changed; use new version')
    if checksum(ROOT/'submission1017_v4.csv')!=BASE_SHA:raise ValueError('Changed baseline')
    dump(OUT/'config.json',config)
    raw=pd.read_parquet(ROOT/'data/test.parquet')
    offers=pd.read_csv(OFFERS/'house_schedules.csv')
    safe,x,y,area,keep=prepare(offers,raw)
    safe.to_csv(OUT/'training_units.csv',index=False,encoding='utf-8-sig')
    offers.loc[~keep,['project','unit']].to_csv(OUT/'excluded_test_units.csv',index=False,encoding='utf-8-sig')
    groups=safe.project.to_numpy()
    outer=np.zeros(len(y));baseline=np.zeros(len(y));folds=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
        assert not set(groups[tr]) & set(groups[va])
        choice,metrics,_=cv_select(x.iloc[tr].reset_index(drop=True),y[tr],area[tr],groups[tr],3)
        model=fit(choice,x.iloc[tr],y[tr],area[tr])
        outer[va]=house_predict(model,x.iloc[va],area[va])
        baseline[va]=house_predict(fit('median_unit',x.iloc[tr],y[tr],area[tr]),x.iloc[va],area[va])
        folds.append({'fold':fold,'fit_projects':sorted(set(groups[tr])), 'heldout_projects':sorted(set(groups[va])),
            'train_rows':len(tr),'heldout_rows':len(va),'selected':choice,'inner_scores':metrics,
            'heldout_mae':float(abs(outer[va]-y[va]).mean()),'baseline_mae':float(abs(baseline[va]-y[va]).mean())})
        print('Outer fold',fold,choice,'offer MAE',folds[-1]['heldout_mae'],flush=True)
    choice,metrics,final_oof=cv_select(x,y,area,groups,5)
    model=fit(choice,x,y,area)
    park=pd.read_csv(OFFERS/'parking_schedules.csv').price.to_numpy(float)
    parking_model=DummyRegressor(strategy='quantile',quantile=.5).fit(np.zeros((len(park),1)),park)
    bundle={'house_model':model,'parking_model':parking_model,'features':FEATURES,
        'source_available':'2024-08-29','training_rows':len(safe),'selected':choice}
    save_model(OUT/'model.joblib',bundle)
    validation=safe[['project','unit']].assign(target_offer=y,outer_prediction=outer,baseline_prediction=baseline)
    validation.to_csv(OUT/'nested_group_oof.csv',index=False,encoding='utf-8-sig')
    by_project=validation.assign(abs_error=abs(outer-y),baseline_error=abs(baseline-y)).groupby('project').agg(
        rows=('unit','size'),mae=('abs_error','mean'),baseline_mae=('baseline_error','mean'))
    by_project.to_csv(OUT/'group_validation.csv',encoding='utf-8-sig')
    mask=raw['備註'].fillna('').str.contains('安置住宅')
    selected=raw.loc[mask].copy().reset_index(drop=True)
    if (roc_date(selected['交易年月日'])<bundle['source_available']).any():raise ValueError('Source newer than query')
    pred=predict(selected,joblib.load(OUT/'model.joblib'))
    # No names or unit identifiers are needed even for inference.
    anonymized=selected.drop(columns=[ID,'建案名稱','棟及號','土地位置建物門牌'])
    np.testing.assert_array_equal(pred.prediction,predict(anonymized,bundle).prediction)
    base=pd.read_csv(ROOT/'submission1017_v4.csv',dtype={ID:str}).set_index(ID)[TARGET]
    full=raw[ID].astype(str).map(base).to_numpy(float)
    full[mask.to_numpy()]=pred.prediction
    submission=export_submission(raw,full,ROOT/'submission1017_v6.csv')
    np.testing.assert_array_equal(submission.loc[~mask,TARGET],raw.loc[~mask,ID].astype(str).map(base))
    pred.insert(0,ID,selected[ID].to_numpy())
    pred['v4_prediction']=selected[ID].astype(str).map(base).to_numpy()
    pred.to_csv(OUT/'prediction_components.csv',index=False,encoding='utf-8-sig')
    report={'training_rows':len(safe),'excluded_test_unit_labels':int((~keep).sum()),
        'training_projects':int(safe.project.nunique()),'features':FEATURES,'selected':choice,
        'nested_group_offer_mae':float(abs(outer-y).mean()),
        'nested_group_baseline_offer_mae':float(abs(baseline-y).mean()),
        'nested_group_macro_project_mae':float(by_project.mae.mean()),
        'final_selection_cv':metrics,'outer_folds':folds,
        'learned_parking_median':float(parking_model.constant_.ravel()[0]),
        'parking_training_rows':len(park),'changed_rows':int(mask.sum()),'unchanged_rows':int((~mask).sum()),
        'relocation_prediction_mean':float(pred.prediction.mean()),
        'relocation_prediction_min':float(pred.prediction.min()),'relocation_prediction_max':float(pred.prediction.max()),
        'identifier_free_inference_verified':True,'public_mae':None,
        'submission_sha256':checksum(ROOT/'submission1017_v6.csv')}
    dump(OUT/'verification.json',report)
    (OUT/'source').mkdir(exist_ok=True)
    for f in sources:shutil.copy2(ROOT/f,OUT/'source'/f)
    print(json.dumps({k:v for k,v in report.items() if k not in ['outer_folds','final_selection_cv']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
