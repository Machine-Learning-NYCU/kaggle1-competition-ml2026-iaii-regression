"""Audit MOI historical presales and build parking-component candidates.

Archive quarter != verified publication vintage. This script never silently
appends external labels or changes a competition submission.
"""
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID, TARGET, normalize, roc_date
from train_mae import checksum, dump

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/moi_v1'
ARCHIVE = OUT/'raw/113S2_csv.zip'
NUMERIC = ['土地移轉總面積平方公尺', '建物移轉總面積平方公尺', '車位移轉總面積平方公尺',
           '交易年月日', '總樓層數', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛']
CORE = ['鄉鎮市區','土地位置建物門牌','交易年月日','建案名稱','棟及號',
        '土地移轉總面積平方公尺','建物移轉總面積平方公尺','車位移轉總面積平方公尺','交易筆棟數']


def fingerprint(frame, columns):
    """No labels or arbitrary row IDs in the transaction fingerprint."""
    normalized = {}
    for col in columns:
        if col in NUMERIC:
            normalized[col] = pd.to_numeric(frame[col],errors='coerce').round(6).astype('Float64').astype(str)
        else:
            normalized[col] = frame[col].map(normalize)
    return pd.util.hash_pandas_object(pd.DataFrame(normalized), index=False).astype('uint64')


def load_archive(path):
    rows=[]; timestamps=[]
    with zipfile.ZipFile(path) as z:
        description = z.read('build_time.xml').decode('utf-8-sig')
        for name in sorted(z.namelist()):
            if re.fullmatch('[a-z]_lvr_land_b.csv',name):
                frame = pd.read_csv(z.open(name),dtype=str)
                frame['_source_file'] = name
                rows.append(frame)
                timestamps.append(list(z.getinfo(name).date_time))
    frame = pd.concat(rows,ignore_index=True)
    dates = roc_date(frame['交易年月日'])
    target = pd.to_numeric(frame[TARGET],errors='coerce')
    keep = dates.between('2024-02-11','2024-05-10') & (target>0)
    clean = frame.loc[keep].copy()
    clean['_transaction_date'] = dates.loc[keep]
    clean['_asof_version_verified'] = False
    clean[ID] = clean[ID].str.strip()
    return clean.reset_index(drop=True), {
        'build_time_xml': description, 'excluded_rows': int((~keep).sum()),
        'zip_member_time_min': min(timestamps), 'zip_member_time_max': max(timestamps),
        'historical_release_vintage_verified': False,
        'vintage_note': 'Downloaded historical archive has 2025 ZIP metadata; original 2024 snapshot is not verified.',
    }


def parking_features(frame):
    address = frame['土地位置建物門牌'].map(normalize)
    city = address.str.extract(r'^(.{2,3}?[縣市])',expand=False).fillna('unknown')
    district = city+'|'+frame['鄉鎮市區'].map(normalize)
    kind = frame['車位類別'].map(normalize)
    project = frame['建案名稱'].map(normalize)
    project = project.where(project.ne(''),address)
    count = pd.to_numeric(frame['交易筆棟數'].str.extract(r'車位(\d+)',expand=False),errors='coerce')
    return pd.DataFrame({'city':city+'|'+kind, 'district':district+'|'+kind,
                         'project':district+'|'+project+'|'+kind, 'cars':count},index=frame.index)


def fit_priors(frame):
    x = parking_features(frame)
    park = pd.to_numeric(frame['車位總價元'],errors='coerce')
    total = pd.to_numeric(frame[TARGET],errors='coerce')
    keep = (x.cars>0)&(park>0)&(park<total)&np.isfinite(park)
    x = x.loc[keep].copy(); x['per_car'] = park.loc[keep]/x.cars
    if x.empty:
        raise ValueError('No priced parking components')
    tables={key:x.groupby(key).per_car.agg(['median','size']) for key in ['city','district','project']}
    return tables,float(x.per_car.median())


def parking_predict(frame,tables,fallback,project=True):
    x=parking_features(frame)
    prediction=pd.Series(fallback,index=frame.index,dtype=float)
    level=pd.Series('global',index=frame.index,dtype=str)
    # Fixed minimum counts, no fit to public score or validation choices.
    tiers=[('city',20),('district',20)]+([('project',5)] if project else [])
    for key,minimum in tiers:
        table=tables[key].loc[tables[key]['size']>=minimum,'median']
        matched=x[key].map(table)
        ok=matched.notna()
        prediction.loc[ok]=matched.loc[ok];level.loc[ok]=key
    total=prediction*x.cars
    level.loc[x.cars==0]='no_parking'
    level.loc[x.cars.isna()]='unknown_count'
    return pd.DataFrame({'parking_prior_total':total,'parking_prior_per_car':prediction,
                         'parking_prior_level':level,'parking_count':x.cars},index=frame.index)


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'raw/113S2_csv.json').read_text())
    if checksum(ARCHIVE)!=manifest['sha256']:
        raise ValueError('Downloaded MOI archive changed')
    external,metadata=load_archive(ARCHIVE)
    common=list(pd.read_parquet(ROOT/'data/test.parquet').drop(columns=ID).columns)
    external['_core_key']=fingerprint(external,CORE)
    external['_strict_key']=fingerprint(external,common)
    external['_duplicate_source_id']=external[ID].duplicated(keep=False)
    external['_duplicate_core']=external['_core_key'].duplicated(keep=False)
    local={};counts={}
    for name in ['train','valid','test']:
        frame=pd.read_parquet(ROOT/f'data/{name}.parquet')
        frame[ID]=frame[ID].str.strip()
        frame['_core_key']=fingerprint(frame,CORE)
        frame['_strict_key']=fingerprint(frame,common)
        id_overlap=external[ID].isin(frame[ID])
        core_overlap=external['_core_key'].isin(frame['_core_key'])
        external[f'_overlap_{name}']=id_overlap|core_overlap
        counts[name]={'by_original_id':int(id_overlap.sum()),'by_core_fields':int(core_overlap.sum()),
                      'by_either':int((id_overlap|core_overlap).sum())}
        local[name]=frame
    # One-to-one original ID AND complete feature fingerprint AND equal total.
    # Existing labels are not replaced by revised government labels.
    train=local['train']
    train_unique=train.loc[~train[ID].duplicated(keep=False),[ID,TARGET,'_strict_key']]
    joined=external.loc[~external['_duplicate_source_id']].merge(train_unique,on=ID,suffixes=('','_local'),validate='one_to_one')
    same_y=pd.to_numeric(joined[TARGET])==pd.to_numeric(joined[TARGET+'_local'])
    same_x=joined['_strict_key']==joined['_strict_key_local']
    eligible=same_y&same_x&~joined['_overlap_valid']&~joined['_overlap_test']&~joined['_duplicate_core']
    enriched=joined.loc[eligible].drop(columns=[TARGET+'_local','_strict_key_local']).copy()
    novel=external.loc[~external[[f'_overlap_{name}' for name in local]].any(axis=1)
                       &~external['_duplicate_source_id']&~external['_duplicate_core']].copy()
    external.to_parquet(OUT/'presales_historical.parquet',index=False)
    enriched.to_parquet(OUT/'train_components_verified.parquet',index=False)
    novel.to_parquet(OUT/'novel_candidates_not_trained.parquet',index=False)
    # This is a retrospective component benchmark, not full-house MAE or a
    # historical online evaluation with independently verified release vintages.
    cutoff=pd.Timestamp('2024-04-10')
    past=enriched.loc[enriched['_transaction_date']<=cutoff]
    future=enriched.loc[enriched['_transaction_date']>cutoff]
    tables,fallback=fit_priors(past)
    pred=parking_predict(future,tables,fallback)
    coarse=parking_predict(future,tables,fallback,project=False)
    true=pd.to_numeric(future['車位總價元'],errors='coerce')
    valid=(pred.parking_count>0)&(true>0)&(true<pd.to_numeric(future[TARGET]))
    benchmark={'status':'Retrospective parking-component MAE; not whole-house MAE and not vintage-verified',
        'cutoff':'2024-04-10','rows':int(valid.sum()),
        'district_backoff_mae':float((coarse.loc[valid,'parking_prior_total']-true.loc[valid]).abs().mean()),
        'project_backoff_mae':float((pred.loc[valid,'parking_prior_total']-true.loc[valid]).abs().mean()),
        'prediction_level_counts':pred.loc[valid,'parking_prior_level'].value_counts().to_dict()}
    future_audit=future.loc[valid,[ID,'_transaction_date','車位總價元']].join(pred.loc[valid])
    future_audit.to_parquet(OUT/'parking_component_evaluation.parquet',index=False)
    tables,fallback=fit_priors(enriched)
    for key,table in tables.items():table.to_csv(OUT/f'parking_{key}_prior.csv',encoding='utf-8-sig')
    test=local['test']
    candidate=parking_predict(test,tables,fallback)
    candidate.insert(0,ID,test[ID])
    candidate['_asof_version_verified']=False
    candidate.to_parquet(OUT/'test_parking_features_candidate.parquet',index=False)
    config={'external_source':manifest, 'source_sha256':{name:checksum(ROOT/name) for name in ['moi_history.py','features.py','test_moi.py']},
            'input_sha256':{name:checksum(ROOT/f'data/{name}.parquet') for name in local},
            'transaction_range':['2024-02-11','2024-05-10'], 'core_key':CORE,'strict_key':common,
            'no_automatic_training_append':True,'excluded_official_valid':True,'historical_vintage_verified':False,
            'parking_prior_min_counts':{'city':20,'district':20,'project':5}}
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:
        raise ValueError('Changed MOI experiment; use a new version')
    dump(OUT/'config.json',config)
    report={**metadata,'source_rows':len(external),'source_files':int(external['_source_file'].nunique()),
            'overlap':counts,'one_to_one_train_ids':len(joined),
            'total_price_conflicts':int((~same_y).sum()),'attribute_conflicts':int((~same_x).sum()),
            'verified_train_components':len(enriched),'novel_candidates_not_trained':len(novel),
            'parking_component_benchmark':benchmark,
            'test_parking_feature_coverage':candidate.parking_prior_level.value_counts().to_dict(),
            'used_in_submission_v5':False}
    dump(OUT/'audit.json',report)
    (OUT/'source').mkdir(exist_ok=True)
    for name in config['source_sha256']:shutil.copy2(ROOT/name,OUT/'source'/name)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
