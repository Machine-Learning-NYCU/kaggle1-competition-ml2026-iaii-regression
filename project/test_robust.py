import numpy as np
import pandas as pd
from robust_features import address_parts, RepeatProjectIndex


def test_canonical_address_and_intersections():
    a=address_parts('桃園縣平鎮市新榮里26鄰富貴街35號十樓','平鎮市')
    b=address_parts('桃園市平鎮區富貴街35號','平鎮區')
    assert a[:3]==b[:3]
    c=address_parts('臺北市北投區洲美里福美路與福善路','北投區')
    assert c[2]=='臺北市|北投區|福美路'
    assert c[3]=='臺北市|北投區|福善路'
    d=address_parts('新竹市新竹市北區金雅路與金雅七街交叉路口','新竹市')
    assert d[1]=='新竹市|北區'
    assert d[2]=='新竹市|北區|金雅路'


def test_unit_target_weight_is_exact_price_mae():
    y=np.array([8e6,2e7,9e8]);b=np.array([80.,160.,4000.]);z=np.array([.9,1.4,2.])
    np.testing.assert_allclose(b/100*abs(y/b/1e5-z),abs(y-z*b*1e5)/1e7)


def test_repeat_project_index_removes_composition_shift():
    # A luxury project starts late; its level must not inflate the common trend.
    frames=[];prices=[]
    for name,start,level in [('old',0,1e5),('new',12,3e5)]:
        t=np.arange(start,24)
        frames.append(pd.DataFrame({'transaction_year':2022+t//12,'transaction_month':1+t%12,
            'geo_city':'city','canonical_project':name,'building_type':'flat','rooms':3}))
        prices.extend(100*level*np.exp(.005*t))
    x=pd.concat(frames,ignore_index=True);area=np.full(len(x),100.)
    model=RepeatProjectIndex().fit(x,np.array(prices),area)
    assert abs(model.global_rate-.005)<1e-10
    # Query rows cannot change fitted rates and extrapolation is bounded at 12 months.
    query=x.iloc[:1].copy();query['transaction_year']=2030
    np.testing.assert_allclose(model.factor(query),np.exp(.005*12))


def test_comparables_do_not_borrow_labels_across_projects():
    from comparable_sales import ComparableSales
    x=pd.DataFrame({'canonical_project':['a','b'],'geo_district':['city']*2,'address_base':['road']*2,
        'building_area':[100.,100.],'parking_fraction':[0.,0.],'floor':[3.,3.],
        'rooms':[3.,3.],'total_floors':[10.,10.],'parking_count':[0.,0.],
        'building_type':['flat']*2,'transaction_days':[100.,100.]})
    query=pd.concat([x.iloc[:1],x.iloc[:1]],ignore_index=True)
    query.loc[1,'canonical_project']='unseen'
    p,confidence=ComparableSales().fit(x,[1e7,9e8],[100.,100.]).predict(query,[100.,100.])
    assert p[0]==1e7 and confidence[0]>0
    assert np.isnan(p[1]) and confidence[1]==0


def test_new_candidate_does_not_inherit_old_2024_calibration(tmp_path,monkeypatch):
    import json,joblib
    import select_robust as selection
    monkeypatch.setattr(selection,'ROOT',tmp_path)
    out=tmp_path/'new';monkeypatch.setattr(selection,'OUT',out)
    path=out/'confirmation';path.mkdir(parents=True)
    old=tmp_path/'artifacts/mae_final';old.mkdir(parents=True)
    calibration=old/'calibration.json'
    calibration.write_text(json.dumps({'scale':1.,'weights':[1.,0.,0.,0.]}))
    (out/'legacy_development_calibration.json').write_text(json.dumps({'weights':[.25]*4}))
    joblib.dump({'vy':np.array([10.,20.])},path/'data.joblib')
    for j,name in enumerate(selection.OLD_SPECS):
        np.save(path/f'{name}_predictions.npy',np.array([10.,20.])*(j+1))
    for name in selection.NEW_NAMES:
        np.save(path/f'{name}_predictions.npy',np.array([15.,25.]))
    np.savez(path/'comparable_predictions.npz',predictions=[np.nan,np.nan],confidence=[0.,0.])
    first,matrix1,_,_=selection.inputs('confirmation')
    # Alter the old blend, which was fitted on later labels.
    calibration.write_text(json.dumps({'scale':2.,'weights':[0.,1.,0.,0.]}))
    second,matrix2,_,_=selection.inputs('confirmation')
    np.testing.assert_array_equal(matrix1,matrix2)
    assert not np.array_equal(first['reference'],second['reference'])
