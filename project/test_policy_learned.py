import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from train_policy_learned import fit, house_predict, predict, prepare, price_features, query_features


def test_all_query_unit_labels_removed_before_learning():
    offers=pd.DataFrame({'project':['P','P','Q'],'unit':['A1-2F','A1-3F','A1-2F'],
        'area_sqm':[100,100,110],'terrace_ping':[0,0,0], 'adjusted_house_land_price':[6e6,6.1e6,7e6]})
    queries=pd.DataFrame({'建案名稱':['P','P'],'棟及號':['A棟1-2F號','A1-2F號']})
    safe,x,y,area,keep=prepare(offers,queries)
    assert keep.tolist()==[False,True,True]
    assert safe.unit.tolist()==['A1-3F','A1-2F']
    changed=offers.copy();changed.loc[0,'adjusted_house_land_price']=999999999
    np.testing.assert_array_equal(y,prepare(changed,queries)[2])


def test_learned_model_predicts_without_query_names_or_unit_tables():
    area=np.tile(np.array([90,120,150]),10)
    floor=np.repeat(np.arange(1,11),3)
    x=price_features(area,floor,np.zeros(len(area)))
    y=area*(50000+800*floor)
    model=fit('linear_l1',x,y,area)
    learned=house_predict(model,x,area)
    assert abs(learned-y).mean()<10000  # learns the planted physical relationship
    parking=DummyRegressor(strategy='median').fit([[0],[0],[0]],[900000,1000000,1100000])
    bundle={'house_model':model,'parking_model':parking}
    raw=pd.DataFrame({'建物移轉總面積平方公尺':['150'],'車位移轉總面積平方公尺':['30'],
        '移轉層次':['五層'],'交易筆棟數':['土地1建物1車位1'],
        '備註':['本交易為安置住宅原房屋價款4000000元差額減收後之房屋價款為3500000元']})
    a=predict(raw,bundle)
    b=predict(raw.assign(編號='arbitrary',建案名稱='new unseen project',棟及號='unknown',總價元=1),bundle)
    np.testing.assert_array_equal(a,b)
    assert a.observed_discount.iloc[0]==500000
    assert a.prediction.iloc[0]==a.house_prediction.iloc[0]+1000000-500000


def test_area_weighted_unit_loss_matches_total_loss():
    area=np.array([75.,150.,225.]);y=np.array([4e6,8e6,13e6]);unit=np.array([.6,.55,.58])
    weighted=(area/area.mean())*abs(y/(area*100000)-unit)
    direct=abs(y-area*100000*unit)/(area.mean()*100000)
    np.testing.assert_allclose(weighted,direct)
