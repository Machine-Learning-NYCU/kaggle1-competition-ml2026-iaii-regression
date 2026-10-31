import pandas as pd

from moi_history import fingerprint, fit_priors, parking_predict


def test_duplicate_fingerprint_ignores_ids_targets_and_normalizes_numbers():
    a=pd.DataFrame({'編號':['original'],'總價元':['123'],'交易年月日':['1130301'],
                    '土地位置建物門牌':['台北市 中山區'],'建物移轉總面積平方公尺':['100.00']})
    b=a.assign(編號='renumbered',總價元='999',土地位置建物門牌='臺北市中山區',建物移轉總面積平方公尺='100')
    cols=['交易年月日','土地位置建物門牌','建物移轉總面積平方公尺']
    assert fingerprint(a,cols).iloc[0]==fingerprint(b,cols).iloc[0]
    b['交易年月日']='1130302'
    assert fingerprint(a,cols).iloc[0]!=fingerprint(b,cols).iloc[0]


def test_parking_priors_use_only_fit_components_and_respect_car_count():
    fit=pd.DataFrame({'土地位置建物門牌':['桃園市大園區某路']*5,'鄉鎮市區':['大園區']*5,
        '車位類別':['坡道平面']*5,'建案名稱':['某建案']*5,'交易筆棟數':['土地1建物1車位1']*5,
        '車位總價元':[1000000]*5,'總價元':[12000000]*5})
    tables,default=fit_priors(fit)
    query=fit.iloc[:1].assign(交易筆棟數='土地1建物1車位2',總價元=1,車位總價元=99999999)
    result=parking_predict(query,tables,default).iloc[0]
    assert result.parking_prior_total==2000000
    assert result.parking_prior_level=='project'
    assert parking_predict(query.assign(交易筆棟數='土地1建物1車位0'),tables,default).iloc[0].parking_prior_total==0
