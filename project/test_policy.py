import numpy as np
import pandas as pd
import pytest

from policy_expert import contract_discount, parking_count, predict_policy
from policy_tables import unit_key


def fixture():
    raw = pd.DataFrame([{'編號': '7', '備註': '本交易為安置住宅，原房屋價款4000000元，差額減收後之房屋價款為3000000元',
        '建案名稱': '桃園航空城A11基地', '棟及號': 'A棟1-3F號', '交易年月日': '1130913',
        '交易筆棟數': '土地2建物1車位1', '建物移轉總面積平方公尺': '130', '車位移轉總面積平方公尺': '30'}])
    houses = pd.DataFrame([dict(project='桃園航空城A11基地', unit='A1-3F', source='example',
        source_page=1, source_published='2023-07-12', area_sqm=100, area_ping=30.25,
        policy_cut_per_ping=30000, published_house_land_price=7000000)])
    parks = pd.DataFrame(dict(project=['桃園航空城A11基地']*3, price=[850000,950000,1250000]))
    return raw, houses, parks


def test_price_components_and_no_double_cut():
    raw, houses, parks = fixture()
    out = predict_policy(raw, houses, parks).iloc[0]
    assert out.policy_prediction == 7000000-30.25*30000+950000-1000000
    houses['published_house_land_price'] -= 30.25*30000
    houses['policy_cut_per_ping'] = 0  # already reduced book
    assert predict_policy(raw, houses, parks).iloc[0].policy_prediction == out.policy_prediction


def test_zero_car_and_explicit_discount_validation():
    raw, houses, parks = fixture()
    raw['交易筆棟數'] = '土地2建物1車位0'
    raw['車位移轉總面積平方公尺'] = '0'
    raw['建物移轉總面積平方公尺'] = '100'
    assert predict_policy(raw,houses,parks).iloc[0].parking_estimate == 0
    assert contract_discount('安置住宅含露臺') == (0,0,0)
    assert contract_discount('原房屋價款4,000,000元，差額減收後之房屋價款為3,000,000元') == (4000000,3000000,1000000)
    with pytest.raises(ValueError): contract_discount('符合減收規定')
    with pytest.raises(ValueError): contract_discount('原房屋價款100元差額減收後之房屋價款為200元')
    with pytest.raises(ValueError): parking_count('土地1建物2車位1')


def test_source_date_and_area_guard():
    raw, houses, parks = fixture()
    raw['交易年月日'] = '1130810'
    with pytest.raises(ValueError): predict_policy(raw,houses,parks)
    raw['交易年月日'] = '1130913'
    raw['建物移轉總面積平方公尺'] = '125.5'
    out = predict_policy(raw,houses,parks).iloc[0]
    assert out.area_scale == .955
    assert out.scenario_low <= out.policy_prediction <= out.scenario_high
    raw['建物移轉總面積平方公尺'] = '120'
    with pytest.raises(ValueError): predict_policy(raw,houses,parks)


def test_gate_ambiguity_and_identity():
    raw, houses, parks = fixture()
    assert unit_key('Ａ棟１－３Ｆ號') == 'A1-3F'
    assert unit_key('A棟1-03F號') == 'A1-3F'
    private = raw.copy(); private['建案名稱']='璟都航空城'
    assert predict_policy(private,houses,parks).empty
    assert predict_policy(raw.assign(備註='一般住宅'),houses,parks).empty
    with pytest.raises(ValueError): predict_policy(raw,pd.concat([houses,houses]),parks)
    with pytest.raises(ValueError): predict_policy(raw.assign(棟及號='不存在'),houses,parks)
    both = pd.concat([raw,raw.assign(編號='8')],ignore_index=True)
    a = predict_policy(both,houses,parks).set_index('編號').policy_prediction.sort_index()
    b = predict_policy(both.iloc[::-1].assign(總價元='do not use'),houses,parks).set_index('編號').policy_prediction.sort_index()
    np.testing.assert_array_equal(a,b)
