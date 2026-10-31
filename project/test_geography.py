import numpy as np
import pandas as pd
import pytest

from geography_features import RoadGeography, address_keys, distance


def references():
    return pd.DataFrame({'address':[f'臺北市大安區仁愛路一段{10+i}號' for i in range(12)],
        'longitude':121.54+np.arange(12)*.0001, 'latitude':25.03+np.arange(12)*.0001,
        'event_date':pd.to_datetime(['2020-01-01']*12)})


def test_address_aliases_and_intersections():
    a = address_keys('新竹市新竹市北區金雅路與金雅七街交叉路口', '新竹市')
    b = address_keys('新竹市北區金雅七街 / 新竹市北區金雅路')
    assert a[4] == b[4] == '新竹市|北區|金雅七街&金雅路'
    a = address_keys('台北市大安區仁愛路一段15號')
    b = address_keys('臺北市大安區仁愛路１段１５號')
    assert a == b
    assert address_keys('臺南市新市區中正路')[1] == '臺南市|新市區'
    assert address_keys('桃園縣大園鄉中正路')[1] == '桃園市|大園區'


def test_external_prices_and_later_years_rejected():
    with pytest.raises(ValueError, match='schema'):
        RoadGeography().fit(references().assign(總價元=10000000))
    later = references()
    later.loc[0, 'event_date'] = pd.Timestamp('2021-01-01')
    with pytest.raises(ValueError, match='2020'):
        RoadGeography().fit(later)


def test_geography_ignores_target_ids_and_project_names():
    m = RoadGeography().fit(references())
    raw = pd.DataFrame({'土地位置建物門牌':['臺北市大安區仁愛路一段15號', '臺北市大安區不存在路'],
        '鄉鎮市區':['大安區']*2, '總價元':[100, 99999999], '編號':['a', 'b'], '建案名稱':['甲', '乙']})
    expected = m.transform(raw)
    changed = raw.assign(總價元=-1, 編號='anything', 建案名稱='different')
    pd.testing.assert_frame_equal(expected, m.transform(changed))
    pd.testing.assert_frame_equal(expected.iloc[::-1].reset_index(drop=True), m.transform(raw.iloc[::-1]))
    assert expected.ext_location_level.tolist() == [4, 2]


def test_unknown_locations_stay_missing_and_coordinate_outliers_excluded():
    a = references()
    invalid = a.iloc[[0]].assign(longitude=0., latitude=0.)
    m = RoadGeography().fit(pd.concat([a, invalid], ignore_index=True))
    assert m.audit['invalid_coordinates'] == 1
    raw = pd.DataFrame({'土地位置建物門牌':['完全無法定位'], '鄉鎮市區':['']})
    result = m.transform(raw)
    assert result.ext_location_level.iloc[0] == 0
    assert np.isnan(result.ext_longitude.iloc[0])
    assert np.isnan(result.ext_reference_density_1km_log.iloc[0])


def test_reference_row_order_and_duplicate_query_do_not_change_coordinates():
    a = references()
    m1 = RoadGeography().fit(a)
    m2 = RoadGeography().fit(a.sample(frac=1, random_state=17))
    raw = pd.DataFrame({'土地位置建物門牌':['臺北市大安區仁愛路一段15號'], '鄉鎮市區':['大安區']})
    pd.testing.assert_frame_equal(m1.transform(raw), m2.transform(raw))
    both = m1.transform(pd.concat([raw, raw], ignore_index=True))
    np.testing.assert_array_equal(both.iloc[0], both.iloc[1])
    assert distance(121., 25., 121., 25.) == 0


def test_selection_requires_external_ablation_and_cross_year_gain():
    from train_geography import choose
    valid = dict(name='cat_geo', alpha=.25, ratios=[.99, .99], late_ratios=[.99, .99],
        control_ratios=[.99, .99], worst_ratio=.99)
    assert choose([valid]) == valid
    assert choose([dict(valid, control_ratios=[1.01, .99])]) is None
    assert choose([dict(valid, ratios=[.99, 1.01])]) is None
    assert choose([dict(valid, late_ratios=[1.01, .99])]) is None
    assert choose([dict(valid, alpha=.5, worst_ratio=.989), valid]) == valid
