import numpy as np
import pandas as pd

from features import CategoryEncoder, chinese_number, duplicate_groups, roc_date, extract_road
from market import CausalMarket
from train import blend, learn_blend, scores


def toy_market():
    return pd.DataFrame({'transaction_year': [2023] * 8,
                         'transaction_month': [1, 1, 2, 2, 3, 3, 4, 4],
                         'city_district': ['A'] * 8, 'road': ['R'] * 8, 'project': ['P'] * 8})


def test_causal_market_excludes_current_month_and_future_labels():
    x = toy_market()
    y = np.arange(8.) / 20 + 12
    a = CausalMarket().fit(x, y).transform(x)
    perturbed = y.copy()
    perturbed[4:] += 100
    b = CausalMarket().fit(x, perturbed).transform(x)
    cols = [c for c in a if c.startswith('market_')]
    # March observations cannot see March or April prices.
    np.testing.assert_allclose(a.loc[:5, cols], b.loc[:5, cols])
    assert not np.allclose(a.loc[6:, 'market_project'], b.loc[6:, 'market_project'])
    assert (a.loc[:1, 'market_project_count'] == 0).all()


def test_market_unseen_groups_and_future_are_finite():
    x = toy_market()
    market = CausalMarket().fit(x, np.arange(8.) / 20 + 12)
    query = x.copy()
    query['project'] = 'unseen'
    query['transaction_year'] = 2025
    result = market.transform(query)
    assert (result['market_project_count'] == 0).all()
    assert np.isfinite(result['market_project']).all()
    np.testing.assert_allclose(result['market_project'], result['market_road'])


def test_date_and_floor_parsing():
    dates = roc_date(pd.Series(['1130810', '1140731', '20240810', '11203', '0', '1600101', '說明']))
    assert dates.iloc[:4].dt.strftime('%Y-%m-%d').tolist() == ['2024-08-10', '2025-07-31', '2024-08-10', '2023-03-01']
    assert dates.iloc[4:].isna().all()
    assert chinese_number('地下二層') == -2
    assert chinese_number('二十一層') == 21
    assert chinese_number('十一層、十二層') == 11
    assert chinese_number('１８樓') == 18


def test_encoder_does_not_learn_from_validation():
    encoder = CategoryEncoder().fit(pd.DataFrame({'c': ['a', 'b']}), ['c'])
    result = encoder.transform(pd.DataFrame({'c': ['a', 'new']}))
    assert result.c.cat.codes.tolist() == [0, -1]
    assert encoder.levels['c'] == ['a', 'b']


def test_road_does_not_stop_at_district_name():
    values = pd.Series(['桃園市大園區領航南路二段12號', '臺北市大安區復興南路一段20號',
                        '臺南市善化區目加溜灣大道與陽光三路交叉口', '新竹縣竹北市勝利八街10號'])
    assert extract_road(values).tolist() == ['桃園市大園區領航南路二段', '臺北市大安區復興南路一段',
                                            '臺南市善化區目加溜灣大道', '新竹縣竹北市勝利八街']


def test_duplicates_ignore_identifier_and_target():
    d = pd.DataFrame({'編號': ['a', 'b'], '總價元': [10, 20], 'area': [30, 30]})
    assert len(set(duplicate_groups(d))) == 1


def test_blend_simplex_and_log_metric():
    y = np.linspace(1e6, 1e7, 100)
    matrix = np.column_stack([y * 1.1, y * .9, y * 1.01, y * 1.5])
    for metric in ['rmsle', 'rmse']:
        weights = learn_blend(matrix, y, metric)
        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-8)
        assert scores(y, blend(matrix, weights, metric))[metric] < scores(y, matrix[:, 3])[metric]
    assert scores(y, y)['rmsle'] == 0
