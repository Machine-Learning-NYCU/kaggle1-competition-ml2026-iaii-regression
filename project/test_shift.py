import numpy as np
import pandas as pd
from launch_index import LaunchIndex
from train_shift import semantic_features


def cohort_data():
    rows=[];prices=[]
    for district in ['A','B']:
        for project,start in enumerate([1,3,5,7,9]):
            for after in [0,1,4]:
                t=start+after
                rows.append(dict(geo_city='C',geo_district=district,
                    canonical_project=f'{district}{project}',building_type='flat',rooms=3,
                    transaction_year=2020+(t-1)//12,transaction_month=(t-1)%12+1))
                prices.append(100*100000*np.exp(.01*start)*(1.6 if district=='B' else 1))
    return pd.DataFrame(rows),np.array(prices),np.full(len(rows),100.)


def test_launch_slope_uses_initial_sales_not_later_project_inflation():
    x,y,a=cohort_data();first=LaunchIndex().fit(x,y,a)
    changed=y.copy();changed[np.arange(len(y))%3==2]*=3
    second=LaunchIndex().fit(x,changed,a)
    np.testing.assert_allclose(first.global_rate,.01,atol=1e-12)
    np.testing.assert_allclose(first.global_rate,second.global_rate,atol=1e-12)


def test_launch_slope_controls_district_level_price_differences():
    x,y,a=cohort_data();first=LaunchIndex().fit(x,y,a)
    changed=y.copy();changed[x.geo_district.eq('B')]*=2
    second=LaunchIndex().fit(x,changed,a)
    np.testing.assert_allclose(first.global_rate,second.global_rate,atol=1e-12)


def test_semantic_features_distinguish_regimes_without_reading_amounts():
    x=pd.DataFrame({'備註':['地上權房屋；','本交易為安置住宅，原房屋價款4500000元，差額減收','本交易為安置住宅，原房屋價款99999999元，差額減收',None]})
    f=semantic_features(x)
    assert f.sem_leasehold.iloc[0]==1
    assert f.sem_relocation.iloc[1]==1 and f.sem_subsidy.iloc[1]==1
    np.testing.assert_array_equal(f.iloc[1],f.iloc[2])
    assert f.iloc[3].sum()==0
