import numpy as np
import pandas as pd
from spatial_prior import CrossProjectPrior,effective_area


def example():
    return pd.DataFrame({'canonical_project':['a','a','b','b','c','c'],
        'geo_city':['city']*6,'geo_district':['city|district']*6,'address_base':['addr']*6,
        'district_type':['district|flat']*6,'building_type':['flat']*6,
        'geo_road1':['roadA','roadA','roadB','roadB','roadC','roadC'],
        'geo_road2':['roadX','roadX','roadA','roadA','roadB','roadB'],
        'geo_section':['section']*6,'geo_cross':['crossA','crossA','crossB','crossB','crossC','crossC'],
        'builder_prefix_2':['builder']*6,'transaction_year':[2023]*6,'transaction_month':[1,2,1,2,1,2]})


def test_entire_project_labels_are_excluded_at_all_levels():
    x=example();y=np.array([10.,11.,12.,13.,14.,15.])*1e6;area=np.full(6,100.)
    first=CrossProjectPrior().fit(x,y,area).transform(x.iloc[:2])
    y[:2]*=3
    second=CrossProjectPrior().fit(x,y,area).transform(x.iloc[:2])
    np.testing.assert_allclose(first,second,rtol=1e-5,atol=1e-5)


def test_roads_share_slots_and_unseen_queries_back_off():
    x=example();model=CrossProjectPrior().fit(x,np.arange(10.,16.)*1e6,np.full(6,100.))
    query=x.iloc[:1].copy();query['canonical_project']='new'
    query['geo_road1']='roadA';query['geo_road2']='roadZ'
    first=model.transform(query)
    query['geo_road1']='roadZ';query['geo_road2']='roadA'
    second=model.transform(query)
    np.testing.assert_array_equal(first,second)
    query['geo_road2']='__MISSING__';query['geo_road1']='__MISSING__'
    fallback=model.transform(query)
    assert np.isfinite(fallback.to_numpy()).all()


def test_effective_area_handles_invalid_parking_and_land():
    x=pd.DataFrame({'building_area':[100.,100.,0.,100.],'parking_area':[20.,120.,0.,-5.],
                    'land_area':[10.,10.,50.,10.]})
    np.testing.assert_array_equal(effective_area(x,'net'),[85.,100.,50.,100.])
