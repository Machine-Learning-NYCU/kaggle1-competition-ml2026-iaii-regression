import numpy as np
import pandas as pd

from features import MISSING
from train_generalization import ProjectIdentityMasker, IDENTITIES, choose, cold_mask


def example():
    keys=['A','A','B','B','C','D','E','E','F','G']
    x=pd.DataFrame({c:keys for c in IDENTITIES})
    x['geo_district']='city|district';x['house_number']=range(len(keys));x['parcel_number']=100
    x['building_area']=120.;x['floor']=3.;x['sem_relocation']=0
    return x


def test_masks_whole_projects_all_aliases_and_is_order_invariant():
    x=example();m=ProjectIdentityMasker(.5,2026).fit(x);z=m.transform(x)
    assert 0<z.identity_masked.sum()<len(x)
    assert z.groupby(x.canonical_project).identity_masked.nunique().max()==1
    selected=z.identity_masked==1
    assert (z.loc[selected,IDENTITIES]==MISSING).all().all()
    assert z.loc[selected,['house_number','parcel_number']].isna().all().all()
    np.testing.assert_array_equal(z.building_area,x.building_area)
    pd.testing.assert_frame_equal(z,m.transform(x.iloc[::-1]).sort_index())
    same=ProjectIdentityMasker(.5,2026).fit(x.iloc[::-1])
    assert m.retained==same.retained


def test_new_project_identity_is_fully_masked_and_policy_not_routed():
    x=example();m=ProjectIdentityMasker(.35,2026).fit(x)
    q=x.iloc[:2].copy()
    for col in IDENTITIES:q[col]=['NEW','POLICY']
    q.loc[q.index[1],'sem_relocation']=1
    z=m.transform(q)
    assert (z.identity_masked==1).all()
    assert (z[IDENTITIES]==MISSING).all().all()
    d={'x':x,'vx':q,'seen':np.zeros(len(q),bool)}
    assert cold_mask(d).tolist()==[True,False]


def test_selection_rejects_single_year_or_late_window_regression():
    def row(name,ratios,late,alpha=.25):
        return {'name':name,'alpha':alpha,'ratios':ratios,'late_ratios':late,'worst_ratio':max(ratios)}
    bad=[row('one_good_year',[.8,1.01],[.8,1.01]),row('late_failure',[.98,.98],[.98,1.01])]
    assert choose(bad) is None
    small=row('small',[.98,.98],[.99,.99]);large=row('large',[.978,.978],[.98,.98],.5)
    assert choose(bad+[small,large])['name']=='small'


def test_unit_l1_weights_match_total_price_mae_without_clipping():
    base=np.array([50.,150.,600.]);y=np.array([5e6,1.5e7,8e7]);unit=np.array([1.1,.9,1.2])
    np.testing.assert_allclose((base/100)*abs(y/base/1e5-unit),abs(y-base*1e5*unit)/1e7)
