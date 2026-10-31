"""Checks for the stress-test boundaries and aggregation, not model score targets."""
import numpy as np
import pytest

from audit_policy_generalization import families, near_feature_pairs, statistics
from train_policy_learned import price_features


def test_physical_purge_uses_floor_terrace_and_area_in_square_metres():
    fitting=price_features([100.,101.,101.01,100.,100.], [5,5,5,6,5], [0,0,0,0,1])
    heldout=price_features([100.], [5], [0])
    assert near_feature_pairs(fitting,heldout).ravel().tolist()==[True,True,False,False,False]
    # Price or project fields cannot change this target-free exclusion boundary.
    changed=fitting.assign(target=999999999, project='arbitrary')
    np.testing.assert_array_equal(near_feature_pairs(fitting,heldout),near_feature_pairs(changed,heldout))


def test_related_a18_phases_cannot_land_in_different_family_groups():
    keys=families(['桃園航空城A18-1基地','桃園航空城A18-2基地','桃園航空城A18-3基地',
                   '桃園航空城A11基地','桃園航空城G17基地'])
    assert keys[0]==keys[1]==keys[2]
    assert len(set(keys))==3


def test_transaction_mae_and_macro_project_mae_are_not_confused():
    y=np.array([100.,100.,100.,100.])
    baseline=np.array([90.,80.,80.,80.])
    candidate=np.array([100.,70.,70.,70.])
    result=statistics(y,candidate,baseline,['A','B','B','B'])
    assert result['row_mae']==22.5
    assert result['baseline_mae']==17.5
    assert result['macro_project_mae']==15.
    assert result['macro_project_baseline_mae']==15.
    assert result['groups_improved']==result['groups_worsened']==1
    assert result['improvement_pct']==pytest.approx(-100*5/17.5)
