import numpy as np
from select_hedonic import choose_branch, apply


def test_two_window_selection_rejects_regime_tradeoff():
    rows=[dict(candidate='unstable',alpha=.75,ratios=[.8,1.01],worst_ratio=1.01),
          dict(candidate='stable',alpha=.5,ratios=[.97,.98],worst_ratio=.98),
          dict(candidate='smaller_change',alpha=.25,ratios=[.98,.984],worst_ratio=.984)]
    selected=choose_branch(rows)
    assert selected['candidate']=='smaller_change'
    assert choose_branch(rows[:1])['alpha']==0
    assert choose_branch([dict(candidate='negligible',alpha=.25,ratios=[.99,.998],worst_ratio=.998)])['alpha']==0


def test_branch_mixture_keeps_original_for_unselected_group():
    previous=np.array([100.,200.,300.]);seen=np.array([True,False,True])
    selection={'branches':{'seen':{'candidate':None,'alpha':0.},
                           'unseen':{'candidate':'new','alpha':.25}}}
    actual=apply(selection,previous,seen,{'new':np.array([50.,400.,150.])})
    np.testing.assert_array_equal(actual,[100.,250.,300.])
    np.testing.assert_array_equal(previous,[100.,200.,300.])
