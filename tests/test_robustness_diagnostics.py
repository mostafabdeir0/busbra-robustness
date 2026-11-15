"""Focused tests for severity diagnostic's data-integrity and flip logic."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.robustness_diagnostics import parse_flags, check_tables, feature_drift, flip_discordance


def test_parse_flags_never_treats_false_as_true():
    assert parse_flags(pd.Series(['True','False',True,False])).tolist()==[True,False,True,False]
    with pytest.raises(ValueError):
        parse_flags(pd.Series(['unknown']))


def synthetic_features():
    n=len(FEATURE_COLUMNS)
    clean=pd.DataFrame([{'ID':'a','Case':1,'split':'development','dev_fold':0,
                         **dict(zip(FEATURE_COLUMNS,np.arange(n,dtype=float)+1))},
                        {'ID':'b','Case':2,'split':'development','dev_fold':1,
                         **dict(zip(FEATURE_COLUMNS,np.arange(n,dtype=float)+2))}])
    pert=[]
    for _,row in clean.iterrows():
        for cond in ('erosion_mild','erosion_moderate'):
            x=row.to_dict(); x['condition']=cond; x['valid']=True; x['shape_circularity']+=.1 if cond.endswith('mild') else .2
            pert.append(x)
    return clean,pd.DataFrame(pert)


def test_feature_drift_is_positive_and_respects_original():
    a,b=synthetic_features(); before=a.copy(deep=True)
    out=feature_drift(a,check_tables(a,b))
    x=out.loc[(out.condition=='erosion_mild') & (out.feature=='shape_circularity')].iloc[0]
    assert x.median_abs_delta==pytest.approx(.1)
    assert x.median_abs_delta_dev_sd==pytest.approx(.2)
    pd.testing.assert_frame_equal(a,before)


def test_metadata_mismatch_raises():
    a,b=synthetic_features(); b.loc[0,'Case']=999
    with pytest.raises(ValueError,match='metadata mismatch'):
        check_tables(a,b)


def test_mild_only_flip_detection():
    records=[]
    for case,truth,clean,minor,major in ((1,0,0,1,0),(2,1,1,0,0),(3,1,1,1,0),(4,0,1,1,1)):
        for cond,value in [('erosion_mild',minor),('erosion_moderate',major),('dilation_mild',minor),('dilation_moderate',major)]:
            records.append({'model':'logreg','condition':cond,'Case':case,'actual':truth,
                            'eligible':True,'clean_predicted':clean,'perturbed_predicted':value})
    summary,details=flip_discordance(pd.DataFrame(records))
    r=summary.loc[summary.family=='erosion'].iloc[0]
    assert r.eligible_both==4 and r.mild_only_flips==1 and r.moderate_only_flips==1
    assert r.both_flip==1 and r.neither_flip==1
    assert set(details.Case)=={1,3}


def test_duplicate_id_condition_is_rejected():
    a,b=synthetic_features(); b=pd.concat([b,b.head(1)],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):
        check_tables(a,b)
