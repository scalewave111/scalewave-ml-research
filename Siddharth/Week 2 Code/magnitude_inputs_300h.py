"""Verified 300h inputs and causal price features for the expanded model search."""
import argparse
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import improve_excursion_features_300h as prior
from long_horizon_features import build
import long_horizon_features as feature_module

b = prior.base
REFERENCE = b.HERE/'experiments/excursion_pipeline_300h_v1/20261113T115754520122Z/price_followup/20261113T121655587383Z'


def load(args):
    contract = json.loads((REFERENCE/'CONTRACT.json').read_text())
    done = json.loads((REFERENCE/'COMPLETE.json').read_text())
    b.check(contract['horizon_minutes'] == 18000 and done['status'] == 'COMPLETE', 'Bad reference')
    b.check(b.sha(REFERENCE/'predictions.csv') == done['predictions_sha256'], 'Reference hash changed')
    args.data_dir = Path(contract['arguments']['data_dir'])
    args.y_dir = Path(contract['arguments']['y_dir'])
    original, y, dev, files = b.load(args)
    original.index = original.index.as_unit('ns')
    y.index, dev = y.index.as_unit('ns'), dev.as_unit('ns')
    returns = Path(contract['arguments']['returns'])
    for path in files+[returns]:
        b.check(b.sha(path) == contract['source_hashes'][str(path)], 'Input changed: '+str(path))
    r = prior.canonical.load_returns(returns)
    x, dictionary = build(r, original.index, 300)
    x.attrs['excluded_labels'] = original.attrs['excluded_labels']
    cut = original.index[len(original)//2]
    keys = original.index[original.index <= cut][-10:]
    pd.testing.assert_frame_equal(build(r, keys, 300)[0], build(r[r.index <= cut], keys,300)[0])
    changed = r.copy(); changed.loc[changed.index >= cut] = 1e6
    pd.testing.assert_frame_equal(build(r, keys,300)[0], build(changed,keys,300)[0])
    old = pd.read_csv(REFERENCE/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time,utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue,utc=True)
    b.check(len(old)==2*len(dev) and not old.duplicated(['decision_time','target']).any(), 'Reference coverage')
    for target in b.TARGETS:
        part = old[old.target.eq(target)].set_index('decision_time').sort_index()
        b.check(part.index.equals(dev), 'Reference timestamps')
        np.testing.assert_allclose(part.actual,y.loc[dev,target],rtol=0,atol=1e-10)
        for i, issue in enumerate(b.ISSUES):
            end = b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            sub = part.loc[(part.index>=issue)&(part.index<end)]
            b.check(sub.issue.eq(issue).all(), 'Reference issuance')
            train = y.loc[y.index+pd.Timedelta(hours=300)<issue,target]
            np.testing.assert_allclose(sub.training_median,train.median(),rtol=0,atol=1e-8)
            np.testing.assert_allclose(sub.training_mean,train.mean(),rtol=0,atol=1e-8)
    return x,y,dev,old,dictionary,files+[returns,REFERENCE/'CONTRACT.json',REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv']

