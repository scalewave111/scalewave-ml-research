"""Non-fitting checks for configuration dispatch, guards and sequence timing."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Week 2/Code'))
import net_300h_refinement as m
assert m.PACKS==['leverage'] and len(m.CONFIG)==4
assert m.CONFIG['hgb_l1']['max_depth']==2 and m.CONFIG['hgb_shallow_l1']['max_depth']==1
assert m.neural.CONFIG['gru16']['decay']==.03 and m.neural.CONFIG['gru16_regularized']['decay']==.1
before=m.tree.CONFIG.copy();original=m._original_fit
try:
    m._original_fit=lambda x,y,spec:dict(depth=m.tree.CONFIG['max_depth'])
    assert m.fit(None,None,dict(model='learned',config='hgb_shallow_l1'))['depth']==1
    assert m.tree.CONFIG==before
    assert m.fit(None,None,dict(model='learned',config='hgb_l1'))['depth']==2
finally:m._original_fit=original
baseline=np.array([[[10.,200.],[9.,180.]]]*4)
assert m.legacy.assess(np.array([[8.,220.]]*4),baseline,[1]*4)['eligible']
assert not m.legacy.assess(np.array([[9.5,100.]]*4),baseline,[1]*4)['eligible']
# Build exact sequence snapshots from all leverage inputs on an unfiltered timeline.
from itertools import product
cols=m.tree.PRICE+[f'STATE__{f}_{stat}__{h}h' for f,stat,h in product(['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind'],['mean','deviation'],[24,300])]+[f'INTERACT__{f}__{h}h' for f,h in product(['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding'],[24,300])]
index=pd.date_range('2020-01-01',periods=400,freq='30min',tz='UTC')
x=pd.DataFrame({c:np.arange(len(index),dtype=float) for c in cols},index=index)
a=m.augment(x)
for lag in m.LAGS:
    np.testing.assert_allclose(a[f'SEQ{lag}__{cols[0]}'],x[cols[0]].reindex(index-pd.Timedelta(hours=lag)),equal_nan=True)
print('PASS: controls/alternatives,configuration dispatch/restoration,dualbaselineMAEguard,38input sequence snapshots; no fitting.')
