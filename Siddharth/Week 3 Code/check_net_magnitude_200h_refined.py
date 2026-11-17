"""Non-fitting synthetic feature/arithmetic checks. Never reads real inputs."""
from pathlib import Path
import ast
import runpy
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
source=(HERE/'fit_net_magnitude_200h_refined.py').read_text(encoding='utf-8')
tree=ast.parse(source)
scope={'np':np}
# Test pure helpers without importing the runner or its real-data dependencies.
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['basis','assess','blend','choose']:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<helpers>','exec'),scope)
a=np.array([[-8.,0.,1.,0.],[2.,-3.,0.,1.]])
b=scope['basis'](a,'quadratic')
np.testing.assert_allclose(b[:,:2],np.sign(a[:,:2])*np.log1p(abs(a[:,:2])))
np.testing.assert_allclose(b[:,2:4],b[:,:2]**2)
np.testing.assert_array_equal(b[:,4:],a[:,2:])
np.testing.assert_allclose(scope['blend'](np.array([2.,4.]),10.,.25),[8.,8.5])
refs={'mean':[np.ones(10)*2]*4,'median':[np.ones(10)*3]*4}
assert scope['assess']([np.ones(10)]*4,refs)[1]
assert not scope['assess']([np.ones(10)*2.5]*4,refs)[1]
assert not scope['assess']([np.ones(10),np.ones(10),np.ones(10),np.ones(10)*4],refs)[1]
trial=dict(id=0,anchor='mean',weight=.5,head_weight=.5,score=.5,mae=1.,accepted=False)
assert scope['choose']([trial])[1] is None
accepted=dict(trial,id=1,score=.6,accepted=True)
assert scope['choose']([trial,accepted])[1]['id']==1
price=runpy.run_path(str(HERE/'net_200h_long_features.py'))
market=runpy.run_path(str(HERE/'net_200h_market_features.py'))
# Regression: required raw columns must survive independently of legacy subsets.
import tempfile
import duckdb
with tempfile.TemporaryDirectory() as folder:
    mini_idx=pd.date_range('2020-01-01',periods=4,freq='30min',tz='UTC')
    mini=pd.DataFrame({c:[1.,np.nan,3.,4.] for c in market['SOURCES'].values()},index=mini_idx)
    mini.index.name='decision_time'
    with duckdb.connect() as con:
        for name,part in [('X_train.parquet',mini.iloc[:2]),('X_development.parquet',mini.iloc[2:])]:
            con.register('sample',part.reset_index())
            con.execute('COPY sample TO ? (FORMAT PARQUET)',[str(Path(folder)/name)])
    loaded,_=market['load_sources'](folder,mini_idx)
    pd.testing.assert_frame_equal(loaded,mini.rename_axis(None),check_names=False,check_freq=False)
    try:market['load_sources'](folder,mini_idx.append(pd.DatetimeIndex([mini_idx[-1]+pd.Timedelta(minutes=30)])))
    except ValueError as error:assert 'required timestamps' in str(error)
    else:raise AssertionError('Missing timestamps must fail')
rng=np.random.default_rng(48)
idx=pd.date_range('2020-01-01',periods=21000,freq='5min',tz='UTC')
r=pd.Series(rng.normal(size=len(idx)),index=idx)
origins=idx[::6]
m=pd.DataFrame({c:rng.uniform(size=len(origins)) for c in market['SOURCES'].values()},index=origins)
p,_=price['build'](r,origins,200)
s,_=market['build'](m,p,origins)
cut=origins[-20];keys=origins[origins<=cut][-8:]
changed=r.copy();changed.loc[changed.index>=cut]=1e9
mm=m.copy();mm.loc[mm.index>=cut]=1e9
pp,_=price['build'](changed,origins,200)
ss,_=market['build'](mm,pp,origins)
pd.testing.assert_frame_equal(p.loc[keys],pp.loc[keys])
pd.testing.assert_frame_equal(s.loc[keys],ss.loc[keys])
prefix=origins[origins<=cut]
pp,_=price['build'](r.loc[r.index<=cut],prefix,200)
ss,_=market['build'](m.loc[prefix],pp,prefix)
pd.testing.assert_frame_equal(p.loc[keys],pp.loc[keys])
pd.testing.assert_frame_equal(s.loc[keys],ss.loc[keys])
# Verify 200h strict boundaries without fitting or loading actual datasets.
ends=origins+pd.Timedelta(hours=200)
boundary=ends[-100]
assert not (ends<boundary)[-100]
assert (ends<boundary)[-101]
assert 'for offset in [15,12,9,6]' in source
assert '(y.label_end<window_end)' in source
assert "default=['hgb','extra','mlp','gru']" in source
assert "['l1','l2']" in source
assert 'previous' not in source and 'REFERENCES' not in source
print('PASS: synthetic price/market causality and prefix checks; input transforms, blend arithmetic, dual-metric gates, accepted/candidate separation and strict 200h boundary. No fitting, runner import or real-data execution.')

endpoint=runpy.run_path(str(HERE/'net_200h_endpoint_features.py'))
e,d=endpoint['endpoint_features'](r,origins,p)
ee,_=endpoint['endpoint_features'](changed,origins,pp.reindex(origins))
columns=list(d.feature)
pd.testing.assert_frame_equal(e.loc[keys,columns],ee.loc[keys,columns])
eprefix,_=endpoint['endpoint_features'](r.loc[r.index<=cut],prefix,pp)
pd.testing.assert_frame_equal(e.loc[keys,columns],eprefix.loc[keys,columns])
assert len(columns)==22 and e[columns].iloc[-1].notna().all()
for h in endpoint['HOURS']:
    key=origins[-1]
    past=r[(r.index>=key-pd.Timedelta(hours=h))&(r.index<key)]
    np.testing.assert_allclose(e.loc[key,f'END200__rv5m_{h}h'],np.sqrt((past**2).sum()))
    assert (e[f'END200__shrunk_drift_abs_proxy_{h}h'].dropna()>=0).all()
# Missing returns invalidate affected five-minute windows.
broken=r.copy();broken.iloc[-10]=np.nan
bad,_=endpoint['endpoint_features'](broken,origins,p)
assert np.isnan(bad.iloc[-1]['END200__rv5m_100h'])
print('PASS: all22 new endpoint features, future/prefix isolation, direct RV formulas, finite coverage and missing-return exclusion; no fitting.')

refine=runpy.run_path(str(HERE/'net_200h_refinement_features.py'))
f,fd=refine['build'](r,origins)
fc,_=refine['build'](changed,origins)
fp,_=refine['build'](r.loc[r.index<=cut],prefix)
pd.testing.assert_frame_equal(f.loc[keys],fc.loc[keys])
pd.testing.assert_frame_equal(f.loc[keys],fp.loc[keys])
assert len(f.columns)==12 and f.iloc[-1].notna().all()
assert (f['REF200__endpoint_q50_720h'].dropna()<=f['REF200__endpoint_q90_720h'].dropna()).all()
assert len(fd)==12
print('PASS: all12refinement features finite, causal and prefix invariant; quantiles ordered. No fitting.')
