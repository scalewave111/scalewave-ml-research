"""Execute the delivered notebook on synthetic Parquet fixtures, entirely inside the workspace."""
from pathlib import Path
import json,sys,types,decimal
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')

ROOT=Path(__file__).resolve().parents[1]
TEST=ROOT/'Reports/F4Z_Notebook_Verification'
DATA=TEST/'Data'
DATA.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(77)
times=pd.date_range('2022-04-22', '2025-01-22', freq='2h', tz='UTC')
n=len(times)
event=rng.random(n)<.16
negative=(~event)&(rng.random(n)<.07)
y=np.where(event,650.,np.where(negative,-650.,0.))
df=pd.DataFrame({'decision_time':times,'canonical_symbol':'BTCUSDT','signed_return':y,
                 'horizon_minutes':'2880','label_status':'VALID',
                 'f4z__flow__available_after':times-pd.Timedelta(seconds=1)})
for i in range(16):
    f=f'f4z__flow__synthetic_{i}_rz__1h'
    x=rng.normal(size=n)+event*(3-i*.07)
    if i==13:x=np.ones(n)
    if i==14:x=rng.normal(size=n)
    if i==15:x=np.where(event,rng.normal(0,.1,n),rng.choice([-3.,3.],n)+rng.normal(0,.1,n))
    df[f+'__value']=x
    df[f+'__valid']=False if i==12 else True
# Test Arrow decimal values and nullable validity flags.
decimal_col='f4z__flow__synthetic_0_rz__1h__value'
df[decimal_col]=[decimal.Decimal(f'{v:.6f}') for v in df[decimal_col]]
valid_col='f4z__flow__synthetic_1_rz__1h__valid'
df[valid_col]=df[valid_col].astype('boolean')
df.loc[::71,valid_col]=pd.NA
df.loc[::91,'f4z__flow__available_after']=df.loc[::91,'decision_time']+pd.Timedelta(seconds=1)
filenames={
 'NON_EXTREME':'BTCUSDT_signed_return_between_minus500_and_plus500_20261027_101854_1aefb791.parquet',
 'EXTREME':'BTCUSDT_signed_return_above_500_or_below_minus500_20261027_093028_e1453352.parquet'}
for name,filename in filenames.items():
    mask=(np.abs(y)<=500) if name=='NON_EXTREME' else (np.abs(y)>500)
    pq.write_table(pa.Table.from_pandas(df.loc[mask],preserve_index=False),DATA/filename,row_group_size=700)

# Emulate only Colab's mount operation; the numerical notebook code runs unchanged.
google=types.ModuleType('google');colab=types.ModuleType('google.colab')
colab.drive=types.SimpleNamespace(mount=lambda *a,**k:None)
google.colab=colab;sys.modules['google']=google;sys.modules['google.colab']=colab
# The local test runner is not a notebook kernel; emulate display without changing calculations.
ipython=types.ModuleType('IPython');ipython.get_ipython=lambda:None;ipython.version_info=(9,0)
ipdisplay=types.ModuleType('IPython.display');ipdisplay.display=lambda *a,**k:None
ipdisplay.Markdown=lambda x:x
sys.modules['IPython']=ipython;sys.modules['IPython.display']=ipdisplay
notebook=json.loads((ROOT/'Notebooks/03_Alternative_F4Z_Top10_Association.ipynb').read_text(encoding='utf-8'))
env={}
for c in notebook['cells']:
    if c['cell_type']!='code':continue
    src=''.join(c['source'])
    print('RUN',src.splitlines()[0],flush=True)
    if src.startswith('# 2.'):
        src=src.replace("Path('/content/drive/MyDrive/ScaleWave')",f'Path({str(TEST)!r})')
    exec(compile(src,'notebook_cell','exec'),env)
    if src.startswith('# 1.'):
        env['display']=lambda *args,**kwargs:None
    if src.startswith('# 2.'):
        # Smaller synthetic sample, same algorithm and temporal boundaries.
        env['CFG'].update({'min_condition_rows':60,'min_control_rows':60,
                          'min_rows_per_group_per_quarter':10,'bootstrap_replicates':100})

result=env['FINAL'];all_results=env['ALL_RESULTS']
assert len(result)==10
assert all_results.set_index('feature').loc['f4z__flow__synthetic_12_rz__1h__value','status']=='too_few_valid_learning_rows'
assert all_results.set_index('feature').loc['f4z__flow__synthetic_13_rz__1h__value','status']=='constant_learning_feature'
assert result['rank_adjusted_rd_pp'].gt(0).all()
assert result['validation_adjusted_rd_pp'].gt(0).all()
assert result['rank_rd_p05_pp'].is_monotonic_decreasing
assert env['audit']['unavailable_rows'].sum()>0
assert env['TOP10']['feature'].tolist()==result['feature'].tolist()
# Perturb future feature values: the selected learning interval must be unchanged.
f='f4z__flow__synthetic_0_rz__1h__value'
x=env['read_feature'](f)
a,_=env['learn_band'](x)
x[~env['PERIOD_MASKS']['learn']]=999999.
b,_=env['learn_band'](x)
assert (a['low'],a['high'])==(b['low'],b['high'])
# Confirm target sorting and feature reads align despite partitioned source files.
original=np.array([float(v) for v in df[decimal_col]])
good=env['AVAILABLE']['f4z__flow__available_after']
actual=env['read_feature'](f)
assert np.allclose(actual[good],original[good])
assert np.isnan(actual[~good]).all()
# Check frozen CSV preserves infinite edges for exact replay.
frozen=pd.read_csv(env['RUN_DIR']/'top10_frozen_discovery.csv')
assert np.allclose(frozen['low'],result['low']) and np.allclose(frozen['high'],result['high'])
verification={'status':'passed','synthetic_features':16,'reported_features':len(result),
 'checks':['full notebook execution','decimal Parquet conversion','nullable validity flags',
           'late availability exclusion','constant/invalid feature rejection','chronological alignment',
           'learning-band independence from future feature values','frozen rank after validation',
           'CSV interval replay'],
 'limitation':'Original company Parquet datasets are not present locally; full-scale Colab runtime not measured.'}
(TEST/'verification.json').write_text(json.dumps(verification,indent=2))
print('PASS:',json.dumps(verification),flush=True)
