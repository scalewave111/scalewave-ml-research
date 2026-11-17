"""Synthetic checks only; no Drive access, market data or artifact creation."""
import io
import json
import zipfile
import runpy
from pathlib import Path
import numpy as np

s=runpy.run_path(str(Path(__file__).with_name('screen_runner.py')),run_name='test')
s['self_test']()

# Real ZIP local headers/CRC with both supported compression types, using memory as remote bytes.
for method in [zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED]:
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',method) as archive:archive.writestr('test.json',json.dumps({'test':'value'}))
    raw=buf.getvalue();remote=s['RemoteArchive'].__new__(s['RemoteArchive'])
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:remote.members={v.filename:v for v in archive.infolist()}
    remote.read_range=lambda start,n:raw[start:start+n]
    assert json.loads(remote.read('test.json'))=={'test':'value'}

class MemoryStore:
    def __init__(self):self.files={}
    def read(self,name):return self.files.get(name)
    def write(self,name,data,mime):self.files[name]=data
store=MemoryStore();identity={'source':'synthetic','cadence':30}
meta={'identity':identity,'completed':['2022-04'],'member_hashes':{}}
times=np.array([1,2],dtype=np.int64);values=np.array([[1.,np.nan],[2.,3.]])
s['write_cache'](store,'cache',meta,times,values)
m,t,x=s['read_cache'](store,'cache',identity)
assert m['completed']==['2022-04'] and np.array_equal(t,times) and np.allclose(x,values,equal_nan=True)
try:s['read_cache'](store,'cache',{'source':'changed'});raise RuntimeError('Changed cache accepted')
except AssertionError:pass

# Recovery after folds were saved but the aggregate matrix was not yet saved.
state={'folds':{'a|15|0':{'validation_rows':2,'absolute_error_sum':4.,'status':'PASS'},
                 'a|15|1':{'validation_rows':3,'absolute_error_sum':9.,'status':'INNER_FEATURE_INELIGIBLE'}},'matrix':{}}
baseline={'completed':{'15':{'pooled':{'median':{'MAE_minutes':3.}}}}}
config={'candidate_features':['a'],'horizons':[15],'folds':[1,2]}
s['update_matrix'](state,baseline,config);row=state['matrix']['a|15']
assert row['rows']==5 and row['MAE_minutes']==2.6 and row['folds_using_fallback']==1
print('PASS: remote ZIP extraction/CRC, compact cache resume/invalidation, weighted scores and interrupted aggregation recovery.')

# Projected F4Z ingestion and cache resume on synthetic Parquet data, never real data.
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
class MemoryArchive:
    def __init__(self):self.members={};self.reads=0
    def read(self,name):self.reads+=1;return self.members[name]
archive=MemoryArchive();field='activity__5m__value';base=field[:-7];hashes={}
for day in pd.date_range('2022-04-22','2022-04-30').strftime('%Y-%m-%d'):
    times=pd.date_range(pd.Timestamp(day,tz='UTC')+pd.Timedelta(minutes=1),periods=1440,freq='min')
    available=list(times);available[59]=times[59]+pd.Timedelta(minutes=1)
    valid=np.ones(1440,dtype=bool);valid[29]=False
    counts=np.full(1440,1000);counts[89]=999
    status=['PROVISIONAL']*1440;status[119]='UNKNOWN'
    frame=pd.DataFrame({'decision_time':times,'canonical_symbol':'BTCUSDT','available_after':available,
                        field:np.arange(1440,dtype=float),base+'__valid':valid,base+'__baseline_n':counts,
                        base+'__scientific_status':status})
    buf=io.BytesIO();pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),buf)
    name=f'FORGE_PRO_V1/dates/date={day}/F4Z/family=flow/source_clock=1m/symbol=BTCUSDT/part.parquet'
    archive.members[name]=buf.getvalue();hashes[name]=s['sha'](buf.getvalue())
config={'months':['2022-04'],'origin_end_exclusive':'2024-07-01T00:00:00Z','reserve_start':'2024-07-24T00:01:00Z'}
store=MemoryStore();columns=['F4Z__flow__'+field]
t,x,h=s['load_family'](archive,store,'flow',columns,{'id':'synthetic'},hashes,config,2)
assert len(t)==432 and np.isnan(x).sum()==36 and archive.reads==9
t2,x2,h2=s['load_family'](archive,store,'flow',columns,{'id':'synthetic'},hashes,config,2)
assert archive.reads==9 and h==h2 and np.allclose(x,x2,equal_nan=True)
yframe=pd.DataFrame({'canonical_symbol':'BTCUSDT','decision_time':t,'horizon_minutes':15,
                     'time_to_high_min':7.,'label_status':'VALID'})
yframe.loc[0,'label_status']='INVALID_PRICE_PATH';yframe.loc[0,'time_to_high_min']=np.nan
buf=io.BytesIO();pq.write_table(pa.Table.from_pandas(yframe,preserve_index=False),buf)
target=MemoryArchive();name='root/y/horizon_minutes=15/symbol=BTCUSDT/source_month=2022-04/part.parquet';target.members[name]=buf.getvalue()
manifest={name:s['sha'](buf.getvalue())}
ty,y,hy=s['load_target'](target,store,'root/',manifest,15,{'id':'synthetic'},config)
assert len(y)==431 and (y==7).all() and target.reads==1
s['load_target'](target,store,'root/',manifest,15,{'id':'synthetic'},config)
assert target.reads==1
print('PASS: F4Z validity/availability/status gates, target-invalid exclusion, timestamp alignment and zero-read completed-cache resume.')
