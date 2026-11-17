"""Synthetic 1h equivalence, resume, target loading and cache write isolation."""
import copy
import io
import json
import runpy
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

root=Path(__file__).parent
ref=runpy.run_path(str(root/'screen_runner.py'))
opt=runpy.run_path(str(root/'screen_throughput.py'),init_globals={'REF':ref})
runner=runpy.run_path(str(root/'screen_1h_runner.py'),init_globals={'REF':ref,'OPT':opt})
config=json.loads((root/'screen_config.json').read_text()); config['minimum_observed_training']=20
times=pd.date_range('2022-04-22','2024-06-30',freq='12h',tz='UTC')
rng=np.random.default_rng(48); x=rng.normal(size=len(times)); x[::7]=np.nan
y=rng.integers(0,61,len(times)).astype(float)
masks=[ref['partition_masks'](times,60,*v) for v in config['folds']]
feature=config['pilot_features'][0]
job={'screen_signature':'test','feature':feature,'horizon_minutes':60,'X_cache_sha256':'x','Y_cache_sha256':'y','fits':{},'folds':{},'matrix':{}}
saved=[]
with threadpool_limits(limits=1):
    complete=runner['fit_job'](copy.deepcopy(job),x,times,y,masks,config,lambda v:saved.append(json.dumps(runner['pack'](v))),float('inf'))
    for i,m in enumerate(masks):
        expected=opt['fold_task'](x,y,m,config,False)
        for key,score in expected['fits'].items():
            assert opt['scientific'](score)==opt['scientific'](complete['fits'][f'{feature}|60|{i}|{key}'])
    snapshot=next(s for s in saved if len(json.loads(s)['job']['fits'])==7 and not json.loads(s)['job']['folds'])
    recovered=runner['fit_job'](runner['unpack'](snapshot,'test',feature,'x','y'),x,times,y,masks,config,lambda v:None,float('inf'))
for key in complete['fits']: assert opt['scientific'](complete['fits'][key])==opt['scientific'](recovered['fits'][key])
original=opt['prepared_fit']; opt['prepared_fit']=lambda *a: (_ for _ in ()).throw(AssertionError('Repeated fit'))
runner['fit_job'](recovered,x,times,y,masks,config,lambda v:None,float('inf')); opt['prepared_fit']=original
edge=pd.DatetimeIndex(['2023-03-31T22:59:00Z','2023-03-31T23:00:00Z'])
assert ref['partition_masks'](edge,60,'2023-04-01','2023-07-01')['outer_train'].tolist()==[True,False]

class MemoryStore:
    def __init__(self): self.files={}; self.writes=[]
    def read(self,name): return self.files.get(name)
    def write(self,name,data,mime='application/json'): self.files[name]=data; self.writes.append(name)
    def json(self,name,obj): self.write(name,json.dumps(obj).encode())
shared,private=MemoryStore(),MemoryStore()
identity={'source':'synthetic'}
meta={'identity':identity,'completed':['2022-04'],'member_hashes':{}}
ref['write_cache'](shared,'SCREEN_X_F4Z_flow.npz',meta,np.array([1,2]),np.array([[1.],[2.]]))
shared.writes.clear()
store=runner['IsolatedStore'](private,shared,['2022-04'])
m,t,v=ref['read_cache'](store,'SCREEN_X_F4Z_flow.npz',identity)
assert m['completed']==['2022-04'] and not private.files and not shared.writes
store.write('SCREEN_X_F4Z_flow.npz',b'private snapshot')
assert shared.files['SCREEN_X_F4Z_flow.npz']!=private.files['SCREEN_X_F4Z_flow.npz'] and not shared.writes
meta['completed']=[]
ref['write_cache'](shared,'SCREEN_X_F4Z_price.npz',meta,np.array([1]),np.array([[1.]])); shared.writes.clear()
assert store.read('SCREEN_X_F4Z_price.npz') is None
shared.files['SCREEN_1h_RESULTS.json']=b'shared must not be used'
assert store.read('SCREEN_1h_RESULTS.json') is None

class Archive:
    def __init__(self,files): self.files=files; self.reads=0
    def read(self,name): self.reads+=1; return self.files[name]
small={'months':['2022-04'],'origin_end_exclusive':'2024-07-01T00:00:00Z','reserve_start':'2024-07-24T00:01:00Z'}
stamp=pd.date_range('2022-04-22',periods=4,freq='30min',tz='UTC')
frame=pd.DataFrame({'canonical_symbol':'BTCUSDT','decision_time':stamp,'horizon_minutes':60,'time_to_high_min':[0.,12.,59.,60.],'label_status':'VALID'})
buf=io.BytesIO(); pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),buf)
name='root/y/horizon_minutes=60/symbol=BTCUSDT/source_month=2022-04/part.parquet'
archive=Archive({name:buf.getvalue()}); manifest={name:ref['sha'](buf.getvalue())}
target_store=runner['IsolatedStore'](MemoryStore(),shared,small['months'])
t,y,h=ref['load_target'](archive,target_store,'root/',manifest,60,{'id':'synthetic'},small)
assert len(y)==4 and (y==frame.time_to_high_min.to_numpy()).all() and not shared.writes
ref['load_target'](archive,target_store,'root/',manifest,60,{'id':'synthetic'},small)
assert archive.reads==1
report=runpy.run_path(str(root/'timing_results_report.py'))
rows,folds,policy=report['tables'](json.loads((root/'screen_config.json').read_text()),json.loads((root/'BASELINE_RESULTS.json').read_text()),json.loads((root/'pilot_fit_scores.json').read_text()),60)
assert len(rows)==386 and all(r['status']=='PENDING' for r in rows) and not folds
print('PASS: 1h five-fold equivalence, resume, strict 60m purge, private writes, shared complete-cache reuse, incomplete-cache rejection, target range-loader resume and pending report.')
