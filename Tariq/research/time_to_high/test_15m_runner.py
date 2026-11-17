"""Synthetic resume, score equivalence and organized-report arithmetic checks."""
import copy
import json
import runpy
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root=Path(__file__).parent
ref=runpy.run_path(str(root/'screen_runner.py'))
opt=runpy.run_path(str(root/'screen_throughput.py'),init_globals={'REF':ref})
report=runpy.run_path(str(root/'timing_results_report.py'))
runner=runpy.run_path(str(root/'screen_15m_runner.py'),init_globals={'REF':ref,'OPT':opt,'REPORT':report})
config=json.loads((root/'screen_config.json').read_text()); config['minimum_observed_training']=20
times=pd.date_range('2022-04-22','2024-06-30',freq='12h',tz='UTC')
rng=np.random.default_rng(48); x=rng.normal(size=len(times)); x[::7]=np.nan
y=rng.integers(0,16,len(times)).astype(float)
masks=[ref['partition_masks'](times,15,*v) for v in config['folds']]
feature=config['pilot_features'][0]
job={'screen_signature':'test','feature':feature,'horizon_minutes':15,'X_cache_sha256':'x','Y_cache_sha256':'y','fits':{},'folds':{},'matrix':{}}
saved=[]
def checkpoint(value): saved.append(json.dumps(runner['pack'](value)))
with threadpool_limits(limits=1):
    complete=runner['fit_job'](copy.deepcopy(job),x,times,y,masks,config,checkpoint,float('inf'))
    for i,m in enumerate(masks):
        expected=opt['fold_task'](x,y,m,config,False)
        for key,score in expected['fits'].items():
            assert opt['scientific'](score)==opt['scientific'](complete['fits'][f'{feature}|15|{i}|{key}'])
assert len(complete['fits'])==35 and len(complete['folds'])==5
lagging={'complete_folds':1,'sha256':'older-partial-snapshot'}
assert runner['recover_indexed'](runner['pack'](complete),lagging,'test',feature+'|15')==complete
assert lagging['complete_folds']==5
try:
    runner['recover_indexed'](runner['pack'](complete),{'complete_folds':5,'sha256':'changed'},'test',feature+'|15')
except AssertionError: pass
else: raise AssertionError('Changed completed receipt accepted')
# Resume the snapshot after the first outer score was uploaded, before fold aggregation.
snapshot=next(raw for raw in saved if len(json.loads(raw)['job']['fits'])==7 and not json.loads(raw)['job']['folds'])
partial=runner['unpack'](snapshot,'test',feature,'x','y')
with threadpool_limits(limits=1):
    recovered=runner['fit_job'](partial,x,times,y,masks,config,lambda v:None,float('inf'))
for key in complete['fits']: assert opt['scientific'](complete['fits'][key])==opt['scientific'](recovered['fits'][key])
original=opt['prepared_fit']
opt['prepared_fit']=lambda *a: (_ for _ in ()).throw(AssertionError('Completed fit repeated'))
assert runner['fit_job'](recovered,x,times,y,masks,config,lambda v:None,float('inf'))==recovered
opt['prepared_fit']=original
fake=[0]
def expire(value): fake[0]=10
limited=runner['fit_job'](copy.deepcopy(job),x,times,y,masks,config,expire,5,clock=lambda:fake[0])
assert len(limited['fits'])==1 and not limited['folds']
try:
    runner['unpack'](snapshot,'wrong',feature,'x','y')
except AssertionError: pass
else: raise AssertionError('Identity mismatch accepted')
def fail(value): raise OSError('simulated checkpoint failure')
try:
    runner['fit_job'](copy.deepcopy(job),x,times,y,masks,config,fail,float('inf'))
except OSError: pass
else: raise AssertionError('Checkpoint failure swallowed')
# Only saved compact pilot scores used for report checks; no source market data.
baseline=json.loads((root/'BASELINE_RESULTS.json').read_text())
pilot=json.loads((root/'pilot_fit_scores.json').read_text())
frozen=json.loads((root/'screen_config.json').read_text())
rows,folds,policy=report['tables'](frozen,baseline,pilot)
assert len(rows)==386 and len(folds)==15 and sum(r['status']=='COMPLETE' for r in rows)==3
assert [r['rank'] for r in rows[:3]]==[1,2,3] and all(r['rank']=='' for r in rows[3:])
for row in rows[:3]:
    expected=pilot[row['feature']+'|15']['matrix'][row['feature']+'|15']['MAE_minutes']
    assert abs(row['MAE_minutes']-expected)<1e-12
print('PASS: 35-fit synthetic equivalence, aggregate recovery, zero-fit completed resume, budget stop, identity rejection, checkpoint failure and 386-row report arithmetic.')
