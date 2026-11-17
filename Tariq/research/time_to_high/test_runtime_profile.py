"""Synthetic timing attribution and unchanged-score check; no external access."""
import runpy
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root = Path(__file__).parent
ref = runpy.run_path(str(root/'screen_runner.py'))
opt = runpy.run_path(str(root/'screen_throughput.py'),init_globals={'REF':ref})
profile = runpy.run_path(str(root/'screen_runtime_profile.py'),init_globals={'REF':ref,'OPT':opt})
config = {'minimum_observed_training':20,'grid':[[2,64],[2,256],[3,64],[3,256],[5,64],[5,256]]}
times = pd.date_range('2022-08-01','2023-07-01',freq='12h',tz='UTC')
rng = np.random.default_rng(48)
x = rng.normal(size=len(times)); x[::7] = np.nan
y = rng.integers(0,18001,len(times)).astype(float)
task = (x,y,ref['partition_masks'](times,18000,'2023-04-01','2023-07-01'),config)
saved = []
def emit(key,score):
    time.sleep(.002)
    saved.append((key,score))
with threadpool_limits(limits=1):
    result,timing = profile['measured_fold'](task,emit)
    reference = opt['fold_task'](*task,False)
opt['assert_equal'](result,reference)
assert len(saved)==7
assert timing['model_fit_seconds']>0 and timing['preparation_seconds']>0
assert timing['checkpoint_emit_seconds']>=.01
assert timing['fit_and_scoring_seconds']>=timing['model_fit_seconds']
assert timing['fold_wall_seconds']>=timing['checkpoint_emit_seconds']
print('PASS: profiled scores/predictions match reference; fit, preparation and checkpoint delay attributed separately.')

# Exercise concurrent workers, including a second task reusing a pool thread.
# This was missing from the original profile regression test.
barrier = threading.Barrier(2)
def concurrent_case(index):
    if index < 2:
        barrier.wait(timeout=10)
    return profile['measured_fold'](task, lambda key, score: time.sleep(.001))
with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(concurrent_case, i) for i in range(3)]
    for future in futures:
        result, timing = future.result()
        opt['assert_equal'](result, reference)
        assert timing['model_fit_seconds'] > 0 and timing['checkpoint_emit_seconds'] > 0
for values in [np.ones(len(x)), np.full(len(x),np.nan)]:
    fallback_task = (values,*task[1:])
    result, timing = profile['measured_fold'](fallback_task, lambda key, score: None)
    opt['assert_equal'](result,opt['fold_task'](*fallback_task,False))
    assert timing['model_fit_seconds'] == 0
print('PASS: two concurrent workers, three tasks, worker reuse and fallback; no profiler tooling used.')
