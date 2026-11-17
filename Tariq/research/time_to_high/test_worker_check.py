"""Synthetic backend equivalence, including actual separate worker processes."""
import runpy
import time
from multiprocessing import TimeoutError
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).parent
    sources={k:(root/f).read_text() for k,f in [('reference','screen_runner.py'),('optimized','screen_throughput.py')]}
    ref=runpy.run_path(str(root/'screen_runner.py'))
    # Match Colab's dynamically embedded namespace, including process serialization.
    check={}
    exec(compile((root/'screen_worker_check.py').read_text(),'screen_worker_check.py','exec'),check)
    times=pd.date_range('2022-08-01','2023-07-01',freq='12h',tz='UTC')
    rng=np.random.default_rng(48)
    x=rng.normal(size=len(times)); x[::7]=np.nan
    y=rng.integers(0,18001,len(times)).astype(float)
    masks=ref['partition_masks'](times,18000,'2023-04-01','2023-07-01')
    config={'minimum_observed_training':20,'grid':[[2,64],[2,256],[3,64],[3,256],[5,64],[5,256]]}
    tasks=[(values,y,masks,config) for values in [x,np.ones(len(x)),np.full(len(x),np.nan)]]
    a=check['run_backend'](sources,tasks,'threading',2)
    b=check['run_backend'](sources,tasks,'loky',2)
    assert [check['scientific_case'](v) for v in a['cases']]==[check['scientific_case'](v) for v in b['cases']]
    assert a['cases'][1]['scores']['outer']['fallback']=='training_median_time_to_high'
    assert all(v['predictions']['outer']['rows']==int(masks['outer_valid'].sum()) for v in a['cases'])
    print('PASS: actual two-process versus two-thread execution; exact scores, selections and all prediction digests; missing data and fallback preserved.')
    original=check['worker_case']
    def delayed_case(*args):
        time.sleep(.2)
        return {}
    check['worker_case']=delayed_case
    try:
        try:
            check['run_backend'](sources,tasks[:2],'threading',2,timeout_seconds=.03)
        except TimeoutError:
            print('PASS: stalled task raises timeout; no success result returned.')
        else:
            raise AssertionError('Timeout failed')
    finally:
        check['worker_case']=original
        time.sleep(.3)  # Let the short synthetic sleepers finish; no native fits here.


if __name__=='__main__':
    main()
