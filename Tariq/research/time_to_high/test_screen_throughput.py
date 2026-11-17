"""Small synthetic regression tests, no market data or network."""
import json
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).parent
ref = runpy.run_path(str(ROOT/'screen_runner.py'))
opt = runpy.run_path(str(ROOT/'screen_throughput.py'), init_globals={'REF': ref})
config = {'minimum_observed_training': 20, 'grid': [[2,64],[2,256],[3,64],[3,256],[5,64],[5,256]]}
times = pd.date_range('2022-08-01', '2023-07-01', freq='12h', tz='UTC')
masks = ref['partition_masks'](times, 18000, '2023-04-01', '2023-07-01')
rng = np.random.default_rng(48)
y = rng.integers(0,18001,len(times)).astype(float)
x = rng.normal(size=len(times)); x[::7] = np.nan
cases = [x, np.ones(len(x)), np.full(len(x), np.nan)]
# Exactly below minimum, plus missing-only evaluation with usable training.
sparse = np.full(len(x), np.nan); sparse[:19] = np.arange(19)
cases.append(sparse)
missing_eval = x.copy(); missing_eval[masks['outer_valid']] = np.nan
cases.append(missing_eval)
with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=2) as pool:
    for values in cases:
        a = pool.submit(opt['fold_task'], values, y, masks, config, False)
        b = pool.submit(opt['fold_task'], values, y, masks, config, True)
        opt['assert_equal'](a.result(), b.result())
assert opt['fold_task'](cases[1], y, masks, config, True)['fits']['outer']['fallback'] == 'training_median_time_to_high'

# Immutable ordered snapshots, bounded queue and propagation of upload failures.
written = []
gate = threading.Event()
def write(name, raw):
    gate.wait(5)
    written.append(json.loads(raw))
writer = opt['OrderedWriter'](write, capacity=2)
value = {'n': 1}; writer.submit('test', value)
value['n'] = 2; writer.submit('test', value)
value['n'] = 3; gate.set(); writer.submit('test', value); writer.close()
assert written == [{'n':1},{'n':2},{'n':3}]
def fail(name, raw): raise OSError('simulated upload failure')
writer = opt['OrderedWriter'](fail)
writer.submit('test', {})
try:
    writer.close()
except OSError:
    pass
else:
    raise AssertionError('Upload failure swallowed')
print('PASS: exact scores, predictions, selections, fallback/missing rows, concurrent fits, ordered immutable checkpoints and upload failure propagation.')
