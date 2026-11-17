"""Compare execution backends only; no experiment settings are changed."""
import hashlib
import json
import os
import platform
import time
import threading


def worker_case(sources, task, label='case'):
    # Executable source is packaged with hashes in the notebook. Each worker gets
    # arrays and pure model code only: no Drive clients, credentials or file writes.
    import numpy as np
    ref = {}
    exec(compile(sources['reference'], 'screen_runner.py', 'exec'), ref)
    opt = {'REF':ref}
    exec(compile(sources['optimized'], 'screen_throughput.py', 'exec'), opt)
    start, cpu = time.perf_counter(), time.thread_time()
    print(label+' started', flush=True)
    def emit(key, score):
        print(label+' '+key+' complete', flush=True)
    result = opt['fold_task'](*task, True, emit)
    compact = {'selected':result['selected'],
               'scores':{k:opt['scientific'](v) for k,v in result['fits'].items()},
               'predictions':{}}
    for key, values in result['predictions'].items():
        if values is None:
            compact['predictions'][key] = None
        else:
            values = np.ascontiguousarray(values, dtype='<f8')
            compact['predictions'][key] = {'rows':len(values), 'sha256':hashlib.sha256(values.tobytes()).hexdigest()}
    compact['timing'] = {'task_wall_seconds':time.perf_counter()-start,
                         'task_thread_cpu_seconds':time.thread_time()-cpu}
    return compact


def scientific_case(case):
    return {k:v for k,v in case.items() if k != 'timing'}


def run_backend(sources, tasks, backend, workers, timeout_seconds=240):
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    # Initialize tree imports in the parent before starting threads or setting limits.
    from sklearn.tree import DecisionTreeRegressor
    assert backend in ('threading','loky') and workers in (1,2)
    start = time.perf_counter()
    # Disable automatic memmapping: no dataset/prediction files on worker disks.
    options = dict(backend=backend, n_jobs=workers, max_nbytes=None)
    if backend == 'loky':
        options['inner_max_num_threads'] = 1
    print(f'Starting {backend}: {workers} workers; {timeout_seconds}s task timeout',flush=True)
    done = threading.Event()
    def heartbeat():
        while not done.wait(20):
            print(f'{backend}: still waiting; {time.perf_counter()-start:.0f}s elapsed',flush=True)
    monitor = threading.Thread(target=heartbeat,daemon=True)
    monitor.start()
    try:
        with threadpool_limits(limits=1), parallel_config(**options):
            result = Parallel(batch_size=1, pre_dispatch=workers, timeout=timeout_seconds)(
                delayed(worker_case)(sources, task, f'{backend} case {i+1}/{len(tasks)}')
                for i,task in enumerate(tasks))
    except BaseException:
        print('Comparison incomplete. Restart the Colab session before retrying: interrupted thread fits may still be running. No result is approved.',flush=True)
        raise
    finally:
        done.set()
        monitor.join(timeout=1)
    return {'wall_seconds_including_dispatch':time.perf_counter()-start, 'cases':result}


def run_check(config, expected_identity, sources, hashes):
    import numpy as np
    import pandas as pd
    import sklearn
    import pyarrow
    import joblib
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    assert all(hashlib.sha256(source.encode()).hexdigest()==hashes[k] for k,source in sources.items())
    auth.authenticate_user()
    print('Authentication complete; checking saved identities and receipts.',flush=True)
    credentials,_ = google.auth.default()
    service = build('drive','v3',credentials=credentials,cache_discovery=False)
    store = REF['DriveStore'](service,config['folder'])
    saved = json.loads(store.read('SCREEN_RESULTS.json'))
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, pyarrow=pyarrow.__version__)
    assert saved['identity']==expected_identity and REF['signature'](expected_identity)==saved['signature']
    assert expected_identity['config']==config and expected_identity['versions']==versions, 'Runtime/config changed'
    raw = store.read('RUNTIME_PROFILE.json')
    assert raw, 'Passing runtime profile required'
    profile = json.loads(raw)
    assert profile['status']=='PASS_PROFILE_AND_SAVED_SCORES'
    assert REF['signature'](profile['identity'])==profile['signature']
    assert profile['identity']['pilot_signature']==saved['signature']
    assert profile['identity']['hashes']=={k:hashes[k] for k in ('reference','optimized','profile')}
    for k,file_id in config['source_ids'].items():
        assert service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()==config['audit_sources'][k]
    workers = min(2, joblib.cpu_count())
    identity = {'hashes':hashes,'pilot_signature':saved['signature'],'profile_sha256':REF['sha'](raw),
                'versions':versions,'joblib':joblib.__version__,'workers':workers,
                'scope':'three fixed pairs, fold 5; threading then loky; 42 fits'}
    sig = REF['signature'](identity)
    prior = store.read('WORKER_CHECK.json')
    if prior:
        prior = json.loads(prior)
        if prior.get('signature')==sig and prior.get('status')=='PASS_WORKER_EQUIVALENCE':
            print('Completed check reused; no fits repeated.')
            print(json.dumps(prior['summary'],indent=2))
            return
    if workers < 2:
        print('Only one usable CPU detected. Parallel-worker comparison skipped; no fits run.')
        return
    print('Identity checks passed; loading and validating existing caches.',flush=True)
    tasks, expected, cases = PROFILE['load_tasks'](store,service,config,saved)
    print('Caches ready. Starting worker comparison.',flush=True)
    output = {'identity':identity,'signature':sig,'status':'RUNNING_WORKER_CHECK',
              'cases':cases,'phases':{},'full_screen_authorized':False}
    for backend in ('threading','loky'):
        phase = run_backend(sources,tasks,backend,workers)
        for i,result in enumerate(phase['cases']):
            assert result['scores']=={k:OPT['scientific'](v) for k,v in expected[i].items()}, 'Saved score mismatch'
            selected = REF['pick_setting']([v for k,v in expected[i].items() if k.startswith('inner|')])
            assert result['selected']==(None if selected is None else [selected['depth'],selected['min_samples_leaf']])
        output['phases'][backend] = phase
        # No uploads inside either timed phase; identical checkpoint policy for both.
        store.json('WORKER_CHECK.json',output)
        print(backend+' complete: '+str(round(phase['wall_seconds_including_dispatch'],2))+' seconds',flush=True)
    a,b = output['phases']['threading'],output['phases']['loky']
    assert [scientific_case(v) for v in a['cases']]==[scientific_case(v) for v in b['cases']], 'Predictions, scores or selections differ'
    output['summary'] = {'fits':42,'workers':workers,
                         'threads_seconds':a['wall_seconds_including_dispatch'],
                         'processes_seconds':b['wall_seconds_including_dispatch'],
                         'process_speedup_ratio':a['wall_seconds_including_dispatch']/b['wall_seconds_including_dispatch'],
                         'exact_prediction_digest_match':True,'saved_scores_and_selections_match':True,
                         'automatic_promotion':False}
    output['limitations'] = ['One fixed-order comparison on three flow cases; order/cache/noise can affect timing.',
                            'Includes dispatch, worker startup and serialization; checkpoint uploads excluded equally.',
                            'Bounded to two workers and one native thread per worker. No automatic worker-count expansion.',
                            'Prediction digests cover every evaluated row for all inner and outer fits; raw predictions stay in memory.',
                            'Not full-universe equivalence, memory-capacity certification or a full-run runtime estimate.',
                            'Interrupted comparison restarts at most 42 diagnostic fits; matching PASS receipt is reused.',
                            'No adoption or full screen without review.']
    output['status']='PASS_WORKER_EQUIVALENCE'
    store.json('WORKER_CHECK.json',output)
    print(output['status'])
    print(json.dumps(output['summary'],indent=2))
