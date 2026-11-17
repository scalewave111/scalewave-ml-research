"""Cache-only timing diagnostic. REF and OPT are embedded notebook namespaces."""
import json
import os
import platform
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd


def measured_fold(task, emit):
    # Per-task timers: no interpreter-wide profiler/tool IDs or global monkeypatches.
    timing = dict.fromkeys(['preparation_seconds', 'model_fit_seconds', 'prediction_seconds',
                           'independent_traversal_seconds', 'fit_and_scoring_seconds',
                           'checkpoint_emit_seconds'], 0.0)
    def timed(name, function, *args):
        start = time.perf_counter()
        try:
            return function(*args)
        finally:
            timing[name] += time.perf_counter()-start
    def prepared_fit(data, depth, leaf):
        from sklearn.tree import DecisionTreeRegressor
        start = time.perf_counter()
        prepared, y_train, y_valid, observed, coverage = data
        if prepared is None:
            return {'status':'INSUFFICIENT_TRAINING_COVERAGE_OR_VARIATION',
                    'fit_count':0, 'coverage':coverage}, None
        train, valid, median = prepared
        model = DecisionTreeRegressor(criterion='absolute_error', max_depth=depth,
                                      min_samples_leaf=leaf, random_state=48)
        timed('model_fit_seconds', model.fit, train, y_train)
        prediction = timed('prediction_seconds', model.predict, valid)
        independent = timed('independent_traversal_seconds', REF['manual_predict'], model, valid)
        assert np.array_equal(prediction, independent)
        error = np.abs(prediction-y_valid)
        return {'status':'PASS', 'fit_count':1, 'depth':depth, 'min_samples_leaf':leaf,
                'MAE_minutes':float(error.mean()), 'absolute_error_sum':float(error.sum()),
                'observed_rows_MAE':float(error[observed].mean()) if observed.any() else None,
                'missing_rows_MAE':float(error[~observed].mean()) if (~observed).any() else None,
                'coverage':coverage, 'imputation_median':median,
                'tree_nodes':int(model.tree_.node_count), 'seconds':time.perf_counter()-start,
                'independent_tree_traversal':'PASS'}, prediction
    namespace = dict(OPT['fold_task'].__globals__,
                     prepare=lambda *args: timed('preparation_seconds', OPT['prepare'], *args),
                     prepared_fit=lambda *args: timed('fit_and_scoring_seconds', prepared_fit, *args))
    fold = types.FunctionType(OPT['fold_task'].__code__, namespace)
    start = time.perf_counter()
    result = fold(*task, True, lambda *args: timed('checkpoint_emit_seconds', emit, *args))
    timing['fold_wall_seconds'] = time.perf_counter()-start
    return result, timing


def load_tasks(store, service, config, saved):
    baseline = json.loads(store.read('BASELINE_RESULTS.json'))
    assert REF['signature'](baseline['identity']) == baseline['signature'] == config['baseline_signature']
    assert baseline['identity']['source'] == config['audit_sources']['Y']
    columns = [f for f in config['candidate_features'] if f.split('__')[1] == 'flow']
    identity = dict(type='f4z_research_cache_v1', source=config['audit_sources']['X'], columns=columns,
                    origin_end=config['origin_end_exclusive'], cadence=30,
                    minimum_baseline_n=1000, allowed_source_status=['PROVISIONAL'])
    meta, xt, matrix = REF['read_cache'](store, 'SCREEN_X_F4Z_flow.npz', identity)
    assert meta['completed'] == config['months'] and xt is not None, 'Complete existing flow cache required'
    xt = pd.DatetimeIndex(xt, tz='UTC')
    assert xt.is_unique and xt.is_monotonic_increasing
    assert (xt < pd.Timestamp(config['origin_end_exclusive'])).all()
    tasks, expected, cases = [], [], []
    for feature, horizon in zip(config['pilot_features'], config['pilot_horizons']):
        identity = dict(type='valid_time_high_research_v1', source=config['audit_sources']['Y'],
                        horizon=horizon, origin_end=config['origin_end_exclusive'], cadence=30)
        ym, times, y = REF['read_cache'](store, f'SCREEN_Y_{horizon}m.npz', identity)
        assert ym['completed'] == config['months'] and times is not None, 'Complete existing Y cache required'
        times = pd.DatetimeIndex(times, tz='UTC')
        assert times.is_unique and times.is_monotonic_increasing
        assert (times < pd.Timestamp(config['origin_end_exclusive'])).all()
        assert ((times+pd.Timedelta(minutes=horizon)) < pd.Timestamp(config['reserve_start'])).all()
        assert np.isfinite(y).all() and ((y >= 0) & (y <= horizon)).all()
        REF['check_baselines'](times, y, horizon, baseline, config)
        indices = xt.get_indexer(times)
        assert (indices >= 0).all()
        tasks.append((matrix[indices, columns.index(feature)], y,
                      REF['partition_masks'](times, horizon, *config['folds'][-1]), config))
        job_id = feature+'|'+str(horizon)
        info = saved['jobs'][job_id]
        job = json.loads(service.files().get_media(fileId=info['score_file_id']).execute())
        assert REF['signature'](job) == info['score_sha256'] and job['screen_signature'] == saved['signature']
        assert job['X_cache_sha256'] == meta['array_sha256'] and job['Y_cache_sha256'] == ym['array_sha256']
        expected.append({key: job['fits'][f'{job_id}|4|{key}']
                         for key in [f'inner|{d}|{l}' for d, l in config['grid']]+['outer']})
        cases.append({'feature':feature, 'horizon_minutes':horizon, 'outer_fold':5})
    return tasks, expected, cases


def run_profile(config, expected_identity, hashes):
    import sklearn
    import pyarrow
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from threadpoolctl import threadpool_limits
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    store = REF['DriveStore'](service, config['folder'])
    saved = json.loads(store.read('SCREEN_RESULTS.json'))
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, pyarrow=pyarrow.__version__)
    assert saved['identity'] == expected_identity and REF['signature'](expected_identity) == saved['signature']
    assert expected_identity['config'] == config and expected_identity['versions'] == versions, 'Runtime/config changed; stop for review'
    prior_raw = store.read('THROUGHPUT_CHECK.json')
    assert prior_raw, 'Completed equivalence receipt required'
    prior = json.loads(prior_raw)
    assert prior['status'] == 'PASS_EXACT_PREDICTIONS_SCORES_SELECTION_AND_SAVED_PILOT'
    assert prior['screen_signature'] == saved['signature']
    assert prior['reference_sha256'] == hashes['reference'] and prior['optimized_sha256'] == hashes['optimized']
    for key, file_id in config['source_ids'].items():
        assert service.files().get(fileId=file_id, fields='id,name,size,md5Checksum,modifiedTime').execute() == config['audit_sources'][key]
    identity = {'hashes':hashes, 'pilot_signature':saved['signature'],
                'equivalence_sha256':REF['sha'](prior_raw), 'versions':versions, 'scope':'three fixed pairs, outer fold 5'}
    signature = REF['signature'](identity)
    existing = store.read('RUNTIME_PROFILE.json')
    if existing:
        previous = json.loads(existing)
        if previous.get('signature') == signature and previous.get('status') == 'PASS_PROFILE_AND_SAVED_SCORES':
            print('Reusing completed profile; no fits repeated.')
            print(json.dumps(previous['summary'], indent=2))
            return
    start = time.perf_counter()
    tasks, expected, cases = load_tasks(store, service, config, saved)
    load_seconds = time.perf_counter()-start
    output = {'identity':identity, 'signature':signature, 'status':'RUNNING_PROFILE_ONLY',
              'cases':cases, 'fits':{}, 'timings':{}, 'cache_load_and_validation_seconds':load_seconds,
              'full_screen_authorized':False}
    # Writer owns its own HTTP transport; never share it with fit threads.
    writer_store = REF['DriveStore'](build('drive', 'v3', credentials=credentials, cache_discovery=False), config['folder'])
    uploads, submission = [], []
    def upload(name, raw):
        start = time.perf_counter()
        writer_store.write(name, raw)
        uploads.append(time.perf_counter()-start)
    writer = OPT['OrderedWriter'](upload, capacity=2)
    lock = threading.Lock()
    def emit(case, key, score):
        start = time.perf_counter()
        with lock:
            assert OPT['scientific'](score) == OPT['scientific'](expected[case][key]), 'Saved pilot score mismatch'
            output['fits'][f'{case}|{key}'] = score
            writer.submit('RUNTIME_PROFILE.json', output)
        submission.append(time.perf_counter()-start)
    workers = min(2, os.cpu_count() or 1)
    cpu_start, start = time.process_time(), time.perf_counter()
    try:
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(measured_fold, task, lambda key, score, i=i: emit(i, key, score))
                       for i, task in enumerate(tasks)]
            for i, future in enumerate(futures):
                result, timing = future.result()
                # Avoid mutating checkpoint snapshots while another thread serializes.
                with lock:
                    output['timings'][str(i)] = timing
                del result  # No row predictions or models retained on disk.
                print(f'Profile case {i+1}/3 complete', flush=True)
        before_drain = time.perf_counter()-start
        writer.close()
        wall, cpu = time.perf_counter()-start, time.process_time()-cpu_start
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        # Partial output remains RUNNING, never a verified result; rerun is bounded.
        raise
    summed = {key:sum(t[key] for t in output['timings'].values()) for key in next(iter(output['timings'].values()))}
    output['summary'] = {'fits':len(output['fits']), 'fit_workers':workers,
                         'wall_including_checkpoints_seconds':wall,
                         'final_checkpoint_drain_seconds':wall-before_drain,
                         'process_cpu_seconds':cpu, 'average_cpu_cores_used':cpu/wall,
                         'summed_task_timings_seconds':summed,
                         'summed_checkpoint_submit_seconds':sum(submission),
                         'summed_upload_seconds':sum(uploads), 'checkpoint_upload_count':len(uploads)}
    output['limitations'] = ['Task durations overlap across workers; do not add them to wall time.',
                            'Fit-and-scoring includes model fitting/prediction/traversal; timing categories overlap.',
                            'Emit includes locking, score verification, serialization and queue backpressure.',
                            'Upload time overlaps computation; final drain alone is not total checkpoint overhead.',
                            'Profiling adds overhead; three flow cases are not a full-universe runtime estimate.',
                            'Predictions independently traversed; saved scores compared; no new reference retraining.',
                            'Interrupted profile may restart at most 21 fits. Completed matching receipt is reused.',
                            'Full screen remains disabled; no automatic pilot identity migration.']
    assert len(output['fits']) == 21 and len(uploads) == 21
    output['status'] = 'PASS_PROFILE_AND_SAVED_SCORES'
    # No other HTTP activity on writer_store remains; final receipt write is untimed.
    writer_store.json('RUNTIME_PROFILE.json', output)
    print(output['status'])
    print(json.dumps(output['summary'], indent=2))
    print('Saved RUNTIME_PROFILE.json. Send this summary; no full screen started.')
