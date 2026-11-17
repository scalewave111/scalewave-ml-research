"""Bounded execution and cache-only equivalence benchmark. No full-screen entry point.

REF is the frozen screen_runner namespace, supplied by the notebook/test.
"""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np


def scientific(score):
    return {k: v for k, v in score.items() if k != 'seconds'}


class OrderedWriter:
    """One writer owns its store; serialize before queuing, bound pending snapshots."""
    def __init__(self, write, capacity=2):
        self.write = write
        self.capacity = capacity
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()
        self.lock = threading.Lock()

    def submit(self, name, value):
        raw = json.dumps(value, allow_nan=False).encode()
        with self.lock:
            while len(self.pending) >= self.capacity:
                self.pending.popleft().result()
            self.pending.append(self.pool.submit(self.write, name, raw))

    def close(self):
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)


def prepare(x_train, y_train, x_valid, y_valid, minimum):
    return (REF['preprocess'](x_train, x_valid, minimum), y_train, y_valid,
            np.isfinite(x_valid),
            {'train_observed': int(np.isfinite(x_train).sum()), 'train_rows': len(x_train),
             'validation_observed': int(np.isfinite(x_valid).sum()), 'validation_rows': len(x_valid)})


def prepared_fit(data, depth, leaf):
    from sklearn.tree import DecisionTreeRegressor
    start = time.perf_counter()
    prepared, y_train, y_valid, observed, coverage = data
    if prepared is None:
        return {'status': 'INSUFFICIENT_TRAINING_COVERAGE_OR_VARIATION',
                'fit_count': 0, 'coverage': coverage}, None
    train, valid, median = prepared
    model = DecisionTreeRegressor(criterion='absolute_error', max_depth=depth,
                                 min_samples_leaf=leaf, random_state=48)
    model.fit(train, y_train)
    prediction = model.predict(valid)
    assert np.array_equal(prediction, REF['manual_predict'](model, valid))
    error = np.abs(prediction-y_valid)
    score = {'status': 'PASS', 'fit_count': 1, 'depth': depth, 'min_samples_leaf': leaf,
             'MAE_minutes': float(error.mean()), 'absolute_error_sum': float(error.sum()),
             'observed_rows_MAE': float(error[observed].mean()) if observed.any() else None,
             'missing_rows_MAE': float(error[~observed].mean()) if (~observed).any() else None,
             'coverage': coverage, 'imputation_median': median,
             'tree_nodes': int(model.tree_.node_count), 'seconds': time.perf_counter()-start,
             'independent_tree_traversal': 'PASS'}
    return score, prediction


def fold_task(x, y, masks, config, optimized, emit=lambda key, score: None):
    """One fold per task; pool has at most two tasks, including outer fits.

    Predictions are kept only in memory for equivalence checks, never checkpointed.
    """
    it, iv, ot, ov = [masks[k] for k in ('inner_train', 'inner_valid', 'outer_train', 'outer_valid')]
    minimum = config['minimum_observed_training']
    fits, predictions = {}, {}
    data = prepare(x[it], y[it], x[iv], y[iv], minimum) if optimized else None
    def fit(train, valid, depth, leaf, prepared):
        if optimized:
            return prepared_fit(prepared, depth, leaf)
        # Capture the frozen implementation's independent prediction traversal.
        local = {}
        original = REF['manual_predict']
        def capture(model, values):
            prediction = original(model, values)
            local['prediction'] = prediction
            return prediction
        import types
        namespace = dict(REF['tree_fit'].__globals__, manual_predict=capture)
        reference = types.FunctionType(REF['tree_fit'].__code__, namespace)
        score = reference(x[train], y[train], x[valid], y[valid], depth, leaf, minimum)
        return score, local.get('prediction')
    for depth, leaf in config['grid']:
        key = f'inner|{depth}|{leaf}'
        fits[key], predictions[key] = fit(it, iv, depth, leaf, data)
        emit(key, fits[key])
    selected = REF['pick_setting'](list(fits.values()))
    data = prepare(x[ot], y[ot], x[ov], y[ov], minimum) if optimized and selected else None
    outer, prediction = fit(ot, ov, selected['depth'], selected['min_samples_leaf'], data) if selected else ({'status': 'INNER_FEATURE_INELIGIBLE', 'fit_count': 0}, None)
    if outer['status'] != 'PASS':
        prediction = np.full(int(ov.sum()), float(np.median(y[ot])))
        error = np.abs(y[ov]-prediction)
        outer.update(fallback='training_median_time_to_high', MAE_minutes=float(error.mean()), absolute_error_sum=float(error.sum()))
    fits['outer'], predictions['outer'] = outer, prediction
    emit('outer', outer)
    return {'fits': fits, 'predictions': predictions,
            'selected': None if selected is None else [selected['depth'], selected['min_samples_leaf']]}


def assert_equal(a, b):
    assert a['selected'] == b['selected']
    assert a['fits'].keys() == b['fits'].keys()
    for key in a['fits']:
        assert scientific(a['fits'][key]) == scientific(b['fits'][key]), key
        left, right = a['predictions'][key], b['predictions'][key]
        assert (left is None and right is None) or np.array_equal(left, right), key


def run_benchmark(config, reference_hash, optimized_hash, expected_pilot_identity):
    import os
    import platform
    import pandas as pd
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
    assert saved['identity'] == expected_pilot_identity, 'Pilot identity changed'
    assert saved['identity']['config'] == config and saved['identity']['sources'] == config['audit_sources']
    assert saved['identity']['versions'] == versions, 'Runtime changed; stop for review. Do not replace pilot.'
    assert REF['signature'](saved['identity']) == saved['signature']
    prior = store.read('THROUGHPUT_CHECK.json')
    if prior:
        receipt = json.loads(prior)
        if (receipt.get('status') == 'PASS_EXACT_PREDICTIONS_SCORES_SELECTION_AND_SAVED_PILOT'
                and receipt.get('reference_sha256') == reference_hash
                and receipt.get('optimized_sha256') == optimized_hash
                and receipt.get('screen_signature') == saved['signature']):
            print('This verification already passed. Reusing receipt; no fits repeated.')
            return
    for key, file_id in config['source_ids'].items():
        current = service.files().get(fileId=file_id, fields='id,name,size,md5Checksum,modifiedTime').execute()
        assert current == config['audit_sources'][key]
    baseline = json.loads(store.read('BASELINE_RESULTS.json'))
    assert REF['signature'](baseline['identity']) == baseline['signature'] == config['baseline_signature']
    columns = [f for f in config['candidate_features'] if f.split('__')[1] == 'flow']
    identity = dict(type='f4z_research_cache_v1', source=config['audit_sources']['X'], columns=columns,
                    origin_end=config['origin_end_exclusive'], cadence=30, minimum_baseline_n=1000,
                    allowed_source_status=['PROVISIONAL'])
    meta, xt, matrix = REF['read_cache'](store, 'SCREEN_X_F4Z_flow.npz', identity)
    assert meta['completed'] == config['months'] and xt is not None
    xt = pd.DatetimeIndex(xt, tz='UTC')
    assert xt.is_unique and xt.is_monotonic_increasing and (xt < pd.Timestamp(config['origin_end_exclusive'])).all()
    tasks, existing = [], []
    # Fixed before seeing scores: each pilot feature paired with one pilot horizon,
    # latest outer fold (largest training population); 21 fits per implementation.
    for feature, horizon in zip(config['pilot_features'], config['pilot_horizons']):
        identity = dict(type='valid_time_high_research_v1', source=config['audit_sources']['Y'],
                        horizon=horizon, origin_end=config['origin_end_exclusive'], cadence=30)
        ym, times, y = REF['read_cache'](store, f'SCREEN_Y_{horizon}m.npz', identity)
        assert ym['completed'] == config['months'] and times is not None
        times = pd.DatetimeIndex(times, tz='UTC')
        assert times.is_unique and times.is_monotonic_increasing
        assert (times < pd.Timestamp(config['origin_end_exclusive'])).all()
        assert ((times+pd.Timedelta(minutes=horizon)) < pd.Timestamp(config['reserve_start'])).all()
        REF['check_baselines'](times, y, horizon, baseline, config)
        indices = xt.get_indexer(times)
        assert (indices >= 0).all()
        masks = REF['partition_masks'](times, horizon, *config['folds'][-1])
        tasks.append((matrix[indices, columns.index(feature)], y, masks, config))
        job_id = feature+'|'+str(horizon)
        info = saved['jobs'][job_id]
        job = json.loads(service.files().get_media(fileId=info['score_file_id']).execute())
        assert REF['signature'](job) == info['score_sha256']
        assert job['screen_signature'] == saved['signature']
        assert job['X_cache_sha256'] == meta['array_sha256'] and job['Y_cache_sha256'] == ym['array_sha256']
        existing.append((job_id, job))
    output = {'status': 'RUNNING_EQUIVALENCE_ONLY', 'reference_sha256': reference_hash,
              'optimized_sha256': optimized_hash, 'screen_signature': saved['signature'],
              'pilot_declared_runner_sha256': saved['identity']['runner_sha256'],
              'reference_hash_matches_pilot_declaration': reference_hash == saved['identity']['runner_sha256'],
              'versions': versions, 'scope': 'three fixed pilot pairs, outer fold 5, 42 total fits',
              'full_run_authorized': False, 'scores': {}, 'timings': {}}
    output['fit_workers'] = min(2, os.cpu_count() or 1)
    # Dedicated service is owned exclusively by the single writer thread.
    writer_store = REF['DriveStore'](build('drive', 'v3', credentials=credentials, cache_discovery=False), config['folder'])
    writer = OrderedWriter(writer_store.write)
    lock = threading.Lock()
    def emit(task_id, key, score):
        with lock:
            output['scores'][f'{task_id}|{key}'] = score
            writer.submit('THROUGHPUT_CHECK.json', output)
    results = {}
    try:
        with threadpool_limits(limits=1):
            for optimized in (False, True):
                label = 'optimized' if optimized else 'reference'
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=output['fit_workers']) as pool:
                    futures = [pool.submit(fold_task, *task, optimized,
                               lambda key, score, i=i: emit(f'{label}|{i}', key, score))
                               for i, task in enumerate(tasks)]
                    results[label] = [f.result() for f in futures]
                output['timings'][label+'_compute_and_enqueue_seconds'] = time.perf_counter()-start
                # Barrier: include all remote checkpoint completion in each timing.
                writer.close()
                output['timings'][label+'_including_upload_seconds'] = time.perf_counter()-start
                writer = OrderedWriter(writer_store.write)
                print(label+' phase complete', flush=True)
        for i, (a, b) in enumerate(zip(results['reference'], results['optimized'])):
            assert_equal(a, b)
            job_id, job = existing[i]
            for key, score in a['fits'].items():
                assert scientific(score) == scientific(job['fits'][f'{job_id}|4|{key}'])
        output['status'] = 'PASS_EXACT_PREDICTIONS_SCORES_SELECTION_AND_SAVED_PILOT'
        output['limitations'] = ['Three fixed cases only; not full-universe equivalence or a runtime guarantee.',
                                'Both phases use bounded fold scheduling and asynchronous writes; timing isolates preparation reuse, not total speedup over the old scheduler.',
                                'No automatic checkpoint migration or full launch. Review this receipt first.',
                                'Current reference content hash differs from historical pilot declaration; saved-score replay is required, not identity equivalence.',
                                'An interrupted verification may repeat at most 42 fits; a matching PASS receipt is reused.']
        writer.submit('THROUGHPUT_CHECK.json', output)
    except BaseException:
        output['status'] = 'FAILED_OR_INTERRUPTED_NOT_APPROVED'
        # Drain earlier snapshots before attempting a final failure receipt.
        # A failed upload must not be hidden by a secondary shutdown error.
        try:
            writer.close()
        except Exception:
            pass
        writer = OrderedWriter(writer_store.write)
        try:
            writer.submit('THROUGHPUT_CHECK.json', output)
            writer.close()
        except Exception:
            print('Failure receipt could not be uploaded; this run is NOT verified.')
        raise
    finally:
        writer.close()
    print(output['status'])
    print(json.dumps(output['timings'], indent=2))
    print('Saved THROUGHPUT_CHECK.json. Full screen remains disabled.')
