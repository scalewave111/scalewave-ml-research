"""Increase RF fit parallelism after replaying a cached fit exactly."""
from pathlib import Path
import json,subprocess,sys,time
import joblib,numpy as np,psutil
from sklearn.base import clone
import search_report_features as run
import focus_report_search_diagnostics as focus


def main():
    source=Path(run.__file__).with_name('search_report_features_focused.py')
    text=source.read_text();assert text.count('model.set_params(n_jobs=1)')==1
    text=text.replace('model.set_params(n_jobs=1)','model.set_params(n_jobs=3)')
    text=text.replace("dict(stage='ablation',completed=", "dict(stage='focused_ablation',completed=")
    target=source.with_name('search_report_features_parallel_rf.py')
    compile(text,str(target),'exec');target.write_text(text,encoding='utf-8')
    (run.OUT/'parallel_rf_worker_snapshot.py').write_text(text,encoding='utf-8')
    x,y,dev,ref,features=joblib.load(run.OUT/'prepared.joblib')
    chosen=None
    for path in sorted((run.OUT/'rf/20240101/screen').glob('*/bundle.joblib')):
        a=joblib.load(path)
        if 2<=len(a['cols'])<=3:chosen=(path,a);break
    assert chosen is not None
    path,a=chosen;old=a['models']['rf']['model'];prep=a['models']['rf']['prep']
    tx=run.q.b.design(x.loc[a['keys']],prep);truth=y.loc[a['keys'],'magnitude_log_bps'].to_numpy()
    new=clone(old).set_params(n_jobs=3);started=time.monotonic();new.fit(tx,truth)
    new.set_params(n_jobs=1);old.set_params(n_jobs=1)
    np.testing.assert_allclose(new.predict(tx),old.predict(tx),atol=1e-12,rtol=0)
    for left,right in zip(new.estimators_,old.estimators_):
        np.testing.assert_array_equal(left.tree_.feature,right.tree_.feature)
        np.testing.assert_array_equal(left.tree_.threshold,right.tree_.threshold)
        np.testing.assert_array_equal(left.tree_.value,right.tree_.value)
    benchmark=dict(status='PASS',source_bundle=str(path),source_hash=run.q.sha(path),
        trees=len(new.estimators_),training_rows=len(tx),features=len(a['cols']),
        exact_tree_structure_thresholds_and_values=True,max_prediction_difference=float(np.max(abs(new.predict(tx)-old.predict(tx)))),
        refit_seconds=time.monotonic()-started)
    run.dump(run.OUT/'RF_PARALLELISM_AMENDMENT.json',dict(date='2026-11-05',fit_workers=3,
        statistical_model_and_seed_unchanged=True,focused_diagnostics_unchanged=True,
        focused_worker_sha256=run.q.sha(source),parallel_worker_sha256=run.q.sha(target),
        orchestrator_sha256=run.q.sha(__file__),replay_benchmark=benchmark))
    print('PARALLEL TREE REPLAY PASS',benchmark,flush=True)
    procs=[];logs=[];replacements=[]
    for ordinal in [2,3]:
        folder=run.OUT/'rf'/f'{run.q.STARTS[ordinal]:%Y%m%d}'
        # Do not interrupt a primary fit to optimize auxiliary diagnostics.
        while not (folder/'DONE.json').exists():
            status=json.loads((folder/'STATUS.json').read_text())
            if status['stage'] not in ['screen','full_finalist','outer_refit']:break
            if ('rf',ordinal) not in focus.workers():break
            time.sleep(10)
        if (folder/'DONE.json').exists():continue
        active=focus.workers();job=('rf',ordinal)
        if job not in active:
            print('LEAVING ALREADY-FOCUSED OR FINISHED WORKER',job,flush=True);continue
        status=json.loads((folder/'STATUS.json').read_text())
        sp=folder/'SELECTION.json'
        if status['stage']=='ablation' and sp.exists() and len(json.loads(sp.read_text())['selected']['cols'])>8:
            print('LEAVING WORKER TO FOCUSED CONTROLLER',job,flush=True);continue
        try:record=focus.stop_one(active[job],folder)
        except psutil.NoSuchProcess:
            print('WORKER FINISHED OR MOVED TO FOCUSED CONTROLLER',job,flush=True);continue
        record['job']=job;replacements.append(record)
        log=(run.OUT/f'rf_{ordinal}.log').open('a',encoding='utf-8');logs.append(log)
        log.write('\nRF TREE-PARALLEL RESUME AFTER EXACT REPLAY CHECK\n');log.flush()
        procs.append(subprocess.Popen([sys.executable,str(target),'worker','--method','rf','--ordinal',str(ordinal)],
                                      cwd=run.HERE,stdout=log,stderr=subprocess.STDOUT))
        print('RF RESUMED WITH THREE TREE WORKERS',ordinal,flush=True)
    run.dump(run.OUT/'RF_PARALLEL_REPLACEMENTS.json',replacements)
    for proc in procs:assert proc.wait()==0
    for log in logs:log.close()
    print('RF PARALLEL CONTINUATION COMPLETE',flush=True)


if __name__=='__main__':
    with run.threadpool_limits(limits=1):main()
