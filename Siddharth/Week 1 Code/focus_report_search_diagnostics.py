"""Focus expensive diagnostics while preserving the frozen primary search.

All features receive validation block-permutation checks when a selected model
has more than eight inputs. Four largest and two smallest effects receive full
leave-one-out refits. Compact models retain exhaustive leave-one-out checks.
"""
from pathlib import Path
import concurrent.futures, json, subprocess, sys, time
import psutil
import joblib
import torch
import search_report_features as original

FOCUSED=Path(__file__).with_name('search_report_features_focused.py')
PRELUDE='''    ablations=[]
    ablation_features=list(selected['cols'])
    permutation_rows=[]
    if len(ablation_features)>8:
        base_mae=metrics(predict(chosen,x,vk),truth)['mae']
        block=(vk.asi8-vk.asi8.min())//pd.Timedelta(days=7).value
        blocks=[np.flatnonzero(block==b) for b in np.unique(block)]
        for c in ablation_features:
            effects=[]
            for repeat in range(3):
                order=np.random.default_rng(20261105+repeat).permutation(len(blocks))
                indices=np.concatenate([blocks[b] for b in order])
                changed=x.loc[vk,selected['cols']].copy()
                changed[c]=changed[c].to_numpy()[indices]
                effects.append(metrics(predict(chosen,changed,vk),truth)['mae']-base_mae)
            permutation_rows.append(dict(feature=c,validation_mae_increase=float(np.mean(effects)),repeats=effects))
        ranked=sorted(permutation_rows,key=lambda r:(-r['validation_mae_increase'],r['feature']))
        ablation_features=[r['feature'] for r in ranked[:4]+ranked[-2:]]
        dump(folder/'VALIDATION_PERMUTATION.json',permutation_rows)
    dump(folder/'DIAGNOSTIC_SCOPE.json',dict(selected_features=selected['cols'],fully_refitted_removals=ablation_features,
        policy='All removals when <=8 inputs; validation block permutation of all inputs and top4/bottom2 full removals otherwise',
        validation_only_diagnostic_choice=True,primary_selection_unchanged=True))
    if len(selected['cols'])>1:
        for removed in ablation_features:
'''


def prepare_worker():
    text=Path(original.__file__).read_text()
    old="    ablations=[]\n    if len(selected['cols'])>1:\n        for removed in selected['cols']:\n"
    assert text.count(old)==1
    text=text.replace(old,PRELUDE)
    text=text.replace("total=len(selected['cols']),removed=removed", "total=len(ablation_features),removed=removed")
    text=text.replace("test_rows=len(te),screening_subsets=len(trials),full_finalists=len(choices),ablations=len(ablations),",
                      "test_rows=len(te),screening_subsets=len(trials),full_finalists=len(choices),ablations=len(ablations),worker_source_sha256=q.sha(__file__),focused_diagnostics=True,")
    compile(text,str(FOCUSED),'exec')
    FOCUSED.write_text(text,encoding='utf-8')
    (original.OUT/'focused_worker_snapshot.py').write_text(text,encoding='utf-8')
    original.dump(original.OUT/'DIAGNOSTIC_AMENDMENT.json',dict(date='2026-11-05',
        reason='Third-window MLP selected32 inputs; full32 leave-one-out refits would dominate runtime without being necessary for the primary model comparison.',
        primary_feature_search_and_models_unchanged=True,original_worker_sha256=original.q.sha(original.__file__),
        focused_worker_sha256=original.q.sha(FOCUSED),orchestrator_sha256=original.q.sha(__file__),
        large_model_threshold=8,permutation='Three seeded permutations of contiguous7-day validation blocks, including missingness patterns, per selected feature.',
        full_removal_choice='Four greatest and two smallest validation permutation MAE increases; no outer-test scores used.',
        compact_models='All selected features removed individually, as before.',
        completed_diagnostics='Preserved, including the already-completed exhaustive12-feature MLP ablations.',
        reporting='Primary test scores retain full18481rows. Removal contributions are explicitly limited to tested windows; all large-model features get validation permutation scores.'))


def workers():
    result={}
    for p in psutil.process_iter():
        try:
            if p.name().lower()!='python.exe':continue
            args=p.cmdline()
            if len(args)<3 or args[2]!='worker':continue
            script=Path(args[1])
            if not script.is_absolute():script=Path(p.cwd())/script
            if script.resolve()!=Path(original.__file__).resolve():continue
            result[(args[args.index('--method')+1],int(args[args.index('--ordinal')+1]))]=p
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return result


def stop_one(p,folder):
    pid=p.pid;p.terminate();p.wait(timeout=10)
    quarantined=[]
    for path in folder.rglob('*'):
        if path.suffix not in ['.pt','.joblib'] or time.time()-path.stat().st_mtime>30:continue
        try:
            if path.suffix=='.pt':torch.load(path,weights_only=False,map_location='cpu')
            else:joblib.load(path)
        except Exception as exc:
            assert path.resolve().is_relative_to(original.OUT.resolve())
            dest=path.with_name(path.name+'.interrupted_focused')
            assert not dest.exists();path.rename(dest)
            quarantined.append(dict(path=str(path),preserved_as=str(dest),error=repr(exc)))
    return dict(stopped_worker_pid=pid,quarantined=quarantined)


def main():
    prepare_worker()
    jobs=[(method,i) for i in range(4) for method in ['rf','mlp']]
    launched={};handles=[];replacements=[];last=None
    while True:
        active=workers();done=[];progress=[]
        for job in jobs:
            method,i=job;folder=original.OUT/method/f'{original.q.STARTS[i]:%Y%m%d}'
            if (folder/'DONE.json').exists():done.append(job);continue
            if job in launched:
                code=launched[job].poll()
                if code is not None:raise RuntimeError(f'Focused worker exited{code} before DONE: {job}')
                continue
            sp=folder/'STATUS.json';selection_path=folder/'SELECTION.json'
            if not sp.exists():continue
            status=json.loads(sp.read_text());progress.append((job,status['stage'],status.get('completed')))
            if status['stage']!='ablation' or not selection_path.exists():continue
            selection=json.loads(selection_path.read_text())
            if len(selection['selected']['cols'])<=8:continue
            assert job in active,f'Expected active original worker: {job}'
            record=stop_one(active[job],folder);record['job']=job;replacements.append(record)
            log=(original.OUT/f'{method}_{i}.log').open('a',encoding='utf-8');handles.append(log)
            log.write('\nFOCUSED DIAGNOSTICS RESUME; PRIMARY SEARCH UNCHANGED\n');log.flush()
            launched[job]=subprocess.Popen([sys.executable,str(FOCUSED),'worker','--method',method,'--ordinal',str(i)],
                                           cwd=original.HERE,stdout=log,stderr=subprocess.STDOUT)
            original.dump(original.OUT/'FOCUSED_REPLACEMENTS.json',replacements)
            print('FOCUSED DIAGNOSTICS',job,len(selection['selected']['cols']),'inputs;6 full removals',flush=True)
        state=(len(done),str(progress))
        if state!=last:print('AUDITED',len(done),'/8',progress,flush=True);last=state
        if len(done)==8:break
        time.sleep(10)
    for proc in launched.values():assert proc.wait()==0
    for log in handles:log.close()
    original.summarize()


if __name__=='__main__':
    with original.threadpool_limits(limits=1):main()
