"""Resume the unchanged feature-search worker with four concurrent processes."""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
import psutil
import joblib
import torch
import search_report_features as run


def replace_parent(pid):
    parent=psutil.Process(pid)
    command=parent.cmdline()
    assert len(command)>2 and command[2]=='run',command
    script=Path(command[1])
    if not script.is_absolute():script=Path(parent.cwd())/script
    assert script.resolve()==Path(run.__file__).resolve(),command
    # This is the specific process launched in this session, not another user's job.
    assert abs(parent.create_time()-1793854583.6253643)<.01
    children=parent.children(recursive=True)
    for child in children:
        argv=child.cmdline()
        assert len(argv)>2 and Path(argv[1]).name=='search_report_features.py' and argv[2]=='worker',argv
    stopped=[dict(pid=p.pid,command=p.cmdline()) for p in [parent]+children]
    parent.suspend()  # prevent its executor from starting another queued child
    for child in children:child.terminate()
    parent.terminate()
    _,alive=psutil.wait_procs(children+[parent],timeout=10)
    assert not alive,'Old processes did not exit; do not launch duplicates'
    # Preserve any checkpoint interrupted during serialization, then refit it.
    quarantined=[]
    for path in run.OUT.rglob('*'):
        if path.suffix not in ['.pt','.joblib'] or time.time()-path.stat().st_mtime>30:continue
        try:
            if path.suffix=='.pt':torch.load(path,weights_only=False,map_location='cpu')
            else:joblib.load(path)
        except Exception as exc:
            assert path.resolve().is_relative_to(run.OUT.resolve())
            dest=path.with_name(path.name+'.interrupted')
            assert not dest.exists()
            path.rename(dest)
            quarantined.append(dict(path=str(path),preserved_as=str(dest),error=repr(exc)))
    return stopped,quarantined


def main(pid):
    stopped=[];quarantined=[]
    if pid:stopped,quarantined=replace_parent(pid)
    run.prepare()
    run.dump(run.OUT/'ORCHESTRATION_AMENDMENT.json',dict(
        date='2026-11-05',reason='Measured38.6percent CPU usage and3.27GB available memory; increase from3 to4 workers.',
        workers=4,worker_source_sha256=run.q.sha(run.__file__),worker_source_unchanged=True,
        model_parameters_and_validation_rules_unchanged=True,stopped_own_processes=stopped,
        interrupted_checkpoints_preserved=quarantined,
        resume='Completed fits and trials reused; unfinished fits restarted; add/drop search continues using saved validation trials. No test scores inspected.'))
    jobs=[(method,i) for i in range(4) for method in ['rf','mlp']
          if not (run.OUT/method/f'{run.q.STARTS[i]:%Y%m%d}'/'DONE.json').exists()]
    print('REMAINING JOBS',jobs,flush=True)
    def launch(job):
        method,i=job
        with (run.OUT/f'{method}_{i}.log').open('a',encoding='utf-8') as log:
            log.write('\nRESUMED WITH FOUR-WORKER ORCHESTRATOR\n');log.flush()
            completed=subprocess.run([sys.executable,str(Path(run.__file__).resolve()),'worker','--method',method,'--ordinal',str(i)],
                                     cwd=run.HERE,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(f'Worker failed: {job}')
        return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(launch,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):print('FINISHED',future.result(),flush=True)
    run.summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replace-parent',type=int)
    args=parser.parse_args()
    with run.threadpool_limits(limits=1):main(args.replace_parent)
