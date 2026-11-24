"""Local, resumable coordinator for the final two PDF feature batches. No API calls."""
from pathlib import Path
import copy, hashlib, json, os, subprocess, sys, time

HERE=Path(__file__).resolve().parent; SUB=HERE.parent; ROOT=SUB.parent
OUT=HERE/'run_001'; LEADER=HERE/'context_leader'; STATE=HERE/'offline_state.json'
BATCHES=[[
 'liquidation_liq_activity_severity_active_severity_rz__4h',
 'flow_flow_activity_rz__1h', 'price_price_excursion_range_rz__5m',
 'liquidation_venue_breadth_percentile__15m'],[
 'volatility_realized_variance_level_rz__12h','price_price_excursion_range_rz__12h',
 'basis_basis_range_rz__12h','liquidation_liq_temporal_concentration_percentile__12h']]

def read(p): return json.loads(p.read_text())
def write(p,value):
 if p.exists(): p.read_text()
 temp=p.with_suffix(p.suffix+'.tmp')
 if temp.exists(): temp.read_text()
 temp.write_text(json.dumps(value,indent=2));temp.replace(p)
def save(state): write(STATE,state)
def safe_files(directory):
 assert directory.resolve() in [OUT.resolve(),LEADER.resolve()]
 assert directory.resolve().is_relative_to(ROOT.resolve()) and not directory.is_symlink()
 files=list(directory.iterdir())
 assert all(p.is_file() and not p.is_symlink() and p.resolve().parent==directory.resolve() for p in files)
 return files
def clean(directory):
 for p in safe_files(directory):p.unlink()
def move(source,dest):
 # Repeating an interrupted move is safe; existing destination files are checked.
 for p in safe_files(source):
  target=dest/p.name
  if target.exists():
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest()
   p.unlink()
  else:p.rename(target)
def run(script):
 proc=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT)
 try:
  while True:
   try:
    code=proc.wait(timeout=60);break
   except subprocess.TimeoutExpired:print('Still running; keep this window open.',flush=True)
  if code:raise RuntimeError(f'{script.name} stopped with exit code {code}. Read the error above.')
 except KeyboardInterrupt:
  if proc.poll() is None:proc.terminate()
  proc.wait();raise
def archive_previous():
 old=read(HERE/'context_results.json');p=SUB/'experiment_history.json';history=read(p)
 for e in old['results']:
  matches=[x for x in history if x.get('experiment')==e['experiment']]
  assert len(matches)==1,'Previous batch must be finalized first.'
  for key in ['fold_MAE','inner_scores']:
   matches[0].setdefault(key,e[key])
 write(p,history)
def preflight():
 from feature_experiment import load,splits
 reg=read(SUB/'magnitude_benchmark_registry.json');c=read(SUB/'magnitude_current/run_001/config.json')
 inv={f['pdf_name']:f for f in read(HERE/'pdf_feature_inventory.json')['features']}
 assert c['seeds']==[48,17,91] and c['forest']['n_estimators']==128
 assert c['forest']['n_jobs']==8
 for name in sum(BATCHES,[]):assert name in inv
 probe=copy.deepcopy(c)
 for name in sum(BATCHES,[]):probe['additions'][name]=inv[name]['column']
 X,y=load(probe,True)
 for _,ot,_,it,_ in splits(probe,y):
  for idx in [it,ot]:
   for name in sum(BATCHES,[]):
    v=X.loc[idx,inv[name]['column']];assert v.count()>=1000 and v.nunique()>=2,name
 run(SUB/'magnitude_current/replay.py')
 print(f"SETUP CHECK PASSED. Current MAE: {reg['current']['MAE']:.6f}",flush=True)
 return inv
def prepare(state,inv):
 assert not safe_files(OUT) and not LEADER.exists(),'Unfinished unrelated output exists; ask for a review.'
 archive_previous()
 reg=read(SUB/'magnitude_benchmark_registry.json');c=read(SUB/'magnitude_current/run_001/config.json')
 names=BATCHES[state['batch']]
 assert all(inv[n]['status']=='PENDING' for n in names)
 c.update(benchmark_mae=reg['current']['MAE'],prior_fit_count=reg['cumulative_counted_steps'],
  authorization='User requested the final two batches locally; four individual additions per batch.',
  change_scope='One feature added to the fixed starting model per batch.',
  stopping_rule='Only the final eight eligible PDF features; combinations and other horizons deferred.')
 plan={'base_config':c,'candidates':{n:inv[n]['column'] for n in names},'inventory':'pdf_feature_inventory.json',
  'run_id':f"offline_remaining_batch_{state['batch']+1}"}
 state.update(phase='start',plan=plan,results=[],candidate=0,best=c['benchmark_mae'],steps=c['prior_fit_count'])
 save(state)
def main():
 inv=preflight()
 if '--check' in sys.argv:
  print('No training started. Run this script without --check when ready.');return
 state=read(STATE) if STATE.exists() else {'batch':0,'phase':'prepare','completed_batches':[]}
 if state['phase']=='finalizing':
  raise RuntimeError('Interrupted during final saving. Stop here and ask Codex to review offline_state.json. Do not delete files or restart training.')
 while state['batch']<len(BATCHES):
  if state['phase']=='prepare':prepare(state,inv)
  if state['phase']=='start':
   write(HERE/'context_plan.json',state['plan']);LEADER.mkdir(exist_ok=True)
   state['phase']='train';save(state)
  plan=state['plan'];items=list(plan['candidates'].items())
  while state['candidate']<len(items):
   name,feature=items[state['candidate']]
   print(f"\nBATCH {state['batch']+1}/2, FEATURE {state['candidate']+1}/4: {name}",flush=True)
   if state['phase']=='train':
    c=copy.deepcopy(plan['base_config']);c['version']='ensemble_context_'+name
    c['additions'][name]=feature;c['prior_fit_count']=state['steps']
    write(HERE/'context_config.json',c)
    run(HERE/'context_feature.py')
    r=read(OUT/'RESULT.json');v=read(OUT/'verification.json');assert v['status']=='PASS'
    replay=SUB/'magnitude_current/replay.py'
    source=replay.read_text().replace("OUT=HERE/'run_001'","OUT=HERE.parent/'magnitude_active'/'run_001'")
    exec(compile(source,str(replay),'exec'),{'__file__':str(replay)})
    state['results'].append({'experiment':c['version'],'config':c,'result':r,'verification':v,
     'inner_scores':read(OUT/'inner_scores.json'),'fold_MAE':(OUT/'fold_MAE.csv').read_text()})
    state['steps']=r['cumulative_counted_steps'];state['promote_candidate']=r['metrics']['prediction']['MAE']<state['best']-1e-12
    state['phase']='clear';save(state)
   if state['phase']=='clear':
    write(HERE/'context_results.json',{'plan':plan,'results':state['results'],'cumulative_counted_steps':state['steps']})
    clean(LEADER if state['promote_candidate'] else OUT)
    state['phase']='move';save(state)
   if state['phase']=='move':
    if state['promote_candidate']:
     move(OUT,LEADER);state['best']=state['results'][-1]['result']['metrics']['prediction']['MAE']
    state['candidate']+=1;state['phase']='train';save(state)
  if state['phase']=='train':state['phase']='gather';save(state)
  if state['phase']=='gather':
   if LEADER.exists():
    move(LEADER,OUT);LEADER.rmdir()
   state['phase']='finalizing';save(state)
   print('\nVERIFYING AND SAVING BATCH. Please let this short step finish.',flush=True)
   run(HERE/'finalize_context_batch.py')
   state['completed_batches'].append({'batch':state['batch']+1,'results':state['results'],
    'current':read(SUB/'magnitude_benchmark_registry.json')['current']})
   state['batch']+=1;state['phase']='prepare' if state['batch']<2 else 'complete';save(state)
   inv={f['pdf_name']:f for f in read(HERE/'pdf_feature_inventory.json')['features']}
 reg=read(SUB/'magnitude_benchmark_registry.json')
 lines=['BOTH REMAINING BATCHES COMPLETE','',f"Current MAE: {reg['current']['MAE']:.9f}",f"Spearman IC: {reg['current']['Spearman_IC']:.9f}",'']
 for batch in state['completed_batches']:
  lines.append(f"Batch {batch['batch']}")
  for e in batch['results']:
   m=e['result']['metrics']['prediction'];lines.append(f"{e['experiment']}: MAE {m['MAE']:.9f}; Spearman {m['Spearman_IC']:.9f}")
 lines+=['','Replay verified. Unused generated models pruned. Original data untouched.',
  'PDF report was not regenerated automatically; ask Codex to update it from these results.']
 p=HERE/'OFFLINE_RESULTS.txt'
 if p.exists():p.read_text()
 p.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':
 # Windows releases this lock automatically when the process exits.
 import msvcrt
 with (HERE/'offline_run.lock').open('a+b') as lock:
  lock.seek(0,os.SEEK_END)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0)
  try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  except OSError:raise SystemExit('Another local runner is already open. Use that window.')
  try:main()
  except KeyboardInterrupt:print('\nStopped. Run the same command to resume saved training checkpoints.',flush=True)
  except Exception as exc:print(f'\nSTOPPED: {exc}\nKeep all files. Share this message if you need help.',flush=True);sys.exit(1)
