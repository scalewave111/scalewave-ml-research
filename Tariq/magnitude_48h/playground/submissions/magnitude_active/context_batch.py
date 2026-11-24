"""Run independent additions against one frozen incumbent; retain one leader."""
from pathlib import Path
import json,copy,subprocess,sys,hashlib
HERE=Path(__file__).resolve().parent;SUB=HERE.parent;ROOT=SUB.parent
OUT=HERE/'run_001';LEADER=HERE/'context_leader'
plan=json.loads((HERE/'context_plan.json').read_text());results=[];best=plan['base_config']['benchmark_mae'];steps=plan['base_config']['prior_fit_count']
execution_path=HERE/'execution_benchmark.json'
execution=json.loads(execution_path.read_text()) if execution_path.exists() else None
if execution:
 assert execution['status']=='PASS' and execution['selected_n_jobs'] in (4,8)
 assert all(x['tree_and_bootstrap_identity']=='PASS' and x['max_canonical_prediction_difference']==0 for x in execution['runs'])
assert not list(OUT.iterdir());assert not LEADER.exists()
LEADER.mkdir()
def clean(directory):
 assert directory.resolve() in [OUT.resolve(),LEADER.resolve()] and directory.resolve().is_relative_to(ROOT.resolve()) and not directory.is_symlink()
 for p in directory.iterdir():
  assert p.is_file() and not p.is_symlink() and p.resolve().parent==directory.resolve();p.unlink()
def move(a,b):
 assert a in [OUT,LEADER] and b in [OUT,LEADER] and not list(b.iterdir())
 for p in list(a.iterdir()):p.rename(b/p.name)
for name,feature in plan['candidates'].items():
 c=copy.deepcopy(plan['base_config']);c['version']='ensemble_context_'+name;c['additions'][name]=feature;c['prior_fit_count']=steps
 if execution:
  assert c['forest']['n_estimators']==execution['trees_per_fit'] and c['seeds']==execution['production_seeds_unchanged']
  c['forest']['n_jobs']=execution['selected_n_jobs']
  c['execution_policy']={'timing_receipt':str(execution_path.relative_to(ROOT)),'receipt_sha256':hashlib.sha256(execution_path.read_bytes()).hexdigest(),'change':'Worker count only; trees, seeds, data, scoring and selection unchanged.'}
 p=HERE/'context_config.json'
 if p.exists():p.read_text()
 p.write_text(json.dumps(c,indent=2))
 print('START CANDIDATE '+name,flush=True)
 subprocess.run([sys.executable,'-u',str(HERE/'context_feature.py')],check=True,cwd=ROOT)
 r=json.loads((OUT/'RESULT.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['status']=='PASS'
 replay=SUB/'magnitude_current/replay.py';s=replay.read_text().replace("OUT=HERE/'run_001'","OUT=HERE.parent/'magnitude_active'/'run_001'")
 exec(compile(s,str(replay),'exec'),{'__file__':str(replay)})
 results.append({'experiment':c['version'],'config':c,'result':r,'verification':v,'inner_scores':json.loads((OUT/'inner_scores.json').read_text()),'fold_MAE':(OUT/'fold_MAE.csv').read_text()})
 steps=r['cumulative_counted_steps']
 p=HERE/'context_results.json'
 if p.exists():p.read_text()
 p.write_text(json.dumps({'plan':plan,'results':results,'cumulative_counted_steps':steps},indent=2))
 if r['metrics']['prediction']['MAE']<best-1e-12:
  clean(LEADER);move(OUT,LEADER);best=r['metrics']['prediction']['MAE']
 else: clean(OUT)
 print('COMPLETED '+name+' MAE '+str(r['metrics']['prediction']['MAE']),flush=True)
if list(LEADER.iterdir()):move(LEADER,OUT)
LEADER.rmdir()
print('BATCH COMPLETE; best MAE '+str(best),flush=True)
