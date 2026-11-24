from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent;SUB=HERE.parent;OUT=HERE/'run_001';LEADER=HERE/'leader'
c=json.loads((HERE/'boost_config.json').read_text());history=json.loads((SUB/'experiment_history.json').read_text())
if (OUT/'RESULT.json').exists():
 old=json.loads((OUT/'RESULT.json').read_text());assert any(e.get('experiment')==old['experiment'] for e in history)
assert not (HERE/'boost_results.json').exists()
LEADER.mkdir(exist_ok=True);assert not list(LEADER.iterdir())
def files(folder):
 assert folder.resolve() in [OUT.resolve(),LEADER.resolve()] and folder.resolve().is_relative_to(HERE.resolve())
 result=list(folder.iterdir())
 for p in result:assert p.is_file() and not p.is_symlink() and p.resolve().parent==folder.resolve()
 return result
def clear(folder):
 for p in files(folder):p.unlink()
results=[];best=c['benchmark_mae']
for mode in ['direct','residual']:
 clear(OUT);print('START '+mode,flush=True)
 subprocess.run([sys.executable,'-u',str(HERE/'boost_experiment.py'),mode],cwd=HERE,check=True)
 r=json.loads((OUT/'RESULT.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert r['status']=='VERIFIED' and v['status']=='PASS'
 entry={'mode':mode,'config':json.loads((OUT/'config.json').read_text()),'result':r,'verification':v,'inner_scores':json.loads((OUT/'inner_scores.json').read_text()),'fold_MAE_csv':(OUT/'fold_MAE.csv').read_text(),'monthly_MAE_csv':(OUT/'monthly_MAE.csv').read_text()}
 if r['metrics']['prediction']['MAE']<best-1e-12:
  clear(LEADER)
  selected={f"fold_{s['fold']}_d{s['depth']}_n{s['trees']}_inner.ubj" for s in r['selected_settings']}
  for p in files(OUT):
   if p.name.endswith('_inner.ubj') and p.name not in selected:continue
   target=LEADER/p.name;assert not target.exists();p.rename(target)
  best=r['metrics']['prediction']['MAE'];entry['decision']='NEW_BATCH_LEADER'
 else:entry['decision']='NO_PROMOTION'
 results.append(entry);(HERE/'boost_results.json').write_text(json.dumps({'starting_MAE':c['benchmark_mae'],'results':results,'best_MAE':best,'cumulative_counted_steps':c['prior_fit_count']+32*len(results)},indent=2))
 clear(OUT);print(f'DONE {mode}: best verified MAE {best}',flush=True)
