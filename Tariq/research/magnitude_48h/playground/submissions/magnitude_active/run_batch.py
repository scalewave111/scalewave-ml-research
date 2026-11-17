"""Execute the registered feature batch in one reusable output area."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
SUB=HERE.parent
OUT=HERE/'run_001';LEADER=HERE/'leader'
plan=json.loads((HERE/'batch_plan.json').read_text())
base=json.loads((SUB/'magnitude_current/run_001/config.json').read_text())
registry=json.loads((SUB/'magnitude_benchmark_registry.json').read_text())
assert registry['current']['MAE']==plan['benchmark_mae']
assert not (HERE/'batch_results.json').exists()
LEADER.mkdir(exist_ok=True)
assert not list(LEADER.iterdir())
def safe_files(folder):
 assert folder.resolve() in [OUT.resolve(),LEADER.resolve()] and folder.resolve().is_relative_to(HERE.resolve())
 files=list(folder.iterdir())
 for p in files:
  assert p.is_file() and not p.is_symlink() and p.resolve().parent==folder.resolve()
 return files

def clear(folder):
 for p in safe_files(folder):p.unlink()

results=[];best=plan['benchmark_mae'];count=plan['prior_counted_steps']
for i,features in enumerate(plan['candidates'],1):
 clear(OUT)
 c=dict(base)
 c.update(version='tuned_flow_plus_'+'_'.join(features),additions={**base['additions'],**{f:plan['features'][f] for f in features}},
  authorization='User authorized venue fragmentation plus temporal concentration, and venue fragmentation plus price range, on top of pair plus flow with unchanged nested tuning.',
  benchmark_predictions=registry['current']['prediction_file'],benchmark_column=registry['current']['prediction_column'],
  benchmark_mae=plan['benchmark_mae'],new_fit_budget=40,prior_fit_count=count,
  change_scope='Only RF gains listed PDF features; liquidation pair and nine-setting nested tuning protocol unchanged.',
  fit_accounting='36 inner RF fits and 4 selected outer RF fits plus 4 joint selections per candidate.',
  stopping_rule=plan['stopping_rule'])
 (HERE/'candidate_config.json').write_text(json.dumps(c,indent=2))
 print(f"START candidate {i}/{len(plan['candidates'])}: {features}",flush=True)
 subprocess.run([sys.executable,'-u',str(HERE/'tune_candidate.py')],cwd=HERE,check=True)
 r=json.loads((OUT/'RESULT.json').read_text());v=json.loads((OUT/'verification.json').read_text())
 assert r['status']=='VERIFIED' and v['status']=='PASS'
 assert r['new_model_fits']==40
 count=r['cumulative_counted_steps']
 entry={'features':features,'config':c,'result':r,'verification':v,
  'inner_scores':json.loads((OUT/'inner_scores.json').read_text()),
  'fold_MAE_csv':(OUT/'fold_MAE.csv').read_text(),'monthly_MAE_csv':(OUT/'monthly_MAE.csv').read_text()}
 score=r['metrics']['prediction']['MAE']
 if score<best-1e-12:
  clear(LEADER)
  selected={f"fold_{x['fold']}_d{x['depth']}_l{x['leaf']}_inner.joblib" for x in r['selected_settings']}
  for p in safe_files(OUT):
   if p.name.endswith('_inner.joblib') and p.name not in selected:continue
   target=LEADER/p.name;assert not target.exists();p.rename(target)
  best=score
  entry['decision']='NEW_BATCH_LEADER'
 else:entry['decision']='NO_PROMOTION'
 results.append(entry)
 (HERE/'batch_results.json').write_text(json.dumps({'plan':plan,'results':results,'best_MAE':best,'counted_steps':count},indent=2))
 clear(OUT)
 print(f"DONE candidate {i}/{len(plan['candidates'])}: MAE={score:.9f}; best={best:.9f}; temporary losing artifacts removed",flush=True)
print('BATCH COMPLETE: '+str(best),flush=True)
