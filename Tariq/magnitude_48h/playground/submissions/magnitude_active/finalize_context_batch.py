"""Finalize the completed fixed-incumbent feature batch and compact inventory."""
from pathlib import Path
import json,hashlib,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;SUB=HERE.parent;ROOT=SUB.parent;OUT=HERE/'run_001'
def read(p):return json.loads(p.read_text())
def write(p,o):
 if p.exists():p.read_text()
 p.write_text(json.dumps(o,indent=2))
batch=read(HERE/'context_results.json');plan=read(HERE/'context_plan.json')
assert batch['plan']==plan and len(batch['results'])==len(plan['candidates'])
assert not (HERE/'context_leader').exists()
rp=SUB/'magnitude_benchmark_registry.json';reg=read(rp);old=reg['current'].copy()
assert old['MAE']==plan['base_config']['benchmark_mae']
assert all(e['verification']['status']=='PASS' for e in batch['results'])
best=min(batch['results'],key=lambda e:e['result']['metrics']['prediction']['MAE'])
promote=best['result']['metrics']['prediction']['MAE']<old['MAE']-1e-12
if promote:
 assert read(OUT/'RESULT.json')['experiment']==best['experiment']
 subprocess.run([sys.executable,str(HERE/'finalize_bootstrap.py')],check=True,cwd=ROOT)
else:assert not list(OUT.iterdir())
reg=read(rp);hp=SUB/'experiment_history.json';hist=read(hp)
inventory_path=HERE/'pdf_feature_inventory.json';inventory=read(inventory_path)
base_feature_count=len(plan['base_config']['anchor_features'])+len(plan['base_config']['additions'])
rows=[{'feature':f'Starting {base_feature_count}-feature ensemble','MAE':old['MAE'],'Spearman_IC':old['Spearman_IC']}]
for e in batch['results']:
 r=e['result'];m=r['metrics']['prediction'];name=e['experiment'].removeprefix('ensemble_context_');winner=promote and e['experiment']==best['experiment']
 rows.append({'feature':name,**m})
 if not any(x.get('experiment')==e['experiment'] for x in reg['evaluated_experiments']):
  hist.append({**e,'retained_full_artifacts':False,'retained_selected_models':False,'retention':'Unselected model binaries, cards and predictions intentionally pruned after replay. Compact evidence retained.'})
  reg['evaluated_experiments'].append({'experiment':e['experiment'],'decision':'NO_PROMOTION',**m,'new_model_fits':r['new_model_fits'],'new_selection_steps':r['new_selection_steps'],'verification_status':'PASS before intentional pruning','details':'submissions/experiment_history.json'})
 for f in inventory['features']:
  if f['pdf_name']==name:f.update(status='TESTED_CURRENT_BATCH',experiment=e['experiment'],MAE=m['MAE'],Spearman_IC=m['Spearman_IC'],promoted=winner)
reg['cumulative_counted_steps']=batch['cumulative_counted_steps'];reg['research_status']='PRIORITY_FEATURE_BATCH_COMPLETE; NO_RUNNING_EXPERIMENTS'
write(hp,hist);write(rp,reg);write(inventory_path,inventory)
run=subprocess.run([sys.executable,str(SUB/'magnitude_current/replay.py')],check=True,text=True,capture_output=True)
sha=hashlib.sha256((SUB/'magnitude_current/run_001/predictions.parquet').read_bytes()).hexdigest();assert sha==reg['current']['prediction_sha256']
if not promote:assert sha==old['prediction_sha256']
frame=pd.DataFrame(rows);frame['delta_MAE_vs_start']=frame.MAE-old['MAE']
summary='INDIVIDUAL PDF FEATURE ADDITIONS - VERIFIED\n\n'+frame.to_string(index=False)+'\n\n'
summary+='Current benchmark MAE: '+str(reg['current']['MAE'])+'; Spearman IC: '+str(reg['current']['Spearman_IC'])+'\n'
summary+=f'Same initial {base_feature_count}-feature ensemble plus one candidate at a time. Three fixed seeds, 128 trees per seed, fixed per-fold depths/leaves/bootstrap draws.\nEarlier three-month validation chooses blend on original grid. Same four periods, 18,481 timestamps and strict label-end purges.\n'
summary+=f"Completed {len(batch['results'])} candidate comparisons; {sum(e['result']['new_model_fits'] for e in batch['results'])} forests plus {sum(e['result']['new_selection_steps'] for e in batch['results'])} blend selections. Cumulative count {batch['cumulative_counted_steps']}.\n"
summary+='All candidates and retained winner replay verified. Losing generated outputs pruned; compact history and scripts retained.\nMAE decides promotion; Spearman tracked only. No combinations or neural networks tested. Fresh data untouched.\n'
summary+=f"Inventory: {sum(f['status']=='PENDING' for f in inventory['features'])} eligible features remain queued; {sum(f['status']=='INSUFFICIENT_TRAINING_COVERAGE' for f in inventory['features'])} unavailable under current training coverage.\n"
p=HERE/'SUMMARY.txt'
if p.exists():p.read_text()
p.write_text(summary)
receipt_path=HERE/'cleanup.json';receipt=read(receipt_path) if receipt_path.exists() else {}
write(receipt_path,{'status':'COMPLETE','scope':str(OUT),'experiments':[e['experiment'] for e in batch['results']],'pruning':'Unselected candidate artifacts removed after verification; verified winner and replay dependencies retained. Original data untouched.','active_directory_empty':not list(OUT.iterdir()),'post_cleanup_replay':json.loads(run.stdout),'current_prediction_sha256':sha,'promotion_cleanup':receipt if promote else None})
print(run.stdout);print(summary)
