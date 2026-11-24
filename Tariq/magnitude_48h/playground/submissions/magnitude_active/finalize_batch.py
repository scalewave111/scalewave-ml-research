"""Publish a completed registered batch, retain only the verified winner."""
from pathlib import Path
import json,hashlib
import pandas as pd
from scipy.stats import pearsonr
HERE=Path(__file__).resolve().parent;SUB=HERE.parent;ROOT=SUB.parent
CURRENT=SUB/'magnitude_current';OUT=CURRENT/'run_001';LEADER=HERE/'leader'
batch=json.loads((HERE/'batch_results.json').read_text())
assert len(batch['results'])==len(batch['plan']['candidates'])==2
p=SUB/'magnitude_benchmark_registry.json';reg=json.loads(p.read_text())
assert reg['current']['MAE']==batch['plan']['benchmark_mae']
old=reg['current'].copy()
rows=[{'features':'Current pair + flow benchmark','MAE':old['MAE'],'Spearman_IC':old['Spearman_IC']}]
for e in batch['results']:
 assert e['verification']['status']=='PASS' and e['result']['status']=='VERIFIED'
 rows.append({'features':' + '.join(e['features']),**e['result']['metrics']['prediction']})
winner=min(batch['results'],key=lambda e:e['result']['metrics']['prediction']['MAE'])
promote=winner['result']['metrics']['prediction']['MAE']<old['MAE']-1e-12
history_path=SUB/'experiment_history.json';history=json.loads(history_path.read_text())
for e in batch['results']:
 history.append({'experiment':e['config']['version'],'features':e['features'],'details':e['result'],
  'grid_depths':e['config']['grid_depths'],'grid_leaves':e['config']['grid_leaves'],
  'retained_selected_models':promote and e is winner,'verification_status':'All inner grid and selected outer replays PASS before pruning'})
# Preserve the old compact winner evidence before replacing its reusable directory.
old_result=json.loads((OUT/'RESULT.json').read_text())
for e in history:
 if promote and e.get('experiment')==old['experiment']:
  e['retained_full_artifacts']=False;e['retention']='Superseded after verified replacement; compact scores retained.'
  e['details']=old_result
history_path.write_text(json.dumps(history,indent=2)+'\n')
if promote:
 replay_path=CURRENT/'replay.py';source=replay_path.read_text()
 alternate=source.replace("OUT=HERE/'run_001'","OUT=HERE.parent/'magnitude_active'/'leader'")
 exec(compile(alternate,str(replay_path),'exec'),{'__file__':str(replay_path)})
 leader_result=json.loads((LEADER/'RESULT.json').read_text())
 assert leader_result['experiment']==winner['config']['version']
 # The two exact output directories are the only permitted mutation targets.
 assert OUT.resolve()==(SUB/'magnitude_current/run_001').resolve()
 assert LEADER.resolve()==(SUB/'magnitude_active/leader').resolve()
 oldfiles=list(OUT.iterdir());newfiles=list(LEADER.iterdir())
 for f in oldfiles+newfiles:
  assert f.is_file() and not f.is_symlink() and f.resolve().parent in [OUT.resolve(),LEADER.resolve()]
 for f in oldfiles:f.unlink()
 for f in newfiles:f.rename(OUT/f.name)
 exec(compile(source,str(replay_path),'exec'),{'__file__':str(replay_path)})
 pred=pd.read_parquet(OUT/'predictions.parquet');m=winner['result']['metrics']['prediction']
 reg['current']={'experiment':winner['config']['version'],'prediction_file':'submissions/magnitude_current/run_001/predictions.parquet',
  'prediction_column':'prediction',**m,'Pearson_IC':float(pearsonr(pred.actual,pred.prediction).statistic),
  'verification':'submissions/magnitude_current/run_001/verification.json','IC_verification':'submissions/magnitude_current/run_001/verification.json',
  'prediction_sha256':hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest(),'replay_script':'submissions/magnitude_current/replay.py'}
 reg['history'].append({'model':reg['current']['experiment'],**m,'Pearson_IC':reg['current']['Pearson_IC']})
for e in batch['results']:
 reg['evaluated_experiments'].append({'experiment':e['config']['version'],'decision':'PROMOTE' if promote and e is winner else 'NO_PROMOTION',
  **e['result']['metrics']['prediction'],'features':e['features'],'new_model_fits':40,'new_selection_steps':4,
  'verification_status':'PASS before intentional losing-model pruning','details':'submissions/experiment_history.json'})
p.write_text(json.dumps(reg,indent=2)+'\n')
frame=pd.DataFrame(rows);frame['delta_MAE_vs_start']=frame.MAE-old['MAE']
if (HERE/'batch_comparison.csv').exists(): (HERE/'batch_comparison.csv').read_text()
frame.to_csv(HERE/'batch_comparison.csv',index=False)
text='VERIFIED TWO CONTEXT FEATURE PAIRS ADDED TO TUNED PAIR + FLOW\n\n'+frame.to_string(index=False)
text+='\n\nLiquidation model retains its original pair. RF retains that pair plus 1h flow activity percentile, and gains the listed pair of context features.\nSame nine depth/leaf configurations and earlier three-month MAE blend selection.\nSame four quarterly periods and 18,481 rows. Strict label-end purges.\nBoth candidates replay-verified; 80 new model fits plus 8 selection steps.\n'
text+='Decision: '+('PROMOTE '+reg['current']['experiment'] if promote else 'NO_PROMOTION')+'\n'
text+='Current MAE: '+str(reg['current']['MAE'])+'; Spearman IC: '+str(reg['current']['Spearman_IC'])+'\n'
text+='Lowest-scoring challenger settings: '+json.dumps(winner['result']['selected_settings'])+'\n'
text+='Lowest-scoring challenger nominal intervals: '+json.dumps(winner['result']['nominal_intervals'])+'\n'
text+='Adaptive development comparisons; nominal intervals do not correct repeated experiment selection. Fresh test not accessed.\n'
text+='Losing grid models and superseded winner files removed; selected winner and compact history retained.\n'
if (HERE/'SUMMARY.txt').exists(): (HERE/'SUMMARY.txt').read_text()
(HERE/'SUMMARY.txt').write_text(text)
if promote:(OUT/'SUMMARY.txt').write_text(text)
policy=SUB/'STORAGE_POLICY.txt';s=policy.read_text();lines=s.splitlines()
for i,line in enumerate(lines):
 if line.startswith('Current best:'):lines[i]='Current best: magnitude_current ('+reg['current']['experiment']+'), MAE '+str(reg['current']['MAE'])+'.'
policy.write_text('\n'.join(lines)+'\n')
print(frame.to_string(index=False))
print('Final benchmark:',reg['current']['MAE'])
