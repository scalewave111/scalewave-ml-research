"""Verify and finalize one bootstrap comparison, preserving only the incumbent models."""
from pathlib import Path
import json, hashlib
import pandas as pd
from scipy.stats import pearsonr

HERE=Path(__file__).resolve().parent; SUB=HERE.parent; ROOT=SUB.parent
SRC=HERE/'run_001'; CURRENT=SUB/'magnitude_current'; DEST=CURRENT/'run_001'
def read(p): return json.loads(p.read_text())
def write(p,value):
 if p.exists(): p.read_text()
 p.write_text(json.dumps(value,indent=2))
reg=read(SUB/'magnitude_benchmark_registry.json'); history=read(SUB/'experiment_history.json')
r=read(SRC/'RESULT.json'); c=read(SRC/'config.json'); v=read(SRC/'verification.json')
assert r['status']=='VERIFIED' and v['status']=='PASS'
assert c['benchmark_mae']==reg['current']['MAE']
assert not any(e.get('experiment')==r['experiment'] for e in reg['evaluated_experiments'])
pred=pd.read_parquet(SRC/'predictions.parquet'); prior=pd.read_parquet(DEST/'predictions.parquet')
assert pred.index.equals(prior.index) and pred.actual.equals(prior.actual)
assert (pred.incumbent==prior.prediction).all()
replay=CURRENT/'replay.py'; source=replay.read_text()
alternate=source.replace("OUT=HERE/'run_001'", "OUT=HERE.parent/'magnitude_active'/'run_001'")
exec(compile(alternate,str(replay),'exec'),{'__file__':str(replay)})
old=reg['current'].copy(); metrics=r['metrics']['prediction']; promote=metrics['MAE']<old['MAE']-1e-12
decision='PROMOTE' if promote else 'NO_PROMOTION'
summary=f"{r['experiment']}: {decision}\nPrevious benchmark: {old['MAE']}\nCandidate MAE: {metrics['MAE']}\nCandidate Spearman IC: {metrics['Spearman_IC']}\n"
summary+=r['interpretation']+'\nSelected settings: '+json.dumps(r['selected_settings'])+'\n'
summary+=f"{len(c['anchor_features'])+len(c['additions'])} raw features, four quarterly periods, 18,481 rows, strict label-end purges.\n"
summary+=f"New fits: {r['new_model_fits']}; selections: {r['new_selection_steps']}; cumulative counted steps: {r['cumulative_counted_steps']}.\n"
summary+='Candidate and retained model independently replayed. Unused generated models pruned.\nRepeated development selection; fresh data not accessed. No global optimum established.\n'
for p in [HERE/'SUMMARY.txt',SRC/'SUMMARY.txt']:
 if p.exists(): p.read_text()
 p.write_text(summary)
history.append({'experiment':r['experiment'],'result':r,'config':c,'verification':v,'inner_scores':read(SRC/'inner_scores.json'),'retained_selected_models':promote,'retention':'Only winning selected models retained; other binary/prediction artifacts intentionally pruned.'})
if promote:
 for e in history:
  if e.get('experiment')==old['experiment']:
   e['retained_selected_models']=False;e['retained_full_artifacts']=False;e['retention']='Superseded by verified replacement; compact scores retained.'
write(SUB/'experiment_history.json',history)
for d in (SRC,DEST):
 assert d.resolve().is_relative_to(ROOT.resolve()) and not d.is_symlink()
 assert all(p.is_file() and not p.is_symlink() and p.resolve().parent==d.resolve() for p in d.iterdir())
if promote: prune=list(DEST.iterdir())+list(SRC.glob('fold_*_inner.joblib'))
else: prune=[p for p in SRC.iterdir() if p.suffix in ('.joblib','.parquet') or p.name.endswith('_selected.json')]
receipt={'experiment':r['experiment'],'scope':[str(SRC),str(DEST)],'delete':[{'path':str(p),'bytes':p.stat().st_size} for p in prune],'reclaimed_bytes':sum(p.stat().st_size for p in prune)}
for p in prune:
 assert p.resolve().parent in (SRC.resolve(),DEST.resolve());p.unlink()
if promote:
 for p in list(SRC.iterdir()): p.rename(DEST/p.name)
 reg['current'].update(experiment=r['experiment'],**metrics,Pearson_IC=float(pearsonr(pred.actual,pred.prediction).statistic),prediction_sha256=hashlib.sha256((DEST/'predictions.parquet').read_bytes()).hexdigest())
 reg['history'].append({'model':r['experiment'],**metrics,'Pearson_IC':reg['current']['Pearson_IC']})
 p=CURRENT/'FINAL_HANDOFF.txt'; previous=p.read_text(); recipe=previous[previous.index('MODEL RECIPE'):]
 recipe=recipe.replace('128 trees, absolute_error, bootstrap=True, max_features=1.0, seed=48, equal row weights.', '128 trees per forest, absolute_error, bootstrap=True, max_features=1.0, equal row weights. Seeds and averaging are specified above.' if 'seeds' in c else '128 trees, absolute_error, bootstrap=True, max_features=1.0, seed=48, equal row weights.')
 recipe='\n'.join(line for line in recipe.splitlines() if not line.startswith('Depth/leaf by period:'))
 recipe='MODEL RECIPE\nRaw RF features: '+', '.join(c['anchor_features']+list(c['additions'].values()))+'\nLiquidation model uses the original anchor pair. Three equally averaged 128-tree forests; settings in selections above.\nQuarterly expanding history, earlier three-month blend validation, strict label-end purges.\nTraining-only preprocessing. Adaptive development; separate fresh data not accessed.\nReplay dependencies: retained v2 outer pair cards and v4 inner pair cards.\n'
 p.write_text(summary+'\n'+recipe+'\nCurrent per-period depth, leaf and bootstrap settings are listed in selected settings above.\n')
exec(compile(source,str(replay),'exec'),{'__file__':str(replay)})
assert hashlib.sha256((DEST/'predictions.parquet').read_bytes()).hexdigest()==reg['current']['prediction_sha256']
reg['evaluated_experiments'].append({'experiment':r['experiment'],'decision':decision,**metrics,'new_model_fits':r['new_model_fits'],'new_selection_steps':r['new_selection_steps'],'verification_status':'PASS before authorized pruning','details':'submissions/experiment_history.json'})
reg['research_status']=r['experiment']+' COMPLETE; NO_RUNNING_EXPERIMENTS'
write(SUB/'magnitude_benchmark_registry.json',reg)
p=SUB/'STORAGE_POLICY.txt'; lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('Current best:'): lines[i]=f"Current best: magnitude_current ({reg['current']['experiment']}), MAE {reg['current']['MAE']}."
p.write_text('\n'.join(lines)+'\n')
receipt.update(status='COMPLETE',candidate_replay='PASS',post_cleanup_incumbent_replay='PASS')
write(HERE/'cleanup.json',receipt)
print(json.dumps({'decision':decision,'current':reg['current'],'removed_bytes':receipt['reclaimed_bytes']},indent=2))
