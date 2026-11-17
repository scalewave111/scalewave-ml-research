"""One extra feature in the fixed three-seed ensemble, with earlier-only blend selection."""
from pathlib import Path
import json,hashlib,joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from feature_experiment import load,splits,transform,anchors,pair_prediction
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'run_001'; CACHE=HERE.parent/'magnitude_current/run_001'
c=json.loads((HERE/'context_config.json').read_text())
resuming=(OUT/'config.json').exists()
if resuming: assert json.loads((OUT/'config.json').read_text())==c
else: assert not list(OUT.iterdir())
X,y=load(c,True); target=c['target_column'];base=pd.read_parquet(CACHE/'predictions.parquet')
assert abs((base.actual-base.prediction).abs().mean()-c['benchmark_mae'])<1e-12
np.testing.assert_array_equal(base.actual,y.loc[base.index,target])
paths=[Path(__file__),HERE/'context_config.json',HERE/'feature_experiment.py']+[ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
paths += [p for p in CACHE.iterdir() if p.suffix in ['.json','.joblib','.parquet']]
paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]+[ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def save(name,o): (OUT/name).write_text(json.dumps(o,indent=2))
if resuming: assert json.loads((OUT/'input_hashes.json').read_text())==hashes
save('input_hashes.json',hashes);save('config.json',c)
parts=[];scores=[];selections=[]
ledger=json.loads((OUT/'fit_ledger.json').read_text()) if (OUT/'fit_ledger.json').exists() else []
ledger=[row for row in ledger if row['status']=='completed']
def predict(models,A,params):
 values=[]
 for seed,model in zip(c['seeds'],models):
  for k,v in dict(params,random_state=seed).items(): assert model.get_params()[k]==v
  assert all(len(indices)==params['max_samples'] for indices in model.estimators_samples_)
  p=np.maximum(np.mean([t.predict(A) for t in model.estimators_],axis=0),0)
  np.testing.assert_allclose(p,model.predict(A),atol=1e-10,rtol=0)
  values.append(p)
 return {**{str(seed):p for seed,p in zip(c['seeds'],values)},'ensemble':np.mean(values,axis=0)}
for fold,ot,ov,it,iv in splits(c,y):
 card=json.loads((CACHE/f'fold_{fold}_selected.json').read_text()); old=card['selection']
 params=dict(c['forest'],max_depth=old['depth'],min_samples_leaf=old['leaf'],max_samples=old['max_samples'])
 inner,outer=anchors(c,fold);ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 predictions={};preps={}
 for role,train,valid in [('inner',it,iv),('outer',ot,ov)]:
  assert all(X.loc[train,f].count()>=1000 for f in X)
  A,prep=transform(X.loc[train]);B,_=transform(X.loc[valid],prep);preps[role]=prep
  name=f"fold_{fold}_d{old['depth']}_l{old['leaf']}_inner.joblib" if role=='inner' else f'fold_{fold}_selected.joblib'
  models=joblib.load(OUT/name) if (OUT/name).exists() else []
  assert len(models)<=len(c['seeds'])
  for seed_index,seed in enumerate(c['seeds']):
   if seed_index<len(models):
    model=models[seed_index]
    for k,v in dict(params,random_state=seed).items():assert model.get_params()[k]==v
    if not any(row['fold']==fold and row['role']==role and row['seed']==seed for row in ledger):
     ledger.append({'fold':fold,'role':role,'seed':seed,'status':'completed','recovered_from_checkpoint':True});save('fit_ledger.json',ledger)
    continue
   cutoff=pd.Timestamp(c['validation_windows'][fold-1][0]);cutoff=cutoff-pd.DateOffset(months=c['inner_validation_months']) if role=='inner' else cutoff
   assert y.loc[train,c['label_end_column']].max()<cutoff and len(ledger)<c['new_fit_budget']
   ledger.append({'fold':fold,'role':role,'seed':seed,'rows':len(train),'max_label_end':str(y.loc[train,c['label_end_column']].max()),'cutoff':str(cutoff),'status':'started'})
   save('fit_ledger.json',ledger);print(f'Fit {len(ledger)}/24: period {fold}, {role}, seed {seed}',flush=True)
   model=RandomForestRegressor(**dict(params,random_state=seed)).fit(A,y.loc[train,target])
   models.append(model)
   checkpoint=OUT/(name+'.partial')
   joblib.dump(models,checkpoint);checkpoint.replace(OUT/name)
   ledger[-1]['status']='completed';save('fit_ledger.json',ledger)
  name=f"fold_{fold}_d{old['depth']}_l{old['leaf']}_inner.joblib" if role=='inner' else f'fold_{fold}_selected.joblib'
  assert len(models)==len(c['seeds'])
  predictions[role]=predict(joblib.load(OUT/name),B,params)
 assert {f:preps['outer'][f] for f in card['preprocessing']}==card['preprocessing']
 output=pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'incumbent':base.loc[ov,'prediction']},index=ov)
 chosen={}
 for name in ['ensemble']:
  rows=[{'fold':fold,'variant':name,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*predictions['inner'][name])).mean())} for w in c['forest_weights']]
  scores.extend(rows);sel=min(rows,key=lambda r:(r['MAE'],r['weight']));chosen[name]=sel;w=sel['weight']
  output['prediction' if name=='ensemble' else 'seed_'+name]=(1-w)*op+w*predictions['outer'][name]
 selected={**chosen['ensemble'],'depth':old['depth'],'leaf':old['leaf'],'max_samples':old['max_samples'],'seeds':c['seeds']}
 selections.append(selected);save(f'fold_{fold}_selected.json',{'selection':selected,'preprocessing':preps['outer'],'training_rows':len(ot)})
 output['forest']=predictions['outer']['ensemble'];output['weight']=selected['weight'];parts.append(output)
 print(f"Period {fold} complete: ensemble RF weight {selected['weight']}",flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index) and np.isfinite(pred.to_numpy()).all()
assert all(r['status']=='completed' for r in ledger)
for p,sha in hashes.items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
pred.to_parquet(OUT/'predictions.parquet');save('inner_scores.json',scores)
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs()
metrics={n:{'MAE':float(loss[n].mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in loss}
loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv');loss.groupby(pred.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int);sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy();draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)));values=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(values,[.025,.975]).tolist()})
r={'status':'VERIFIED','experiment':c['version'],'metrics':metrics,'selected_settings':selections,'rows':len(pred),'new_model_fits':len(ledger),'new_selection_steps':4,'cumulative_counted_steps':c['prior_fit_count']+len(ledger)+4,'nominal_intervals':intervals,'interpretation':'One additional context feature in RF only. Fixed seeds 48/17/91 equally averaged; incumbent per-fold depth, leaf size, bootstrap samples and 128 trees per seed unchanged. Earlier validation chooses blend weight on original grid. Same target, rows and temporal purges. Adaptive development; fresh data untouched.'}
save('RESULT.json',r);save('verification.json',{'status':'PASS','rows':len(pred),'checks':['source hashes unchanged','strict label-end purges','exact targets and timestamps','fixed params and seed identities','saved forests replayed through individual trees for inner and outer predictions','earlier-only blend selection'],'new_verification_fits':0})
print(json.dumps(r,indent=2),flush=True)
