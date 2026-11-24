"""RF bootstrap-size selection with fixed earlier-selected depth/leaf and nested blend weights."""
from pathlib import Path
import json,hashlib,joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from feature_experiment import load,splits,transform,anchors,pair_prediction
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'run_001';CACHE=HERE.parent/'magnitude_current/run_001'
c=json.loads((HERE/'bootstrap_config.json').read_text());assert not (OUT/'RESULT.json').exists()
X,y=load(c,True);target=c['target_column'];base=pd.read_parquet(CACHE/'predictions.parquet')
np.testing.assert_array_equal(base.actual,y.loc[base.index,target]);assert abs((base.actual-base.prediction).abs().mean()-c['benchmark_mae'])<1e-12
paths=[Path(__file__),HERE/'bootstrap_config.json',HERE/'feature_experiment.py']+[ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
paths += [p for p in CACHE.iterdir() if p.suffix in ['.json','.joblib','.parquet']]
paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]+[ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2));(OUT/'config.json').write_text(json.dumps(c,indent=2));(OUT/'environment.json').write_text(json.dumps({'sklearn':sklearn.__version__}))
ledger=[];parts=[];selections=[];score_rows=[]
def fit(fold,role,sample,train,valid,cutoff,params):
 assert len(ledger)<c['new_fit_budget'] and sample<=len(train)
 A,prep=transform(X.loc[train])
 row={'fold':fold,'role':role,'max_samples':sample,'rows':len(train),'cutoff':str(cutoff),'max_label_end':str(y.loc[train,c['label_end_column']].max()),'status':'started'}
 ledger.append(row);(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2));print(f'Fit {len(ledger)}/12 maximum: fold {fold}, {role}, bootstrap draws {sample}',flush=True)
 model=RandomForestRegressor(**dict(params,max_samples=sample)).fit(A,y.loc[train,target])
 assert all(len(indices)==sample for indices in model.estimators_samples_)
 row['status']='completed';(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2))
 return model,prep


def selection_key(r):
 return (r['MAE'],r['weight'],0 if r['max_samples']==old_sample else 1,r['max_samples'])

for fold,ot,ov,it,iv in splits(c,y):
 start=pd.Timestamp(c['validation_windows'][fold-1][0]);inner_start=start-pd.DateOffset(months=c['inner_validation_months'])
 inner,outer=anchors(c,fold);card=json.loads((CACHE/f'fold_{fold}_selected.json').read_text());old=card['selection'];old_sample=old.get('max_samples',c['forest']['max_samples'])
 params=dict(c['forest'],max_depth=old['depth'],min_samples_leaf=old['leaf'])
 ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 _,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep)
 assert all(X.loc[it,f].count()>=1000 for f in X)
 scores=[]
 for sample in c['sample_sizes']:
  if sample==old_sample:
   model=joblib.load(CACHE/f"fold_{fold}_d{old['depth']}_l{old['leaf']}_inner.joblib")
  else:
   model,_=fit(fold,'inner',sample,it,iv,inner_start,params)
  joblib.dump(model,OUT/f'fold_{fold}_samples_{sample}_inner.joblib')
  fp=np.maximum(model.predict(B),0)
  for w in c['forest_weights']:scores.append({'fold':fold,'max_samples':sample,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 uniform=min([r for r in scores if r['max_samples']==old_sample],key=selection_key);assert uniform['weight']==old['weight']
 chosen=min(scores,key=selection_key);sample=chosen['max_samples'];w=chosen['weight'];score_rows.extend(scores)
 _,prep=transform(X.loc[ot]);B,_=transform(X.loc[ov],prep);assert prep==card['preprocessing']
 if sample==old_sample:model=joblib.load(CACHE/f'fold_{fold}_selected.joblib')
 else:model,prep=fit(fold,'outer_selected',sample,ot,ov,start,params)
 fp=np.maximum(model.predict(B),0)
 selections.append({**chosen,'depth':old['depth'],'leaf':old['leaf']})
 joblib.dump(model,OUT/f'fold_{fold}_selected.joblib')
 (OUT/f'fold_{fold}_selected.json').write_text(json.dumps({'selection':selections[-1],'preprocessing':prep,'training_rows':len(ot)},indent=2))
 # Keep compatibility with current replay and future fixed-input experiments.
 selected_inner=joblib.load(OUT/f'fold_{fold}_samples_{sample}_inner.joblib')
 joblib.dump(selected_inner,OUT/f"fold_{fold}_d{old['depth']}_l{old['leaf']}_inner.joblib")
 parts.append(pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'forest':fp,'prediction':(1-w)*op+w*fp,'weight':w,'incumbent':base.loc[ov,'prediction']},index=ov))
 print(f'Fold {fold} selected bootstrap draws {sample}, RF weight {w}',flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index);assert np.isfinite(pred[['actual','prediction']]).all().all()
# Replay every inner score and outer selected model from saved individual trees.
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold);ip=pair_prediction(X.loc[iv,c['anchor_features']],inner)
 _,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep);replayed=[]
 for sample in c['sample_sizes']:
  model=joblib.load(OUT/f'fold_{fold}_samples_{sample}_inner.joblib');sel=selections[fold-1]
  for k,v in dict(c['forest'],max_depth=sel['depth'],min_samples_leaf=sel['leaf'],max_samples=sample).items():assert model.get_params()[k]==v
  assert all(len(indices)==sample for indices in model.estimators_samples_)
  fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0)
  for w in c['forest_weights']:replayed.append({'fold':fold,'max_samples':sample,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 saved=[r for r in score_rows if r['fold']==fold]
 for r,s in zip(replayed,saved):assert abs(r['MAE']-s['MAE'])<1e-10
 chosen=min(replayed,key=selection_key);sel=selections[fold-1];assert chosen['max_samples']==sel['max_samples'] and chosen['weight']==sel['weight']
 card=json.loads((OUT/f'fold_{fold}_selected.json').read_text());_,prep=transform(X.loc[ot]);assert prep==card['preprocessing']
 B,_=transform(X.loc[ov],prep);model=joblib.load(OUT/f'fold_{fold}_selected.joblib')
 for k,v in dict(c['forest'],max_depth=sel['depth'],min_samples_leaf=sel['leaf'],max_samples=sel['max_samples']).items():assert model.get_params()[k]==v
 assert all(len(indices)==sel['max_samples'] for indices in model.estimators_samples_)
 fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0);op=pair_prediction(X.loc[ov,c['anchor_features']],outer);w=sel['weight']
 np.testing.assert_allclose((1-w)*op+w*fp,pred.loc[ov,'prediction'],rtol=0,atol=1e-10)
assert all(row['status']=='completed' for row in ledger)
for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
pred.to_parquet(OUT/'predictions.parquet');(OUT/'inner_scores.json').write_text(json.dumps(score_rows,indent=2))
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs();metrics={n:{'MAE':float(loss[n].mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in loss}
for n in metrics:assert abs(metrics[n]['Spearman_IC']-np.corrcoef(pred.actual.rank(),pred[n].rank())[0,1])<1e-12
loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv');loss.groupby(pred.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int);sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy();draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)));samples=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(samples,[.025,.975]).tolist()})
r={'status':'VERIFIED','experiment':c['version'],'metrics':metrics,'selected_settings':selections,'rows':len(pred),'new_model_fits':len(ledger),'new_selection_steps':4,'cumulative_counted_steps':c['prior_fit_count']+len(ledger)+4,'nominal_intervals':intervals,'interpretation':'All eligible history retained with equal training weights. Bootstrap draws per tree selected from 4096/8192/16384 using earlier inner validation, jointly with blend weight. Existing depth/leaf settings and 128 trees held fixed. Sampling is with replacement. Adaptive development; fresh test not accessed.'}
(OUT/'RESULT.json').write_text(json.dumps(r,indent=2));(OUT/'verification.json').write_text(json.dumps({'status':'PASS','rows':len(pred),'checks':['source hashes','strict label-end purges','exact targets and rows','fixed earlier-selected tree settings','actual bootstrap index counts checked for every saved forest','Incumbent sample-size option reproduces incumbent inner blend selection','all inner scores replayed','all selected outer predictions replayed','MAE and Spearman recomputed'],'new_verification_fits':0},indent=2))
print(json.dumps(r,indent=2),flush=True)
