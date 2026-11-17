"""Bounded nested walk-forward RF tuning, MAE selection, independent artifact replay."""
from pathlib import Path
import json, hashlib, platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from feature_experiment import load, splits, transform, anchors, pair_prediction
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'run_001'
c=json.loads((HERE/'fine_blend_config.json').read_text())
assert not (OUT/'RESULT.json').exists()
X,y=load(c,True);target=c['target_column']
base=pd.read_parquet(ROOT/c['benchmark_predictions'])
np.testing.assert_array_equal(base.actual,y.loc[base.index,target])
assert abs((base.actual-base[c['benchmark_column']]).abs().mean()-c['benchmark_mae'])<1e-12
assert c['horizon_minutes']==c['anchor_horizon_minutes']==c['benchmark_horizon_minutes']
prior=json.loads((ROOT/c['source_hybrid_config']).read_text())
assert c['forest']==prior['forest'] and c['validation_windows']==prior['validation_windows']
assert set(prior['forest_weight_candidates']).issubset(c['forest_weights']) and c['anchor_features']==prior['features']
assert c['forest_weights']==[i/20 for i in range(11)]
CACHE=ROOT/'submissions/magnitude_current/run_001'
cached_config=json.loads((CACHE/'config.json').read_text())
for k in ['forest','anchor_features','additions','validation_windows','grid_depths','grid_leaves']:
 assert c[k]==cached_config[k]
paths=[HERE/'fine_blend_config.json',Path(__file__),HERE/'feature_experiment.py',ROOT/c['benchmark_predictions']]
paths += [ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
for f in range(1,5):
 paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json',ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json']
paths += [p for p in CACHE.iterdir() if p.suffix in ['.joblib','.json']]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'config.json').write_text(json.dumps(c,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'sklearn':sklearn.__version__},indent=2))
(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
ledger=[];parts=[];choices=[];scores_all=[]
def fit(fold,role,depth,leaf,A,t):
 assert len(ledger)<c['new_fit_budget']
 row={'fold':fold,'role':role,'depth':depth,'leaf':leaf,'status':'started'};ledger.append(row)
 (OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2))
 print(f"Model fit {len(ledger)} (maximum 36): fold {fold}, {role}, depth {depth}, leaf {leaf}",flush=True)
 params=dict(c['forest'],max_depth=depth,min_samples_leaf=leaf)
 model=RandomForestRegressor(**params).fit(A,t)
 row['status']='completed';(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2))
 return model

def key(r):
 return (r['MAE'],r['weight'],0 if (r['depth'],r['leaf'])==(3,128) else 1,r['depth'],-r['leaf'])

for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold)
 A,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep)
 assert {f:prep[f] for f in c['anchor_features']}==inner['preprocessing']
 ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 np.testing.assert_allclose(op,base.loc[ov,'pair'],rtol=0,atol=1e-10)
 assert all(X.loc[it,f].count()>=1000 and X.loc[it,f].nunique()>1 for f in X)
 scores=[]
 # Select settings strictly before examining outer target losses.
 for depth in c['grid_depths']:
  for leaf in c['grid_leaves']:
   cached=CACHE/f'fold_{fold}_d{depth}_l{leaf}_inner.joblib'
   if cached.exists():
    model=joblib.load(cached)
    print(f'Reusing retained inner model: fold {fold}, depth {depth}, leaf {leaf}',flush=True)
   else:
    model=fit(fold,'inner',depth,leaf,A,y.loc[it,target])
   joblib.dump(model,OUT/f'fold_{fold}_d{depth}_l{leaf}_inner.joblib')
   fp=np.maximum(model.predict(B),0)
   for w in c['forest_weights']:
    scores.append({'fold':fold,'depth':depth,'leaf':leaf,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 chosen=min(scores,key=key);choices.append(chosen);scores_all.extend(scores)
 (OUT/'inner_scores.json').write_text(json.dumps(scores_all,indent=2))
 depth,leaf,w=chosen['depth'],chosen['leaf'],chosen['weight']
 A,prep=transform(X.loc[ot]);B,_=transform(X.loc[ov],prep)
 assert {f:prep[f] for f in c['anchor_features']}==outer['preprocessing']
 cached_card=json.loads((CACHE/f'fold_{fold}_selected.json').read_text())
 assert prep==cached_card['preprocessing']
 if (depth,leaf)==(cached_card['selection']['depth'],cached_card['selection']['leaf']):
  model=joblib.load(CACHE/f'fold_{fold}_selected.joblib')
  print(f'Reusing retained outer model: fold {fold}',flush=True)
 else:
  model=fit(fold,'outer_selected',depth,leaf,A,y.loc[ot,target])
 joblib.dump(model,OUT/f'fold_{fold}_selected.joblib')
 fp=np.maximum(model.predict(B),0)
 (OUT/f'fold_{fold}_selected.json').write_text(json.dumps({'selection':chosen,'preprocessing':prep,'training_rows':len(ot)},indent=2))
 parts.append(pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'forest':fp,'prediction':(1-w)*op+w*fp,'weight':w},index=ov))
 print(f"Fold {fold} complete: selected depth {depth}, leaf {leaf}, RF weight {w}",flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index)
pred['incumbent']=base[c['benchmark_column']]
pred.to_parquet(OUT/'predictions.parquet')
# Replay every inner configuration and outer selection without fitting.
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold)
 _,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep)
 ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);replay_scores=[]
 for depth in c['grid_depths']:
  for leaf in c['grid_leaves']:
   path=OUT/f'fold_{fold}_d{depth}_l{leaf}_inner.joblib'
   model=joblib.load(path)
   for k,v in dict(c['forest'],max_depth=depth,min_samples_leaf=leaf).items():assert model.get_params()[k]==v
   fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0)
   for w in c['forest_weights']:
    replay_scores.append({'fold':fold,'depth':depth,'leaf':leaf,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 saved=[r for r in scores_all if r['fold']==fold]
 for r,s in zip(replay_scores,saved):assert abs(r['MAE']-s['MAE'])<1e-10
 selected=min(replay_scores,key=key);chosen=choices[fold-1]
 assert all(selected[k]==chosen[k] for k in ['depth','leaf','weight'])
 card=json.loads((OUT/f'fold_{fold}_selected.json').read_text());_,prep=transform(X.loc[ot]);assert prep==card['preprocessing']
 B,_=transform(X.loc[ov],prep);model=joblib.load(OUT/f'fold_{fold}_selected.joblib')
 for k,v in dict(c['forest'],max_depth=chosen['depth'],min_samples_leaf=chosen['leaf']).items():assert model.get_params()[k]==v
 fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0)
 op=pair_prediction(X.loc[ov,c['anchor_features']],outer);w=chosen['weight']
 np.testing.assert_allclose((1-w)*op+w*fp,pred.loc[ov,'prediction'],rtol=0,atol=1e-10)
for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs()
metrics={n:{'MAE':float(loss[n].mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in loss}
for n in metrics:assert abs(metrics[n]['Spearman_IC']-np.corrcoef(pred.actual.rank(),pred[n].rank())[0,1])<1e-12
loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv')
loss.groupby(loss.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int)
 sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy()
 draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)))
 samples=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(samples,[.025,.975]).tolist()})
result={'status':'VERIFIED','experiment':c['version'],'metrics':metrics,'selected_settings':choices,'rows':len(pred),'new_model_fits':len(ledger),'new_selection_steps':4,'cumulative_counted_steps':c['prior_fit_count']+len(ledger)+4,'nominal_intervals':intervals,'interpretation':'Adaptive development experiment; earlier inner MAE selects depth, leaf and blend jointly. No fresh test accessed. No outer-period grid cherry-picking.'}
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2))
(OUT/'verification.json').write_text(json.dumps({'status':'PASS','checks':['strict label-end purges','input hashes','exact rows and target values','train-only preprocessing','all inner scores replayed using individual trees','earlier-data-only settings/weight selection','selected outer models replayed','MAE and Spearman checks'],'rows':len(pred),'new_verification_fits':0},indent=2))
print(json.dumps({'experiment':c['version'],'metrics':metrics,'fits':len(ledger),'verified':True}),flush=True)
