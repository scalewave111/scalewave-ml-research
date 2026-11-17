"""Add depth 7 to a previously verified grid, reusing compact old-grid scores."""
from pathlib import Path
import json,hashlib,joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from feature_experiment import load,splits,transform,anchors,pair_prediction
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'run_001';CACHE=HERE.parent/'magnitude_current/run_001'
c=json.loads((HERE/'depth7_config.json').read_text());oldconfig=json.loads((CACHE/'config.json').read_text());old_scores=json.loads((CACHE/'inner_scores.json').read_text())
assert json.loads((CACHE/'verification.json').read_text())['status']=='PASS'
for k in ['anchor_features','additions','forest','grid_leaves','forest_weights','validation_windows','inner_validation_months']:assert c[k]==oldconfig[k]
assert oldconfig['grid_depths']==[2,3,5] and c['grid_depths']==[2,3,5,7]
assert not (OUT/'RESULT.json').exists()
X,y=load(c,True);target=c['target_column'];base=pd.read_parquet(CACHE/'predictions.parquet')
np.testing.assert_array_equal(base.actual,y.loc[base.index,target]);assert abs((base.actual-base.prediction).abs().mean()-c['benchmark_mae'])<1e-12
paths=[Path(__file__),HERE/'depth7_config.json',HERE/'feature_experiment.py']+[ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
paths += [p for p in CACHE.iterdir() if p.suffix in ['.json','.joblib','.parquet']]
paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]+[ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2));(OUT/'config.json').write_text(json.dumps(c,indent=2))
ledger=[];parts=[];choices=[];all_scores=[]
def key(r):return (r['MAE'],r['weight'],0 if (r['depth'],r['leaf'])==(3,128) else 1,r['depth'],-r['leaf'])
def fit(fold,role,leaf,A,truth):
 assert len(ledger)<c['new_fit_budget']
 row={'fold':fold,'role':role,'depth':7,'leaf':leaf,'status':'started'};ledger.append(row);(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2))
 print(f'Fit {len(ledger)}/16 maximum: fold {fold}, {role}, depth 7, leaf {leaf}',flush=True)
 model=RandomForestRegressor(**dict(c['forest'],max_depth=7,min_samples_leaf=leaf)).fit(A,truth)
 row['status']='completed';(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2));return model
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold);card=json.loads((CACHE/f'fold_{fold}_selected.json').read_text());saved_choice=card['selection']
 A,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep)
 ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 assert all(X.loc[it,f].count()>=1000 and X.loc[it,f].nunique()>1 for f in X)
 scores=[r for r in old_scores if r['fold']==fold]
 assert len(scores)==36 and {(r['depth'],r['leaf'],r['weight']) for r in scores}=={(d,l,w) for d in [2,3,5] for l in c['grid_leaves'] for w in c['forest_weights']}
 oldbest=min(scores,key=key);assert all(oldbest[k]==saved_choice[k] for k in ['depth','leaf','weight'])
 oldmodel=joblib.load(CACHE/f"fold_{fold}_d{oldbest['depth']}_l{oldbest['leaf']}_inner.joblib")
 fp=np.maximum(oldmodel.predict(B),0)
 for w in c['forest_weights']:
  saved=next(r for r in scores if (r['depth'],r['leaf'],r['weight'])==(oldbest['depth'],oldbest['leaf'],w))
  assert abs(float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())-saved['MAE'])<1e-10
 # Existing scores were fully replay-verified before pruning. Only new depth is trained.
 for leaf in c['grid_leaves']:
  model=fit(fold,'inner',leaf,A,y.loc[it,target]);joblib.dump(model,OUT/f'fold_{fold}_d7_l{leaf}_inner.joblib');fp=np.maximum(model.predict(B),0)
  for w in c['forest_weights']:scores.append({'fold':fold,'depth':7,'leaf':leaf,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 chosen=min(scores,key=key);choices.append(chosen);all_scores.extend(scores);d,l,w=chosen['depth'],chosen['leaf'],chosen['weight']
 A,prep=transform(X.loc[ot]);B,_=transform(X.loc[ov],prep);assert prep==card['preprocessing']
 if d==7:model=fit(fold,'outer_selected',l,A,y.loc[ot,target])
 else:
  assert (d,l)==(oldbest['depth'],oldbest['leaf']);model=joblib.load(CACHE/f'fold_{fold}_selected.joblib')
 joblib.dump(model,OUT/f'fold_{fold}_selected.joblib')
 if d!=7:joblib.dump(oldmodel,OUT/f'fold_{fold}_d{d}_l{l}_inner.joblib')
 fp=np.maximum(model.predict(B),0)
 (OUT/f'fold_{fold}_selected.json').write_text(json.dumps({'selection':chosen,'preprocessing':prep,'training_rows':len(ot)},indent=2))
 parts.append(pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'forest':fp,'prediction':(1-w)*op+w*fp,'weight':w,'incumbent':base.loc[ov,'prediction']},index=ov))
 print(f'Fold {fold}: selected depth {d}, leaf {l}, RF weight {w}',flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index) and np.isfinite(pred[['actual','prediction']]).all().all()
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold);_,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep);ip=pair_prediction(X.loc[iv,c['anchor_features']],inner)
 replayed=[r for r in old_scores if r['fold']==fold]
 for leaf in c['grid_leaves']:
  model=joblib.load(OUT/f'fold_{fold}_d7_l{leaf}_inner.joblib')
  for k,v in dict(c['forest'],max_depth=7,min_samples_leaf=leaf).items():assert model.get_params()[k]==v
  fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0)
  for w in c['forest_weights']:replayed.append({'fold':fold,'depth':7,'leaf':leaf,'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*ip+w*fp)).mean())})
 chosen=min(replayed,key=key);sel=choices[fold-1];assert all(chosen[k]==sel[k] for k in ['depth','leaf','weight'])
 saved=[r for r in all_scores if r['fold']==fold]
 for r,s in zip(replayed,saved):assert abs(r['MAE']-s['MAE'])<1e-10
 card=json.loads((OUT/f'fold_{fold}_selected.json').read_text());_,prep=transform(X.loc[ot]);assert prep==card['preprocessing'];B,_=transform(X.loc[ov],prep)
 model=joblib.load(OUT/f'fold_{fold}_selected.joblib')
 for k,v in dict(c['forest'],max_depth=sel['depth'],min_samples_leaf=sel['leaf']).items():assert model.get_params()[k]==v
 fp=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0);op=pair_prediction(X.loc[ov,c['anchor_features']],outer);w=sel['weight']
 np.testing.assert_allclose((1-w)*op+w*fp,pred.loc[ov,'prediction'],atol=1e-10,rtol=0)
for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
pred.to_parquet(OUT/'predictions.parquet');(OUT/'inner_scores.json').write_text(json.dumps(all_scores,indent=2))
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs();metrics={n:{'MAE':float(loss[n].mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in loss}
for n in metrics:assert abs(metrics[n]['Spearman_IC']-np.corrcoef(pred.actual.rank(),pred[n].rank())[0,1])<1e-12
loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv');loss.groupby(pred.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int);sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy();draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)));samples=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(samples,[.025,.975]).tolist()})
r={'status':'VERIFIED','experiment':c['version'],'metrics':metrics,'selected_settings':choices,'rows':len(pred),'new_model_fits':len(ledger),'new_selection_steps':4,'cumulative_counted_steps':c['prior_fit_count']+len(ledger)+4,'nominal_intervals':intervals,'interpretation':'Only depth 7 added; same inputs, leaves, blend weights, folds. Existing grid scores reused from prior fully verified run; old winners replayed. New depth-7 scores and all selected outer models independently replayed. Adaptive development; fresh test untouched.'}
(OUT/'RESULT.json').write_text(json.dumps(r,indent=2));(OUT/'verification.json').write_text(json.dumps({'status':'PASS','rows':len(pred),'checks':['source hashes','strict label-end purges','previous verified grid score provenance','old winner score replay','new-depth tree-level score replay','earlier-inner-only selection','all selected outer model replays','exact row population','MAE and Spearman checked'],'new_verification_fits':0},indent=2))
print(json.dumps(r,indent=2),flush=True)
