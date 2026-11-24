"""Registered direct/residual MAE XGBoost with nested chronological selection."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from feature_experiment import load,splits,transform,anchors,pair_prediction
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'run_001';CACHE=HERE.parent/'magnitude_current/run_001'
c=json.loads((HERE/'boost_config.json').read_text());mode=sys.argv[1];assert mode in ['direct','residual'];c['mode']=mode;c['version']='xgb_mae_'+mode
assert not (OUT/'RESULT.json').exists();X,y=load(c,True);target=c['target_column'];base=pd.read_parquet(CACHE/'predictions.parquet')
np.testing.assert_array_equal(base.actual,y.loc[base.index,target]);assert abs((base.actual-base.prediction).abs().mean()-c['benchmark_mae'])<1e-12
paths=[Path(__file__),HERE/'boost_config.json',HERE/'feature_experiment.py',CACHE/'predictions.parquet',CACHE/'verification.json']+[ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]+[ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'config.json').write_text(json.dumps(c,indent=2));(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2));(OUT/'environment.json').write_text(json.dumps({'xgboost':xgb.__version__,'numpy':np.__version__,'pandas':pd.__version__}))
def combine(pair,boost,strength):
 if mode=='direct':return np.maximum((1-strength)*pair+strength*np.maximum(boost,0),0)
 return np.maximum(pair+strength*boost,0)
def key(r):return (r['MAE'],r['strength'],r['depth'],r['trees'])
ledger=[];parts=[];choices=[];scores_all=[]
def fit(fold,role,depth,trees,A,t):
 assert len(ledger)<c['new_fit_budget'];r={'fold':fold,'role':role,'depth':depth,'trees':trees,'training_rows':len(t),'training_target_sha256':hashlib.sha256(np.asarray(t,dtype=np.float64).tobytes()).hexdigest(),'status':'started'};ledger.append(r)
 (OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2));print(f'{mode}: fit {len(ledger)}/28, fold {fold}, {role}, depth {depth}, trees {trees}',flush=True)
 model=xgb.XGBRegressor(**c['boost'],max_depth=depth,n_estimators=trees).fit(A,t)
 r['status']='completed';(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2));return model
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold);ip=pair_prediction(X.loc[iv,c['anchor_features']],inner);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 np.testing.assert_allclose(op,base.loc[ov,'pair'],rtol=0,atol=1e-10)
 for train,card in [(it,inner),(ot,outer)]:
  _,prep=transform(X.loc[train,c['anchor_features']]);assert prep==card['preprocessing']
  assert all(X.loc[train,f].count()>=1000 and X.loc[train,f].nunique()>1 for f in X)
 A,prep=transform(X.loc[it]);B,_=transform(X.loc[iv],prep)
 truth=y.loc[it,target].to_numpy()
 if mode=='residual':truth=truth-pair_prediction(X.loc[it,c['anchor_features']],inner)
 scores=[]
 for depth in c['boost_depths']:
  for trees in c['boost_trees']:
   model=fit(fold,'inner',depth,trees,A,truth);model.save_model(OUT/f'fold_{fold}_d{depth}_n{trees}_inner.ubj')
   bp=model.predict(B).astype(float)
   for strength in c['strengths']:scores.append({'fold':fold,'depth':depth,'trees':trees,'strength':strength,'MAE':float(np.abs(y.loc[iv,target]-combine(ip,bp,strength)).mean())})
 chosen=min(scores,key=key);choices.append(chosen);scores_all.extend(scores)
 (OUT/f'fold_{fold}_inner.json').write_text(json.dumps({'preprocessing':prep,'training_rows':len(it)},indent=2))
 A,prep=transform(X.loc[ot]);B,_=transform(X.loc[ov],prep);truth=y.loc[ot,target].to_numpy()
 if mode=='residual':truth=truth-pair_prediction(X.loc[ot,c['anchor_features']],outer)
 model=fit(fold,'outer_selected',chosen['depth'],chosen['trees'],A,truth);model.save_model(OUT/f'fold_{fold}_selected.ubj');bp=model.predict(B).astype(float)
 (OUT/f'fold_{fold}_selected.json').write_text(json.dumps({'selection':chosen,'mode':mode,'preprocessing':prep,'training_rows':len(ot)},indent=2))
 parts.append(pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'boost':bp,'prediction':combine(op,bp,chosen['strength']),'strength':chosen['strength'],'incumbent':base.loc[ov,'prediction']},index=ov))
 print(f'{mode}: fold {fold} complete; selected {chosen}',flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index) and np.isfinite(pred[['actual','prediction']]).all().all()
# Reconstruct transforms from training and replay all serialized boosters via native Booster API.
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold);ip=pair_prediction(X.loc[iv,c['anchor_features']],inner)
 _,prep=transform(X.loc[it]);assert prep==json.loads((OUT/f'fold_{fold}_inner.json').read_text())['preprocessing'];B,_=transform(X.loc[iv],prep)
 replayed=[]
 for depth in c['boost_depths']:
  for trees in c['boost_trees']:
   model=xgb.Booster();model.load_model(OUT/f'fold_{fold}_d{depth}_n{trees}_inner.ubj');assert model.num_boosted_rounds()==trees
   assert json.loads(model.save_config())['learner']['objective']['name']=='reg:absoluteerror'
   bp=model.predict(xgb.DMatrix(B)).astype(float)
   for strength in c['strengths']:replayed.append({'fold':fold,'depth':depth,'trees':trees,'strength':strength,'MAE':float(np.abs(y.loc[iv,target]-combine(ip,bp,strength)).mean())})
 saved=[r for r in scores_all if r['fold']==fold]
 for r,s in zip(replayed,saved):assert abs(r['MAE']-s['MAE'])<1e-10
 chosen=min(replayed,key=key);sel=choices[fold-1];assert all(chosen[k]==sel[k] for k in ['depth','trees','strength'])
 _,prep=transform(X.loc[ot]);assert prep==json.loads((OUT/f'fold_{fold}_selected.json').read_text())['preprocessing'];B,_=transform(X.loc[ov],prep)
 model=xgb.Booster();model.load_model(OUT/f'fold_{fold}_selected.ubj');assert model.num_boosted_rounds()==sel['trees'];bp=model.predict(xgb.DMatrix(B)).astype(float);op=pair_prediction(X.loc[ov,c['anchor_features']],outer)
 replay=np.maximum((1-sel['strength'])*op+sel['strength']*np.maximum(bp,0),0) if mode=='direct' else np.maximum(op+sel['strength']*bp,0)
 np.testing.assert_allclose(replay,pred.loc[ov,'prediction'],rtol=0,atol=1e-10)
 for row in [r for r in ledger if r['fold']==fold]:
  train,card=(it,inner) if row['role']=='inner' else (ot,outer)
  truth=y.loc[train,target].to_numpy()
  if mode=='residual':truth=truth-pair_prediction(X.loc[train,c['anchor_features']],card)
  assert row['training_target_sha256']==hashlib.sha256(np.asarray(truth,dtype=np.float64).tobytes()).hexdigest() and row['status']=='completed'
for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
pred.to_parquet(OUT/'predictions.parquet');(OUT/'inner_scores.json').write_text(json.dumps(scores_all,indent=2))
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs();metrics={n:{'MAE':float(loss[n].mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in loss}
for n in metrics:assert abs(metrics[n]['Spearman_IC']-np.corrcoef(pred.actual.rank(),pred[n].rank())[0,1])<1e-12
loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv');loss.groupby(pred.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int);sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy();draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)));samples=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(samples,[.025,.975]).tolist()})
r={'status':'VERIFIED','experiment':c['version'],'mode':mode,'metrics':metrics,'selected_settings':choices,'rows':len(pred),'new_model_fits':len(ledger),'new_selection_steps':4,'nominal_intervals':intervals,'interpretation':'Earlier inner labels select depth, tree count and contribution. Pair model fixed. Residual targets, when used, are training-only in-sample residuals of the training-only pair. No outer labels used for fitting or selection. Adaptive development; fresh test untouched.'}
(OUT/'RESULT.json').write_text(json.dumps(r,indent=2));(OUT/'verification.json').write_text(json.dumps({'status':'PASS','rows':len(pred),'checks':['source hashes','strict label-end purges','training-only preprocessing','training target hashes incl residual construction','native Booster replay of every inner score','earlier-data-only model and strength selection','native Booster outer replay and independent combination formulas','exact scoring rows','MAE and Spearman recomputation'],'new_verification_fits':0},indent=2))
print(json.dumps({'mode':mode,'metrics':metrics,'verified':True}),flush=True)
