"""Single training-median volatility gate using retained, verified base models."""
from pathlib import Path
import json,hashlib,joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from feature_experiment import load,splits,transform,anchors,pair_prediction
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'run_001';CACHE=HERE.parent/'magnitude_current/run_001'
c=json.loads((HERE/'regime_config.json').read_text());assert not (OUT/'RESULT.json').exists()
X,y=load(c,True);target=c['target_column'];gf=c['gate_feature']
gate=pd.concat([pd.read_parquet(ROOT/p['X'],columns=[gf]) for p in c['data_parts']]).sort_index()[gf].replace([np.inf,-np.inf],np.nan)
assert gate.index.equals(y.index)
base=pd.read_parquet(CACHE/'predictions.parquet');np.testing.assert_array_equal(base.actual,y.loc[base.index,target])
assert abs((base.actual-base.prediction).abs().mean()-c['benchmark_mae'])<1e-12
paths=[Path(__file__),HERE/'regime_config.json',HERE/'feature_experiment.py']
paths += [ROOT/p[k] for p in c['data_parts'] for k in ['X','y']]
paths += [p for p in CACHE.iterdir() if p.suffix in ['.json','.parquet','.joblib']]
paths += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]
paths += [ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'config.json').write_text(json.dumps(c,indent=2));(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
parts=[];rules=[];diagnostics=[]
for fold,ot,ov,it,iv in splits(c,y):
 inner,outer=anchors(c,fold)
 card=json.loads((CACHE/f'fold_{fold}_selected.json').read_text());sel=card['selection']
 forest_predictions={};pair_predictions={}
 for role,train,valid in [('inner',it,iv),('outer',ot,ov)]:
  assert all(X.loc[train,f].count()>=1000 for f in X)
  _,prep=transform(X.loc[train]);A,_=transform(X.loc[valid],prep)
  if role=='outer':assert prep==card['preprocessing']
  modelpath=CACHE/f"fold_{fold}_d{sel['depth']}_l{sel['leaf']}_inner.joblib" if role=='inner' else CACHE/f'fold_{fold}_selected.joblib'
  model=joblib.load(modelpath)
  for k,v in dict(c['forest'],max_depth=sel['depth'],min_samples_leaf=sel['leaf']).items():assert model.get_params()[k]==v
  prediction=np.maximum(model.predict(A),0)
  np.testing.assert_allclose(prediction,np.maximum(np.mean([t.predict(A) for t in model.estimators_],axis=0),0),atol=1e-10,rtol=0)
  forest_predictions[role]=prediction
  pair_predictions[role]=pair_prediction(X.loc[valid,c['anchor_features']],inner if role=='inner' else outer)
 ip,fp=pair_predictions['inner'],forest_predictions['inner'];truth=y.loc[iv,target].to_numpy()
 scores=[{'weight':w,'MAE':float(np.abs(truth-((1-w)*ip+w*fp)).mean())} for w in c['forest_weights']]
 shared=min(scores,key=lambda r:(r['MAE'],r['weight']))['weight'];assert shared==sel['weight']
 assert gate.loc[it].count()>=1000 and gate.loc[ot].count()>=1000
 ti=float(gate.loc[it].quantile(c['gate_quantile']));to=float(gate.loc[ot].quantile(c['gate_quantile']))
 regime=np.where(gate.loc[iv].isna(),'missing',np.where(gate.loc[iv]<=ti,'low','high'))
 weights={};per_regime={}
 for name in ['low','high']:
  mask=regime==name;n=int(mask.sum())
  sc=[{'weight':w,'MAE':float(np.abs(truth[mask]-((1-w)*ip[mask]+w*fp[mask])).mean())} for w in c['forest_weights']] if n else []
  weights[name]=min(sc,key=lambda r:(r['MAE'],r['weight']))['weight'] if n>=c['minimum_inner_regime_rows'] else shared
  per_regime[name]={'rows':n,'scores':sc,'eligible':n>=c['minimum_inner_regime_rows']}
 weights['missing']=shared
 wi=np.array([weights[x] for x in regime]);gated_inner=float(np.abs(truth-((1-wi)*ip+wi*fp)).mean())
 shared_inner=float(np.abs(truth-((1-shared)*ip+shared*fp)).mean())
 # Prefer simpler shared blend on a tie or if a gate provides no inner improvement.
 enabled=gated_inner<shared_inner-1e-12
 if not enabled:weights={name:shared for name in ['low','high','missing']}
 out_regime=np.where(gate.loc[ov].isna(),'missing',np.where(gate.loc[ov]<=to,'low','high'))
 wo=np.array([weights[x] for x in out_regime])
 op,of=pair_predictions['outer'],forest_predictions['outer']
 np.testing.assert_allclose((1-shared)*op+shared*of,base.loc[ov,'prediction'],atol=1e-10,rtol=0)
 forecast=(1-wo)*op+wo*of
 # Independent row-wise gate replay and medians from sorted training observations.
 assert abs(ti-np.median(gate.loc[it].dropna().to_numpy()))<1e-12
 assert abs(to-np.median(gate.loc[ot].dropna().to_numpy()))<1e-12
 independent=[]
 for value,pair,forest in zip(gate.loc[ov],op,of):
  name='missing' if pd.isna(value) else ('low' if value<=to else 'high')
  w=weights[name];independent.append((1-w)*pair+w*forest)
 np.testing.assert_allclose(independent,forecast,rtol=0,atol=1e-10)
 rule={'fold':fold,'inner_training_threshold':ti,'outer_training_threshold':to,'enabled':enabled,'shared_weight':shared,'weights':weights,'shared_inner_MAE':shared_inner,'gated_inner_MAE':gated_inner,'inner_regimes':per_regime,'inner_missing_rows':int((regime=='missing').sum()),'selected_RF':sel}
 rules.append(rule)
 frame=pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':op,'forest':of,'prediction':forecast,'weight':wo,'regime':out_regime,'incumbent':base.loc[ov,'prediction']},index=ov);parts.append(frame)
 for name,group in frame.groupby('regime'):
  diagnostics.append({'fold':fold,'regime':name,'rows':len(group),'MAE':float((group.actual-group.prediction).abs().mean()),'incumbent_MAE':float((group.actual-group.incumbent).abs().mean())})
 print(f'Fold {fold}: threshold={to:.6f}; gate={enabled}; weights={weights}',flush=True)
pred=pd.concat(parts);assert pred.index.equals(base.index) and pred.index.is_unique
assert np.isfinite(pred[['actual','prediction','incumbent']]).all().all()
metrics={n:{'MAE':float((pred.actual-pred[n]).abs().mean()),'Spearman_IC':float(spearmanr(pred.actual,pred[n]).statistic)} for n in ['incumbent','prediction']}
for n in metrics:assert abs(metrics[n]['Spearman_IC']-np.corrcoef(pred.actual.rank(),pred[n].rank())[0,1])<1e-12
for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
pred.to_parquet(OUT/'predictions.parquet');(OUT/'gate_rules.json').write_text(json.dumps(rules,indent=2))
pd.DataFrame(diagnostics).to_csv(OUT/'regime_metrics.csv',index=False)
loss=pred[['incumbent','prediction']].sub(pred.actual,axis=0).abs();loss.groupby(pred.fold).mean().to_csv(OUT/'fold_MAE.csv');loss.groupby(pred.index.strftime('%Y-%m')).mean().to_csv(OUT/'monthly_MAE.csv')
delta=loss.prediction-loss.incumbent;intervals=[]
for days in [7,14]:
 group=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int);sums=delta.groupby(group).sum().to_numpy();sizes=delta.groupby(group).size().to_numpy()
 draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)));samples=sums[draws].sum(1)/sizes[draws].sum(1)
 intervals.append({'block_days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(samples,[.025,.975]).tolist()})
r={'status':'VERIFIED','experiment':c['version'],'metrics':metrics,'gate_feature':gf,'threshold_rule':'Training-only median, reestimated on outer training for deployment','rules':rules,'rows':len(pred),'new_model_fits':0,'new_selection_steps':4,'cumulative_counted_steps':506,'nominal_intervals':intervals,'interpretation':'Single preregistered volatility gate; base RF settings reused from earlier-inner-selected incumbent. Weights selected only on earlier inner labels. Missing volatility falls back to shared blend; fewer than 200 regime rows also falls back. Adaptive development only; fresh data not accessed.'}
(OUT/'RESULT.json').write_text(json.dumps(r,indent=2))
(OUT/'verification.json').write_text(json.dumps({'status':'PASS','rows':len(pred),'checks':['strict temporal purges','source hashes','cached model parameters and tree-level replay','shared blend exactly reproduces incumbent','training-only median thresholds independently checked','earlier-validation per-regime weight selection','independent row-wise gate replay','exact scoring rows and targets','MAE and Spearman recomputation'],'new_verification_fits':0},indent=2))
print(json.dumps(metrics,indent=2),flush=True)
