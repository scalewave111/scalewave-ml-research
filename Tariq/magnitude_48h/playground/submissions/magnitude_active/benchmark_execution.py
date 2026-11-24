"""Time CPU parallelism on an unchanged full forest; retain only a receipt."""
from pathlib import Path
import hashlib,json,time,os
import joblib,numpy as np
from sklearn.ensemble import RandomForestRegressor
from feature_experiment import load,splits,transform

HERE=Path(__file__).resolve().parent
CURRENT=HERE.parent/'magnitude_current/run_001'
c=json.loads((CURRENT/'config.json').read_text())
prediction_hash=hashlib.sha256((CURRENT/'predictions.parquet').read_bytes()).hexdigest()
X,y=load(c,True)
fold,train,valid,_,_=next(splits(c,y))
card=json.loads((CURRENT/'fold_1_selected.json').read_text())
A,prep=transform(X.loc[train]);B,_=transform(X.loc[valid],prep)
assert prep==card['preprocessing']
old=joblib.load(CURRENT/'fold_1_selected.joblib')[0]
params=old.get_params();assert params['n_estimators']==128 and params['random_state']==c['seeds'][0]
reference=np.mean([t.predict(B) for t in old.estimators_],axis=0)
rows=[]
for workers in [4,8]:
 print(f'Start full 128-tree timing fit: {workers} workers',flush=True)
 start=time.perf_counter()
 model=RandomForestRegressor(**dict(params,n_jobs=workers)).fit(A,y.loc[train,c['target_column']])
 elapsed=time.perf_counter()-start
 for a,b in zip(old.estimators_,model.estimators_):
  for field in ['children_left','children_right','feature','threshold','value','n_node_samples']:
   np.testing.assert_array_equal(getattr(a.tree_,field),getattr(b.tree_,field))
 for a,b in zip(old.estimators_samples_,model.estimators_samples_):np.testing.assert_array_equal(a,b)
 predicted=np.mean([t.predict(B) for t in model.estimators_],axis=0)
 np.testing.assert_array_equal(predicted,reference)
 np.testing.assert_allclose(model.predict(B),reference,atol=1e-10,rtol=0)
 rows.append({'workers':workers,'fit_seconds':elapsed,'tree_and_bootstrap_identity':'PASS','max_canonical_prediction_difference':float(np.max(np.abs(predicted-reference)))})
 print(json.dumps(rows[-1]),flush=True)
 del model
assert hashlib.sha256((CURRENT/'predictions.parquet').read_bytes()).hexdigest()==prediction_hash
selected=8 if rows[1]['fit_seconds']<0.95*rows[0]['fit_seconds'] else 4
receipt={'status':'PASS','purpose':'Execution timing only; no feature or seed selection','logical_cpus':os.cpu_count(),'training_rows':len(train),'prediction_rows':len(valid),'trees_per_fit':128,'seed_for_timing':c['seeds'][0],'production_seeds_unchanged':c['seeds'],'runs':rows,'speedup_4_over_8':rows[0]['fit_seconds']/rows[1]['fit_seconds'],'selected_n_jobs':selected,'new_model_fits':2,'new_model_selection_steps':0,'limitation':'One representative full fit per worker count; runtime varies. No full-batch speedup guarantee.','incumbent_prediction_sha256':prediction_hash,'model_binaries_saved':False}
p=HERE/'execution_benchmark.json'
if p.exists():p.read_text()
p.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
