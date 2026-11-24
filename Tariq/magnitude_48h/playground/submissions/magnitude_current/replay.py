"""Replay retained winning forests and liquidation anchors; no fits or output writes."""
from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'run_001'
c=json.loads((OUT/'config.json').read_text());r=json.loads((OUT/'RESULT.json').read_text())
X=pd.concat([pd.read_parquet(ROOT/p['X'],columns=list(dict.fromkeys(c['anchor_features']+list(c.get('additions',{}).values())))) for p in c['data_parts']]).sort_index().replace([np.inf,-np.inf],np.nan)
y=pd.concat([pd.read_parquet(ROOT/p['y']) for p in c['data_parts']]).sort_index()
pred=pd.read_parquet(OUT/'predictions.parquet')
assert X.index.equals(y.index) and X.index.is_unique
assert (y[c['label_end_column']]-y.index==pd.Timedelta(minutes=c['horizon_minutes'])).all()
np.testing.assert_array_equal(pred.actual,y.loc[pred.index,c['target_column']])
keys=[]
for fold,(start,end) in enumerate(c['validation_windows'],1):
 start,end=pd.Timestamp(start),pd.Timestamp(end)
 train=y.index[y[c['label_end_column']]<start];valid=y.index[(y.index>=start)&(y.index<end)];keys.extend(valid)
 assert y.loc[train,c['label_end_column']].max()<valid.min()
 card=json.loads((OUT/f'fold_{fold}_selected.json').read_text());selection=card['selection'];columns=[]
 assert card['training_rows']==len(train)
 for f,a in card['preprocessing'].items():
  t=X.loc[train,f];assert a['median']==t.median()
  filled=t.fillna(a['median']);assert abs(a['center']-filled.mean())<1e-12 and abs(a['scale']-filled.std(ddof=0))<1e-12
  v=X.loc[valid,f];columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
 A=np.column_stack(columns);m=joblib.load(OUT/f'fold_{fold}_selected.joblib')
 models=m if isinstance(m,list) else [m]
 seeds=selection.get('seeds',[c['forest']['random_state']]);assert len(models)==len(seeds)
 forests=[]
 for model,seed in zip(models,seeds):
  for k,v in dict(c['forest'],max_depth=selection['depth'],min_samples_leaf=selection['leaf'],max_samples=selection.get('max_samples',c['forest']['max_samples']),random_state=seed).items():assert model.get_params()[k]==v
  forests.append(np.maximum(np.mean([tree.predict(A) for tree in model.estimators_],axis=0),0))
 forest=np.mean(forests,axis=0)
 pair=json.loads((ROOT/c['outer_pair_profile']/f'fold_{fold}_pair.json').read_text());assert pair['preprocessing']=={f:card['preprocessing'][f] for f in c['anchor_features']}
 pair_columns=[]
 for f,a in pair['preprocessing'].items():
  v=X.loc[valid,f];pair_columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
 anchor=np.maximum(pair['intercept']+np.column_stack(pair_columns)@pair['coefficients'],0);w=selection['weight']
 np.testing.assert_allclose((1-w)*anchor+w*forest,pred.loc[valid,'prediction'],atol=1e-10,rtol=0)
assert pd.DatetimeIndex(keys).equals(pred.index)
mae=float((pred.actual-pred.prediction).abs().mean());ic=float(spearmanr(pred.actual,pred.prediction).statistic)
assert abs(mae-r['metrics']['prediction']['MAE'])<1e-12 and abs(ic-r['metrics']['prediction']['Spearman_IC'])<1e-12
print(json.dumps({'status':'PASS','rows':len(pred),'MAE':mae,'Spearman_IC':ic,'new_fits':0}))
