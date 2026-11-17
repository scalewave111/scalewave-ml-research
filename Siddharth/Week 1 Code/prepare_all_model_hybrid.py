"""Collect compatible historical forecasts and complete missing current-campaign components."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
import tree_improvement_campaign as c
from threadpoolctl import threadpool_limits
ROOT=c.ROOT;OUT=ROOT/'Experiments/all_model_hybrid_v1'
PACKAGE=ROOT/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
def load(path):return pd.read_csv(path,index_col=0,parse_dates=True)
def main():
 OUT.mkdir(exist_ok=True)
 raw,y,dev,report=c.old.old.load()
 reference=load(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv')
 assert reference.index.equals(dev)
 truth=y.loc[dev,'magnitude_log_bps']
 np.testing.assert_allclose(truth,reference.truth,atol=1e-8,rtol=0)
 matrix=pd.DataFrame({'baseline':reference.report_hybrid},index=dev);registry=[]
 def add(path,columns,group=None,prefix=''):
  frame=load(path);assert frame.index.is_unique
  if 'truth' in frame:
   np.testing.assert_allclose(frame.truth,y.loc[frame.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  for col in columns:
   name=prefix+col;series=frame[col].reindex(dev)
   assert pd.api.types.is_numeric_dtype(series) and np.isfinite(series.dropna()).all()
   matrix[name]=series
   family=group or ('rf' if 'rf' in col.lower() else 'median' if col.startswith('pair') else 'boosting')
   registry.append(dict(name=name,family=family,source=str(path),column=col,source_sha256=digest,rows=int(series.notna().sum()),first=str(series.first_valid_index()),last=str(series.last_valid_index())))
 add(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',['report_pair'],'median','reference:')
 add(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',['report_rf'],'rf','reference:')
 path=PACKAGE/'submissions/member_02/rf_walk_forward_v2/all_candidate_predictions.csv'
 cols=list(load(path))
 for col in cols:
  group='hybrid' if '__blend_' in col else 'median' if col in ['rolling_median','frozen_pair','M0'] else 'rf'
  add(path,[col],group,'yesterday:')
 np.testing.assert_allclose(abs(matrix['yesterday:v1_RF__blend_0.25']-truth).mean(),181.79975329632526,rtol=0,atol=1e-8)
 np.testing.assert_allclose(abs(matrix['yesterday:v1_RF']-truth).mean(),183.74958391311714,rtol=0,atol=1e-8)
 path=ROOT/'Experiments/filtered_walkforward_v1/predictions.csv'
 for col in load(path):
  if col in ['truth','baseline','outer']:continue
  group='hybrid' if col in ['selected','standalone_selected'] else 'median' if col.startswith('pair') else 'boosting'
  add(path,[col],group,'filtered:')
 add(ROOT/'Experiments/residual_incremental_v1/predictions.csv',['selected'],'hybrid','residual:')
 add(ROOT/'Experiments/residual_absolute_vol_v2/predictions.csv',['selected'],'hybrid','residual_vol:')
 add(ROOT/'Experiments/tree_magnitude_filter_v1/48h/predictions.csv',['prediction'],'boosting','eight_test:')
 add(ROOT/'Experiments/tree_magnitude_filter_v1/48h/predictions.csv',['rf_baseline'],'rf','eight_test:')
 add(ROOT/'Experiments/tree_document_methods_v2/predictions.csv',['prediction'],'boosting','document:')
 add(ROOT/'Experiments/tree_improvement_campaign_v3/48h/predictions.csv',['prediction'],'hybrid','today:')
 add(ROOT/'Experiments/nested_ic_rf_v3/predictions.csv',['nested_raw_rf','nested_standalone_rf','nested_selected'],'rf','nested_rf:')
 add(ROOT/'Experiments/nested_boost_48h_v1/predictions.csv',['selected','catboost','lightgbm'],'boosting','nested_boost:')
 r=c.crf.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet')
 parents=list(dict.fromkeys(c.PAIR+[v for v in report if c.old.family(v)!='other']))
 frame=pd.concat([raw,c.engineering(raw,parents),c.state_frame(r,raw.index)],axis=1).replace([np.inf,-np.inf],np.nan).astype('float32')
 # Complete all pre-registered current-campaign recipes, not just winners.
 ledger=[]
 for date in pd.date_range('2024-01-01','2025-01-01',freq='MS',tz='UTC'):
  source=ROOT/'Experiments/tree_improvement_campaign_v3/48h'/date.strftime('%Y%m%d')
  folder=OUT/'components'/date.strftime('%Y%m%d');folder.mkdir(parents=True,exist_ok=True)
  fs=json.loads((source/'feature_selection.json').read_text());cache={}
  for path in source.glob('*.joblib'):
   if '_inner' in path.name:continue
   art=joblib.load(path)
   cache[(art['config'],tuple(sorted(art['columns'])))]=(art,path,'REUSED_ORIGINAL')
  end=date+pd.DateOffset(months=1);test=dev[(dev>=date)&(dev<end)]
  for cfgid,cfg in c.CONFIGS.items():
   if cfgid=='probe':continue
   for pool,columns in [('core',fs['core']),('augmented',fs['augmented'])]:
    key=(cfgid,tuple(sorted(columns)))
    if key in cache:art,path,status=cache[key]
    else:
     slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];path=folder/(slug+'.joblib')
     if path.exists():art=joblib.load(path);status='REUSED_NEW_CACHE'
     else:
      art=c.fit(frame,y,date,cfgid,columns,48);art['engineering_parents']=parents
      joblib.dump(art,path,compress=3);status='NEW'
     cache[key]=(art,path,status)
    assert pd.Timestamp(art['issue'])==date and pd.Timestamp(art['max_label_end'])<date
    name='component:'+cfgid+'__'+pool
    if name not in matrix:matrix[name]=np.nan
    matrix.loc[test,name]=c.predict(art,frame.loc[test])
    family='regime' if cfgid in ['extra_regime','lgb_normal'] else 'rf' if cfg['model']=='rf' else 'extra' if cfg['model']=='extra' else 'boosting'
    ledger.append(dict(issue=str(date),name=name,config=cfgid,columns=columns,artifact=str(path),status=status,fit_count=art['fit_count'],family=family,train_rows=art['train_rows'],max_label_end=art['max_label_end']))
  print(f'Completed component forecasts {date.date()}',flush=True)
 for name in [n for n in matrix if n.startswith('component:')]:
  registry.append(dict(name=name,family=next(a['family'] for a in ledger if a['name']==name),source='components ledger',column=name,rows=int(matrix[name].notna().sum()),first=str(matrix[name].first_valid_index()),last=str(matrix[name].last_valid_index())))
 # Duplicate detection is descriptive only; temporal combiner performs its own past-only deduplication.
 hashes={};duplicates=[]
 for name in matrix:
  values=matrix[name].to_numpy();digest=hashlib.sha256(np.isnan(values).tobytes()+np.nan_to_num(values,nan=-1e100).round(8).tobytes()).hexdigest()
  if digest in hashes:duplicates.append(dict(name=name,same_forecast_as=hashes[digest]))
  else:hashes[digest]=name
 matrix.to_csv(OUT/'component_predictions.csv',index_label='decision_time')
 pd.DataFrame({'truth':truth,'label_end':y.loc[dev,'label_end']}).to_csv(OUT/'targets.csv',index_label='decision_time')
 c.dump(OUT/'REGISTRY.json',registry);c.dump(OUT/'COMPONENT_FITS.json',ledger);c.dump(OUT/'DUPLICATE_DIAGNOSTIC.json',duplicates)
 unique={a['artifact']:a for a in ledger}
 c.dump(OUT/'PREPARATION.json',dict(status='COMPLETE',rows=len(matrix),candidate_forecasts=matrix.shape[1]-1,new_component_fits=sum(a['fit_count'] for a in unique.values() if a['status']!='REUSED_ORIGINAL'),reused_original_artifacts=sum(a['status']=='REUSED_ORIGINAL' for a in unique.values()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256((ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv').read_bytes()).hexdigest()))
 print('Prepared',matrix.shape,'duplicate diagnostics',len(duplicates),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
