"""No-fit audit and inference utilities for filtered walk-forward."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import filtered_walkforward as fw
import numpy as np
import pandas as pd
import joblib
def load_models(out):
 return {p.stem:joblib.load(p) for p in (out/'models').glob('*.joblib')}
def base_prediction(out,models,raw,y,start,keys):
 card=json.loads((out/'baseline'/f'{start:%Y-%m}.json').read_text())
 pair=models[f'{start:%Y-%m}__pair_exp'];rf=models[f'{start:%Y-%m}__reference_rf'];w=card['rf_weight']
 return (1-w)*fw.prediction(pair,raw,y,keys)+w*fw.prediction(rf,raw,y,keys)
def selected_prediction(out,models,raw,y,start,keys):
 s=json.loads((out/'outer'/f'{start:%Y-%m}_selection.json').read_text())['selected']
 baseline=base_prediction(out,models,raw,y,start,keys)
 if s['model'] is None:return baseline
 return (1-s['weight'])*baseline+s['weight']*fw.prediction(models[f'{start:%Y-%m}__'+s['model']],raw,y,keys)
def audit(out):
 c=json.loads((out/'CONTRACT.json').read_text());raw,y,dev,_=fw.nb.load_data(c['horizon_hours'],c['prices'])
 models=load_models(out);p=fw.nb.old.read_csv(out/'predictions.csv')
 assert p.index.equals(dev) and p.index.is_unique
 np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-10)
 ledger=json.loads((out/'fit_ledger.json').read_text())
 for record in ledger:
  assert record['status'] in ['PASS','REUSED'];a=models[record['id']];start=a['issuance']
  window='exp' if a.get('reuse_nb') else a['config']['window']
  tk=fw.keys_for(y,start,window)
  assert a['train_keys'].equals(tk) and y.loc[tk,'label_end'].max()<start
  if record['status']=='REUSED':
   assert fw.hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()==record['sha256']
  frame=raw
  if not a.get('reuse_nb'):
   if a['parents']:frame=frame.join(fw.engineered(raw,a['parents']))
   if a['history']:frame=frame.join(fw.nb.old.histories(y,raw.index))
  z=frame.loc[tk,a['prep']['columns']]
  np.testing.assert_allclose(a['prep']['median'],z.median(),rtol=0,atol=1e-12)
  if not a.get('reuse_nb') and a['prep']['transform']=='robust':
   np.testing.assert_allclose(a['prep']['lo'],z.quantile(.01),rtol=0,atol=1e-12)
   np.testing.assert_allclose(a['prep']['hi'],z.quantile(.99),rtol=0,atol=1e-12)
   np.testing.assert_allclose(a['prep']['iqr'],(z.quantile(.75)-z.quantile(.25)).clip(lower=1e-4),rtol=0,atol=1e-12)
  if not a.get('reuse_nb') and a['config']['pool']!='pair':
   screen=pd.read_csv(out/'filters'/f'{start:%Y-%m}__{window}.csv').set_index('feature')
   for feature in a['prep']['columns']:
    if feature.startswith('F4Z__') and feature not in fw.nb.PAIR:
     assert bool(screen.loc[feature,'passed'])
     rho,ks,lift=fw.effect(raw.loc[tk,feature].to_numpy(),y.loc[tk,'magnitude_log_bps'].to_numpy())
     np.testing.assert_allclose([rho,ks,lift],screen.loc[feature,['rank_ic','ks_stat','magnitude_lift']].astype(float).to_numpy(),atol=1e-10,rtol=0)
 maximum=0.
 for path in sorted((out/'outer').glob('*_selection.json')):
  s=json.loads(path.read_text());start=pd.Timestamp(s['outer'])
  val=fw.nb.old.read_csv(out/'inner'/f'{start:%Y-%m}.csv')
  assert (y.loc[val.index,'label_end']<start).all()
  chosen=fw.select(val)
  assert chosen['selected']['recipe']==s['selected']['recipe'] and chosen['best_standalone']['recipe']==s['best_standalone']['recipe']
  keys=p.index[p.outer==str(start)]
  got=selected_prediction(out,models,raw,y,start,keys)
  maximum=max(maximum,float(np.max(abs(got-p.loc[keys,'selected']))))
  np.testing.assert_allclose(got,p.loc[keys,'selected'],atol=1e-8,rtol=0)
  # Independently reconstruct every inner forecast used by the selector.
  for inner in [start-pd.DateOffset(months=6),start-pd.DateOffset(months=3)]:
   ik=val.index[(val.index>=inner)&(val.index<inner+pd.DateOffset(months=3))]
   np.testing.assert_allclose(base_prediction(out,models,raw,y,inner,ik),val.loc[ik,'baseline'],atol=1e-8,rtol=0)
   for cfg in fw.CONFIGS:
    np.testing.assert_allclose(fw.prediction(models[f'{inner:%Y-%m}__'+cfg['id']],raw,y,ik),val.loc[ik,cfg['id']],atol=1e-8,rtol=0)
 # Verify baseline inner weight selection.
 for cardpath in (out/'baseline').glob('*.json'):
  card=json.loads(cardpath.read_text());start=pd.Timestamp(card['start']);inner=pd.Timestamp(card['inner_start'])
  ik=y.index[(y.index>=inner)&(y.index<start)&(y.label_end<start)]
  pp=fw.prediction(models[f'{inner:%Y-%m}__pair_exp'],raw,y,ik)
  pr=fw.prediction(models[f'{inner:%Y-%m}__reference_rf'],raw,y,ik);truth=y.loc[ik,'magnitude_log_bps']
  w=min([0.,.1,.25,.5],key=lambda w:(float(np.mean(abs((1-w)*pp+w*pr-truth))),w))
  assert w==card['rf_weight']
 for split,file in [('train','training_predictions.csv'),('validation_selected','validation_predictions.csv'),('test','predictions.csv')]:
  f=fw.nb.old.read_csv(out/file)
  for _,r in pd.read_csv(out/'scores.csv').query('split==@split').iterrows():
   m=fw.nb.metrics(f[r.model],f.truth)
   assert abs(m['mae']-r.mae)<1e-8
   if m['rank_ic'] is not None:assert abs(m['rank_ic']-r.rank_ic)<1e-10
 for path,digest in json.loads((out/'source_hashes.json').read_text()).items():
  assert fw.hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 manifest=json.loads((fw.nb.old.PACKAGE/'MANIFEST.json').read_text())
 assert all(fw.hashlib.sha256((fw.nb.old.PACKAGE/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
 fw.nb.dump(out/'VERIFICATION.json',dict(status='PASS',all_models=len(models),unique_test_rows=len(p),
  all_inner_and_selected_outer_replay=True,all_training_populations=True,training_transforms=True,
  selected_feature_filter_statistics=True,joint_selection_gate=True,reference_weight_selection=True,score_arithmetic=True,
  original_manifest_hashes=True,max_replay_error=maximum,new_fits=0))
 print('AUDIT PASS',len(models),'models;',len(p),'unique test rows',flush=True)
if __name__=='__main__':audit(Path(sys.argv[1]))
