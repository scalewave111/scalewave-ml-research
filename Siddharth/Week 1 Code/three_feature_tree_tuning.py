"""Three-feature tree-model tuning using nested chronological walk-forward only."""
from pathlib import Path
import sys,json,hashlib,time,warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.boost_deps'))
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
import targeted_feature_extension as m
warnings.filterwarnings('ignore',message='X does not have valid feature names')
OUT=ROOT/'Experiments/three_feature_tree_tuning_v1'
FLOW='F4Z__flow__flow_activity_rz__4h__value'
COLS=m.PAIR+[FLOW]
CONFIGS={
 'rf_base':dict(kind='RF',depth=3,leaf=128,trees=128),
 'rf_regular':dict(kind='RF',depth=5,leaf=256,trees=128),
 'rf_shallow':dict(kind='RF',depth=2,leaf=256,trees=128),
 'cat_shallow':dict(kind='CatBoost',depth=2,trees=400,rate=.03,l2=20),
 'cat_regular':dict(kind='CatBoost',depth=4,trees=400,rate=.025,l2=50),
 'lgb_stump':dict(kind='LightGBM',depth=1,leaves=2,trees=300,leaf=256,rate=.025,l2=20),
 'lgb_shallow':dict(kind='LightGBM',depth=2,leaves=4,trees=400,leaf=256,rate=.025,l2=30),
 'lgb_regular':dict(kind='LightGBM',depth=3,leaves=7,trees=400,leaf=512,rate=.025,l2=60)}
def estimator(cfg):
 if cfg['kind']=='RF':
  return RandomForestRegressor(n_estimators=cfg['trees'],criterion='absolute_error',max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=1.,bootstrap=True,max_samples=4096,n_jobs=8,random_state=48)
 if cfg['kind']=='CatBoost':
  return CatBoostRegressor(loss_function='MAE',iterations=cfg['trees'],depth=cfg['depth'],learning_rate=cfg['rate'],l2_leaf_reg=cfg['l2'],random_seed=48,thread_count=8,verbose=False,allow_writing_files=False,has_time=True)
 return LGBMRegressor(objective='regression_l1',n_estimators=cfg['trees'],max_depth=cfg['depth'],num_leaves=cfg['leaves'],min_child_samples=cfg['leaf'],learning_rate=cfg['rate'],reg_lambda=cfg['l2'],reg_alpha=2.,random_state=48,n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True)
def fit(frame,y,keys,issue,cfgid,path):
 assert y.loc[keys,'label_end'].max()<issue
 prep=m.prep(frame.loc[keys],COLS);model=estimator(CONFIGS[cfgid]);begin=time.monotonic()
 model.fit(m.design(frame.loc[keys],prep),y.loc[keys,'magnitude_log_bps'])
 a=dict(model=model,prep=prep,train_keys=keys,issuance=issue,max_label_end=str(y.loc[keys,'label_end'].max()),config=cfgid)
 joblib.dump(a,path,compress=3)
 print('FIT',issue.date(),cfgid,len(keys),round(time.monotonic()-begin,1),flush=True)
 return a
def select(candidates,baseline):
 good=[c for c in candidates if c['mae']<baseline['mae']-.01 and c['rank_ic']>baseline['rank_ic']+.0001]
 return min(good or candidates,key=lambda c:(c['mae'],-c['rank_ic'],c['config'],c['weight']))
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETE.json').exists():raise RuntimeError('Refusing to overwrite completed run')
 raw,y,dev,_=m.old.load();frame=raw[COLS].replace([np.inf,-np.inf],np.nan)
 reference=pd.read_csv(m.BASE/'report_baseline_predictions.csv',index_col=0,parse_dates=True)
 starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 predictions=[];training=[];validation=[];trials=[];ledger=[];decisions=[]
 for issue in starts:
  inner=issue-pd.DateOffset(months=3);folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  ik=y.index[y.label_end<inner];vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)];ot=y.index[y.label_end<issue]
  end=issue+pd.DateOffset(months=3) if issue!=starts[-1] else dev.max()+pd.Timedelta(minutes=30)
  te=dev[(dev>=issue)&(dev<end)]
  ip=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_pair__expanding.joblib')
  ir=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_rf__expanding.joblib')
  assert ip['train_keys'].equals(ik) and ir['train_keys'].equals(ik)
  pp=m.predict(ip,frame.loc[vk]);rr=m.predict(ir,frame.loc[vk]);target=y.loc[vk,'magnitude_log_bps']
  bw=min([0.,.1,.25,.5],key=lambda w:(m.metric((1-w)*pp+w*rr,target)['mae'],w))
  bv=(1-bw)*pp+bw*rr;bm=m.metric(bv,target)
  cache={};window=[]
  for cfgid in CONFIGS:
   path=folder/(cfgid+'_inner.joblib');a=fit(frame,y,ik,inner,cfgid,path);pred=m.predict(a,frame.loc[vk]);cache[cfgid]=(a,pred)
   ledger.append(dict(path=str(path),phase='inner',issue=str(issue),config=cfgid,train_rows=len(ik)))
   for w in [.1,.25,.5,1.]:
    met=m.metric((1-w)*pp+w*pred,target)
    rec=dict(issue=str(issue),config=cfgid,model=CONFIGS[cfgid]['kind'],weight=w,**met,baseline=bm)
    window.append(rec);trials.append(rec)
   m.dump(OUT/'FIT_LEDGER.json',ledger)
  standalone=select([c for c in window if c['weight']==1.],bm)
  hybrid=select(window,bm)
  opath=m.BASE/'models'/f'{issue:%Y-%m}__report_outer_pair__expanding.joblib';op=joblib.load(opath)
  orf=joblib.load(m.BASE/'models'/f'{issue:%Y-%m}__report_outer_rf__expanding.joblib')
  assert op['train_keys'].equals(ot) and orf['train_keys'].equals(ot)
  outer={}
  for chosen in [standalone,hybrid]:
   name=chosen['config']
   if name not in outer:
    path=folder/(name+'_outer.joblib');outer[name]=fit(frame,y,ot,issue,name,path)
    ledger.append(dict(path=str(path),phase='outer',issue=str(issue),config=name,train_rows=len(ot)))
  sp=m.predict(outer[standalone['config']],frame.loc[te]);w=hybrid['weight']
  hp=(1-w)*m.predict(op,frame.loc[te])+w*m.predict(outer[hybrid['config']],frame.loc[te])
  control=(1-w)*m.predict(op,frame.loc[te])+w*m.predict(orf,frame.loc[te])
  base=(1-bw)*m.predict(op,frame.loc[te])+bw*m.predict(orf,frame.loc[te])
  np.testing.assert_allclose(base,reference.loc[te,'report_hybrid'],rtol=0,atol=1e-7)
  predictions.append(pd.DataFrame(dict(standalone=sp,hybrid=hp,baseline=base,same_weight_two_feature_rf=control,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te))
  training.append(pd.DataFrame(dict(standalone=m.predict(outer[standalone['config']],frame.loc[ot]),hybrid=(1-w)*m.predict(op,frame.loc[ot])+w*m.predict(outer[hybrid['config']],frame.loc[ot]),truth=y.loc[ot,'magnitude_log_bps'],issue=str(issue)),index=ot))
  validation.append(pd.DataFrame(dict(standalone=cache[standalone['config']][1],hybrid=(1-w)*pp+w*cache[hybrid['config']][1],baseline=bv,truth=target,issue=str(issue)),index=vk))
  d=dict(issue=str(issue),inner=str(inner),features=COLS,standalone=standalone,hybrid=hybrid,baseline_weight=bw,pair_artifact=str(opath),standalone_artifact=str(folder/(standalone['config']+'_outer.joblib')),hybrid_artifact=str(folder/(hybrid['config']+'_outer.joblib')),train_rows=len(ot),inner_train_rows=len(ik),validation_rows=len(vk),test_rows=len(te))
  decisions.append(d);m.dump(folder/'selection.json',d);m.dump(OUT/'SELECTIONS.json',decisions);m.dump(OUT/'FIT_LEDGER.json',ledger);m.dump(OUT/'VALIDATION_TRIALS.json',trials)
  print('WINDOW',issue.date(),'standalone',standalone['config'],'hybrid',hybrid['config'],w,flush=True)
 p=pd.concat(predictions);t=pd.concat(training);v=pd.concat(validation);assert p.index.equals(dev) and p.index.is_unique
 for filename,f in [('predictions.csv',p),('training_predictions.csv',t),('validation_predictions.csv',v)]:f.to_csv(OUT/filename,index_label='decision_time')
 scores=[]
 for split,f in [('training',t),('validation',v),('test',p),('last3months',p.loc[p.index>=pd.Timestamp('2024-11-01',tz='UTC')])]:
  for name in ['standalone','hybrid']+(['baseline'] if 'baseline' in f else []):
   scores.append(dict(split=split,model=name,**m.metric(f[name],f.truth),unique_rows=f.index.nunique()))
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 m.dump(OUT/'COMPLETE.json',dict(status='EXECUTED',scores=scores,new_fits=len(ledger),inner_fits=sum(a['phase']=='inner' for a in ledger),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=COLS,scope='48h only; no baseline fallback; retrospective exposed data'))
 (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 print(pd.DataFrame(scores).to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
