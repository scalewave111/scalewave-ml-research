"""Controlled baseline plus one feature, with unchanged model recipe and walk-forward splits."""
from pathlib import Path
import json,time,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/baseline_single_feature_v6'
FUNDING='F4Z__funding__funding_persistence_percentile__15m__value'
CANDIDATES={
 'rv4h':'PRICE__realized_vol_4h',
 'rv24h':'PRICE__realized_vol_24h',
 'rv7d':'PRICE__realized_vol_168h',
 'past48h_move':'PRICE__absolute_net_move_48h',
 'vol_ratio':'PRICE__vol_ratio_4h_24h',
 'funding_persistence':FUNDING}
WEIGHTS=[0.,.1,.25,.5]

def price_features(r,index):
 f=q.cr.features(r,index)
 f['PRICE__absolute_net_move_48h']=r.rolling('48h',closed='left',min_periods=576).sum().abs().reindex(index)
 return f

def prepare(x,keys,cols):
 a=x.loc[keys,cols];med=a.median();filled=a.fillna(med)
 assert med.notna().all()
 return dict(columns=cols,median=med,center=filled.mean(),scale=filled.std(ddof=0).replace(0,1.))

def fit(x,y,keys,issue,cols,kind,path,prototype):
 expected=prepare(x,keys,cols)
 if path.exists():
  a=joblib.load(path)
  assert a['train_keys'].equals(keys) and a['issuance']==issue and a['prep']['columns']==cols
  return a
 assert y.loc[keys,'label_end'].max()<issue
 model=clone(prototype)
 if kind=='rf':model.set_params(n_jobs=2)
 start=time.monotonic();model.fit(q.b.design(x.loc[keys],expected),y.loc[keys,'magnitude_log_bps'])
 a=dict(model=model,prep=expected,train_keys=keys,issuance=issue,kind=kind,seconds=time.monotonic()-start)
 joblib.dump(a,path,compress=3)
 return a

def choose_weight(pp,rp,truth):
 return min(WEIGHTS,key=lambda w:(float(abs((1-w)*pp+w*rp-truth).mean()),w))

def selection(trials,base_mae):
 good=[r for r in trials if r['mae']<=base_mae-.1 and r['monthly_wins']>=2]
 return min(good,key=lambda r:(r['mae'],r['name'])) if good else None

def run():
 begin=time.monotonic()
 if (OUT/'COMPLETE.json').exists():print((OUT/'COMPLETE.json').read_text(),flush=True);return
 OUT.mkdir(exist_ok=True)
 source_sha=q.sha(__file__)
 if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())['source_sha256']==source_sha
 raw,y,dev,_=q.b.old.load();x=raw[q.b.PAIR+[FUNDING]].copy();del raw
 r=q.cr.load_returns(q.ROOT/'Dataset/canonical_returns_5m.parquet')
 checks=q.cr.validate(r,y);f=price_features(r,y.index)
 cuts=[y.index[len(y)//3],y.index[2*len(y)//3]]
 for cut in cuts:
  idx=y.index[y.index<=cut]
  pd.testing.assert_frame_equal(price_features(r.loc[:cut],idx),f.loc[idx])
  changed=r.copy();changed.loc[changed.index>=cut]+=100000
  np.testing.assert_allclose(price_features(changed,pd.DatetimeIndex([cut])),f.loc[[cut]],rtol=0,atol=0,equal_nan=True)
 for origin in y.index[::3000]:
  past=r[(r.index>=origin-pd.Timedelta(hours=48))&(r.index<origin)]
  if len(past)==576 and past.notna().all():assert abs(abs(float(past.sum()))-f.loc[origin,'PRICE__absolute_net_move_48h'])<1e-8
 x=pd.concat([x,f[[c for c in CANDIDATES.values() if c!=FUNDING]].astype('float32')],axis=1)
 assert x.index.equals(y.index) and x.index.is_unique
 ref=pd.read_csv(q.b.BASE/'report_baseline_predictions.csv',index_col=0,parse_dates=True)
 assert ref.index.equals(dev) and len(dev)==18481
 np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 np.testing.assert_allclose(y.magnitude_log_bps,abs(y.signed_return_log_bps),atol=1e-8,rtol=0)
 template_paths={kind:q.b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib' for kind in ['pair','rf']}
 prototypes={kind:joblib.load(p)['model'] for kind,p in template_paths.items()}
 contract=dict(target='48-hour absolute endpoint log return in bps',original_features=q.b.PAIR,candidates=CANDIDATES,
  feature_counts='Each challenger has exactly the original two source features plus one candidate; original missing-value preprocessing retained',
  model_recipe='Clone original median-regression and RF parameters; extend BOTH components by one feature; no hyperparameter tuning',
  weights=WEIGHTS,weight_selection='Same minimum prior-quarter validation MAE rule as baseline',
  outer_starts=list(q.STARTS),training='Same expanding rows; label_end strictly before issuance',
  validation='Same preceding quarter; validation label_end strictly before outer issuance',
  primary='Per-window feature selected only on validation, requiring >=0.1 bps pooled gain and wins in >=2 months; otherwise baseline',
  ablations='All six fixed-feature strategies preregistered; each uses its own validation-chosen blend weight, then reports all outer windows. Do not choose deployment model using their test ranking.',
  price_availability='Canonical five-minute returns indexed at endpoint; only endpoints strictly before forecast origin',
  stopping='One frozen experiment; no test-driven retries; no time cap',source_sha256=source_sha,
  limitation='Previously examined historical evaluation; multiple candidates; not untouched holdout confirmation')
 if not (OUT/'CONTRACT.json').exists():
  q.dump(OUT/'CONTRACT.json',contract);(OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
  fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
  docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
  inputs=[q.b.old.PACKAGE/'data'/name for name in ['X_train.parquet','y_train.parquet','X_development.parquet','y_development.parquet']]+[q.ROOT/'Dataset/canonical_returns_5m.parquet',q.b.BASE/'report_baseline_predictions.csv']
  q.dump(OUT/'MANIFEST.json',dict(source_sha256=source_sha,input_hashes={str(p):q.sha(p) for p in inputs},
   baseline_template_hashes={kind:q.sha(p) for kind,p in template_paths.items()},instruction_hashes={str(p):q.sha(p) for p in docs},
   dependency_hashes={p:q.sha(HERE/p) for p in ['lstm_walk_forward.py','targeted_feature_extension.py','canonical_return_features.py','nested_ic_rf.py']},
   versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','duckdb']},canonical_checks=checks))
  joblib.dump((x,y,dev,ref),OUT/'prepared.joblib',compress=3)
 q.dump(OUT/'FEATURE_AUDIT.json',dict(canonical_checks=checks,new48h_feature_prefix_and_future_perturbation=True,past48h_manual_checks=True,
  missing_counts=x.isna().sum().to_dict(),unique_counts=x.nunique().to_dict()))
 pieces=[];validations=[];decisions=[]
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  inner,ik,vk,ot,te=q.windows(y,dev,issue)
  _,bv,_,bw=q.base_window(x,y,dev,ref,issue);truth=y.loc[vk,'magnitude_log_bps'].to_numpy();bm=q.b.metric(bv,truth)
  trials=[];vp={}
  for name,col in CANDIDATES.items():
   cols=q.b.PAIR+[col]
   ap=fit(x,y,ik,inner,cols,'pair',folder/f'{name}_pair_inner.joblib',prototypes['pair'])
   ar=fit(x,y,ik,inner,cols,'rf',folder/f'{name}_rf_inner.joblib',prototypes['rf'])
   pp=q.b.predict(ap,x.loc[vk]);rp=q.b.predict(ar,x.loc[vk]);w=choose_weight(pp,rp,truth)
   pred=(1-w)*pp+w*rp;vp[name]=pred
   gains=[]
   for month in sorted(set(vk.strftime('%Y-%m'))):
    mask=vk.strftime('%Y-%m')==month
    gains.append(float(abs(bv[mask]-truth[mask]).mean()-abs(pred[mask]-truth[mask]).mean()))
   row=dict(name=name,feature=col,weight=w,baseline_weight=bw,monthly_gains=gains,monthly_wins=sum(g>.01 for g in gains),
    gain_bps=float(bm['mae']-abs(pred-truth).mean()),**q.b.metric(pred,truth))
   trials.append(row);q.dump(folder/'VALIDATION_TRIALS.json',trials)
   q.dump(OUT/'STATUS.json',dict(stage='validation',issue=str(issue),completed_candidates=len(trials),total_candidates=6))
   print('VALIDATED',issue.date(),name,'MAE',round(row['mae'],4),'gain',round(row['gain_bps'],4),flush=True)
  chosen=selection(trials,bm['mae']);selected=chosen['name'] if chosen else 'baseline'
  decision=dict(issue=str(issue),selected=selected,baseline_weight=bw,baseline_validation_mae=bm['mae'],candidates=trials,
   train_rows=len(ot),validation_rows=len(vk),test_rows=len(te))
  q.dump(folder/'selection.json',decision) # Freeze choices before outer predictions.
  data=dict(baseline=ref.loc[te,'report_hybrid'].to_numpy())
  for row in trials:
   name=row['name'];cols=q.b.PAIR+[row['feature']]
   ap=fit(x,y,ot,issue,cols,'pair',folder/f'{name}_pair_outer.joblib',prototypes['pair'])
   ar=fit(x,y,ot,issue,cols,'rf',folder/f'{name}_rf_outer.joblib',prototypes['rf'])
   data[name]=(1-row['weight'])*q.b.predict(ap,x.loc[te])+row['weight']*q.b.predict(ar,x.loc[te])
  data['selected']=data[selected].copy();data['issue']=str(issue)
  pieces.append(pd.DataFrame(data,index=te))
  val=dict(baseline=bv,**vp);val['selected']=val[selected].copy();val['truth']=truth;val['issue']=str(issue)
  validations.append(pd.DataFrame(val,index=vk));decisions.append(decision);q.dump(OUT/'SELECTIONS.json',decisions)
  print('WINDOW FROZEN',issue.date(),selected,'elapsed',round(time.monotonic()-begin,1),flush=True)
 p=pd.concat(pieces);assert p.index.equals(dev);p['truth']=y.loc[dev,'magnitude_log_bps'];v=pd.concat(validations)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
 names=['baseline','selected']+list(CANDIDATES)
 scores=[dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 ws=[dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]
 pd.DataFrame(ws).to_csv(OUT/'window_scores.csv',index=False)
 saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True);assert saved.index.equals(dev)
 np.testing.assert_allclose(saved.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 np.testing.assert_allclose(saved.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
 count=0;max_error=0.
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');s=json.loads((folder/'selection.json').read_text())
  inner,ik,vk,ot,te=q.windows(y,dev,issue);_,bv,_,bw=q.base_window(x,y,dev,ref,issue)
  truth=y.loc[vk,'magnitude_log_bps'].to_numpy();reconstructed=[]
  for name,col in CANDIDATES.items():
   predictions={}
   for phase,keys,cut,pk in [('inner',ik,inner,vk),('outer',ot,issue,te)]:
    for kind in ['pair','rf']:
     a=joblib.load(folder/f'{name}_{kind}_{phase}.joblib')
     assert a['train_keys'].equals(keys) and a['issuance']==cut and y.loc[keys,'label_end'].max()<cut
     assert a['prep']['columns']==q.b.PAIR+[col]
     expected=prepare(x,keys,q.b.PAIR+[col])
     for field in ['median','center','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
     expected_params=prototypes[kind].get_params();actual_params=a['model'].get_params()
     for key,value in expected_params.items():
      if key!='n_jobs':assert actual_params[key]==value
     predictions[phase,kind]=q.b.predict(a,x.loc[pk]);count+=1
   w=choose_weight(predictions['inner','pair'],predictions['inner','rf'],truth)
   row=next(a for a in s['candidates'] if a['name']==name);assert w==row['weight']
   vp=(1-w)*predictions['inner','pair']+w*predictions['inner','rf']
   np.testing.assert_allclose(q.b.metric(vp,truth)['mae'],row['mae'],atol=1e-10,rtol=0)
   gains=[]
   for month in sorted(set(vk.strftime('%Y-%m'))):
    mask=vk.strftime('%Y-%m')==month;gains.append(float(abs(bv[mask]-truth[mask]).mean()-abs(vp[mask]-truth[mask]).mean()))
   np.testing.assert_allclose(gains,row['monthly_gains'],atol=1e-10,rtol=0)
   assert sum(g>.01 for g in gains)==row['monthly_wins'];reconstructed.append(row)
   rp=(1-w)*predictions['outer','pair']+w*predictions['outer','rf']
   max_error=max(max_error,float(abs(rp-saved.loc[te,name]).max()))
   np.testing.assert_allclose(rp,saved.loc[te,name],atol=1e-8,rtol=0)
  chosen=selection(reconstructed,float(abs(bv-truth).mean()))
  assert s['selected']==(chosen['name'] if chosen else 'baseline')
  np.testing.assert_allclose(saved.loc[te,'selected'],saved.loc[te,s['selected']],atol=0,rtol=0)
 assert count==96
 audit=dict(status='PASS',model_fits=count,rows=len(p),exactly_one_added_source_feature=True,unchanged_model_parameters=True,
  canonical_feature_causality=True,strict_label_purges=True,training_only_preprocessing=True,baseline_reproduced=True,
  validation_selection_reproduced=True,max_replay_error=max_error)
 q.dump(OUT/'AUDIT.json',audit)
 delta=abs(p.selected-p.truth)-abs(p.baseline-p.truth)
 groups=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
 blocks=pd.DataFrame(dict(delta=delta.to_numpy(),group=groups)).groupby('group').delta.agg(['sum','count'])
 ids=np.random.default_rng(2026).integers(0,len(blocks),(2000,len(blocks)));boot=blocks['sum'].to_numpy()[ids].sum(1)/blocks['count'].to_numpy()[ids].sum(1)
 test=pd.DataFrame(scores).query("split=='test'").set_index('model').to_dict('index')
 result=dict(status='EXECUTED_AND_AUDITED',audit=audit,test_scores=test,primary_beats_baseline=bool(delta.mean()<0),
  primary_mae_difference=float(delta.mean()),weekly_block_bootstrap_95pct_ci=np.quantile(boot,[.025,.975]).tolist(),
  fixed_feature_point_estimate_wins=[n for n in CANDIDATES if test[n]['mae']<test['baseline']['mae']],
  elapsed_this_invocation_seconds=time.monotonic()-begin,limitation=contract['limitation'])
 q.dump(OUT/'COMPLETE.json',result);q.dump(OUT/'STATUS.json',dict(stage='complete',**result))
 (OUT/'RESULTS.txt').write_text('Baseline plus exactly one feature: controlled, preregistered comparison.\n\n'+pd.DataFrame(scores).to_string(index=False)+'\n\n'+json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):run()

