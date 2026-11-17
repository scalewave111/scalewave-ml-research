"""Independent replay and reporting for the three-feature tuned experiment. No fits."""
import json,hashlib,html
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import three_feature_tree_tuning as t
m=t.m
def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')
def main():
 out=t.OUT;done=json.loads((out/'COMPLETE.json').read_text());protocol=json.loads((out/'PROTOCOL.json').read_text())
 digest=hashlib.sha256((t.ROOT/'Code/three_feature_tree_tuning.py').read_bytes()).hexdigest()
 assert done['source_sha256']==protocol['source_sha256']==digest and protocol['models']==t.CONFIGS and protocol['features']==t.COLS
 raw,y,dev,_=m.old.load();frame=raw[t.COLS].replace([np.inf,-np.inf],np.nan)
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
 training=pd.read_csv(out/'training_predictions.csv',index_col=0,parse_dates=True)
 validation=pd.read_csv(out/'validation_predictions.csv',index_col=0,parse_dates=True)
 assert p.index.equals(dev) and p.index.is_unique
 ref=pd.read_csv(m.BASE/'report_baseline_predictions.csv',index_col=0,parse_dates=True)
 np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 np.testing.assert_allclose(p.baseline,ref.report_hybrid,rtol=0,atol=1e-7)
 ds=json.loads((out/'SELECTIONS.json').read_text());trials=json.loads((out/'VALIDATION_TRIALS.json').read_text());ledger=json.loads((out/'FIT_LEDGER.json').read_text())
 assert len(ledger)==done['new_fits']<=40 and sum(a['phase']=='inner' for a in ledger)==32
 checked=0;trial_count=0;windows=[];importance=[]
 for d in ds:
  issue=pd.Timestamp(d['issue']);inner=pd.Timestamp(d['inner'])
  ik=y.index[y.label_end<inner];ot=y.index[y.label_end<issue];vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)]
  te=p.index[p.issue==d['issue']]
  ip=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_pair__expanding.joblib')
  ir=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_rf__expanding.joblib')
  assert ip['train_keys'].equals(ik) and ir['train_keys'].equals(ik)
  pp=m.predict(ip,frame.loc[vk]);rr=m.predict(ir,frame.loc[vk]);truth=y.loc[vk,'magnitude_log_bps']
  bw=min([0.,.1,.25,.5],key=lambda w:(m.metric((1-w)*pp+w*rr,truth)['mae'],w))
  assert d['baseline_weight']==bw;bm=m.metric((1-bw)*pp+bw*rr,truth)
  cache={}
  for rec in [a for a in ledger if a['issue']==d['issue']]:
   a=joblib.load(rec['path']);keys=ik if rec['phase']=='inner' else ot;start=inner if rec['phase']=='inner' else issue
   assert a['train_keys'].equals(keys) and pd.Timestamp(a['issuance'])==start and pd.Timestamp(a['max_label_end'])<start
   assert a['prep']['columns']==t.COLS and a['config']==rec['config']
   expected=m.prep(frame.loc[keys],t.COLS)
   for key in ['median','center','scale']:np.testing.assert_allclose(a['prep'][key],expected[key],rtol=0,atol=0)
   assert a['model'].__class__==t.estimator(t.CONFIGS[rec['config']]).__class__
   expected_params=t.estimator(t.CONFIGS[rec['config']]).get_params();actual_params=a['model'].get_params()
   for key in expected_params:
    if key not in ['n_jobs','thread_count']:assert actual_params[key]==expected_params[key]
   checked+=1
   if rec['phase']=='inner':cache[rec['config']]=m.predict(a,frame.loc[vk])
   else:
    fi=np.asarray(a['model'].feature_importances_,dtype=float)
    importance.append(dict(issue=d['issue'],config=rec['config'],flow_internal_importance_fraction=float((fi[2]+fi[5])/fi.sum()) if fi.sum()>0 else 0.,note='Internal model importance, not causal contribution or permutation ablation'))
  window=[a for a in trials if a['issue']==d['issue']]
  assert len(window)==32
  for a in window:
   met=m.metric((1-a['weight'])*pp+a['weight']*cache[a['config']],truth)
   for k in ['mae','rank_ic','pearson_ic','rows']:np.testing.assert_allclose(met[k],a[k],rtol=0,atol=1e-8)
   trial_count+=1
  for strategy in ['standalone','hybrid']:
   candidates=[a for a in window if a['weight']==1.] if strategy=='standalone' else window
   good=[a for a in candidates if a['mae']<bm['mae']-.01 and a['rank_ic']>bm['rank_ic']+.0001]
   best=min(good or candidates,key=lambda a:(a['mae'],-a['rank_ic'],a['config'],a['weight']))
   assert d[strategy]==best
   a=joblib.load(d[strategy+'_artifact']);pair=joblib.load(d['pair_artifact']);w=1. if strategy=='standalone' else best['weight']
   for keys,saved in [(te,p[p.issue==d['issue']]),(ot,training[training.issue==d['issue']])]:
    pred=(1-w)*m.predict(pair,frame.loc[keys])+w*m.predict(a,frame.loc[keys])
    np.testing.assert_allclose(pred,saved[strategy],rtol=0,atol=1e-7)
   vp=(1-w)*pp+w*cache[best['config']]
   np.testing.assert_allclose(vp,validation.loc[validation.issue==d['issue'],strategy],rtol=0,atol=1e-7)
  window_result=dict(issue=d['issue'][:10],standalone=d['standalone']['config'],hybrid_model=d['hybrid']['config'],hybrid_weight=d['hybrid']['weight'])
  for strategy in ['standalone','hybrid','baseline']:
   met=m.metric(p.loc[te,strategy],p.loc[te,'truth']);window_result.update({strategy+'_'+k:met[k] for k in ['mae','rank_ic']})
  from predict_three_feature import forecast
  replay=forecast(d,frame.loc[te])
  np.testing.assert_allclose(replay.standalone_magnitude_log_bps,p.loc[te,'standalone'],rtol=0,atol=1e-7)
  np.testing.assert_allclose(replay.hybrid_magnitude_log_bps,p.loc[te,'hybrid'],rtol=0,atol=1e-7)
  windows.append(window_result);print('AUDIT',issue.date(),'PASS',flush=True)
 for score in done['scores']:
  f=training if score['split']=='training' else validation if score['split']=='validation' else p
  if score['split']=='last3months':f=f.loc[f.index>=pd.Timestamp('2024-11-01',tz='UTC')]
  met=m.metric(f[score['model']],f.truth)
  for k in met:np.testing.assert_allclose(met[k],score[k],rtol=0,atol=1e-8)
 audit=dict(status='PASS',artifacts_checked=checked,validation_candidates_recomputed=trial_count,all_train_validation_test_predictions_replayed=True,strict_label_purges=True,exact_three_features=True,no_baseline_fallback=True,unique_test_rows=len(p),audit_fits=0)
 m.dump(out/'AUDIT.json',audit)
 pd.DataFrame(windows).to_csv(out/'window_scores.csv',index=False);pd.DataFrame(importance).to_csv(out/'flow_internal_importance.csv',index=False)
 scoreframe=pd.DataFrame(done['scores']);configs=pd.DataFrame([dict(config=k,**v) for k,v in t.CONFIGS.items()])
 text=f"""<!doctype html><html><head><meta charset="utf-8"><title>Three-feature walk-forward results</title><style>body{{font:16px system-ui;max-width:1300px;margin:35px auto;padding:24px;color:#183247}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}th{{background:#edf3f6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body>
<h1>Three-feature tuned tree models: 48-hour magnitude</h1>
<p>Four chronological outer windows, each with the preceding three months as inner validation and strict 48-hour label maturity. No random train/test split. {done['new_fits']} predictive fits:32 inner fits and {done['new_fits']-32} selected outer refits. Selected strategies always use newly trained three-feature models; no baseline substitution.</p>
<h2>Exact input columns</h2><pre>{html.escape(chr(10).join(t.COLS))}</pre>
<p>Training-only median imputation and standardization, with one missing-value indicator per predictor. These are three market features and six design columns. The median-regression blend component retains the original two liquidation features.</p>
<h2>Overall test comparison:18,481 matching rows</h2>{table(scoreframe[scoreframe.split=='test'])}
<p>The target is MAE below180bps and Spearman IC above0.07738982. Compare both metrics; validation performance is not the test result. The historical baseline reconstruction is181.525993bps; its source PDF reports181.523868bps.</p>
<h2>Training, validation and last-three-month results</h2>{table(scoreframe[scoreframe.split!='test'])}
<p>Training rows repeat across expanding fits; unique counts are supplied. Validation selected parameters and weights. Test timestamps are unique. Only48h was trained.</p>
<h2>Models and hyperparameters</h2>{table(configs.fillna(''))}
<p>RF uses absolute-error splitting and bootstrap samples of4096. CatBoost uses MAE loss. LightGBM uses L1 regression. All seeds are48. Source code contains complete settings.</p>
<h2>Selected model per outer window</h2>{table(pd.DataFrame(windows))}
<p>Standalone and blend strategies are selected separately. Candidates improving both earlier validation MAE and IC are preferred; otherwise the lowest-MAE challenger is still evaluated. Blend weights are10%,25%,50%,100%; zero is excluded. The baseline itself is never selected as a fallback.</p>
<h2>Verification</h2><pre>{html.escape(json.dumps(audit,indent=2))}</pre>
<p>The feature was nominated after earlier historical research. This is a retrospective comparison, not untouched confirmation. No model is declared superior unless the same-row outer metrics support it. Internal feature importance is saved separately and does not establish the flow feature's causal or incremental contribution.</p>
<p>Training: Code/three_feature_tree_tuning.py. Audit/report: Code/audit_three_feature_tuning.py. Exact predictions, parameters, selection records and fitted artifacts are retained alongside this report.</p></body></html>"""
 for a,b in [('below180','below 180'),('above0.077','above 0.077'),('is181.525','is 181.525'),('reports181.523','reports 181.523'),('Only48h','Only 48h'),('of4096','of 4096'),('are48','are 48'),('are10%','are 10%'),('comparison:18,481','comparison: 18,481'),('fits:32','fits: 32')]:text=text.replace(a,b)
 (out/'RESULTS.html').write_text(text,encoding='utf-8')
 import importlib.metadata as md
 m.dump(out/'ENVIRONMENT.json',dict(python=t.sys.version,packages={name:md.version(name) for name in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','joblib']}))
 print('ALL CHECKS PASS',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
