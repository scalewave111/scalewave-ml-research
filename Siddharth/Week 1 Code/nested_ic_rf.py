"""IC-guided nested monthly RandomForest experiment; no fits on import."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, itertools, json, sys, time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
OUT=ROOT/'Experiments/nested_ic_rf_v3'
sys.path.insert(0,str(PACKAGE/'submissions/member_02'))
import rf_walk_forward as prior
from rf_tuned_walk_forward import histories
from parquet_compat import read_frame
SEED=48
CONFIGS=[]
for name,pool,k,depth,leaf,mf,target,transform,interact in [
 ('raw_all','raw',0,4,64,'sqrt','raw','raw',False),
 ('raw_all_deeper','raw',0,8,32,.3,'raw','raw',False),
 ('report_only','report',0,5,48,.7,'raw','raw',False),
 ('engineered_ic24','engineered',24,4,64,1.,'raw','raw',False),
 ('engineered_ic64','engineered',64,6,32,.7,'raw','raw',False),
 ('engineered_ic128','engineered',128,8,32,.5,'raw','raw',False),
 ('engineered_ic64_log','engineered',64,7,32,.7,'log','raw',False),
 ('engineered_ic64_sqrt','engineered',64,7,48,.7,'sqrt','raw',False),
 ('ic64_rank_interactions','engineered',64,6,48,.7,'raw','rank',True),
 ('report_history','report_history',0,4,64,.7,'raw','raw',False),
 ('ic64_robust_interactions','engineered',64,5,48,.7,'raw','robust',True),
 ('residual_ic64','engineered',64,4,64,.7,'residual','robust',True)]:
 CONFIGS.append(dict(id=name,pool=pool,k=k,depth=depth,leaf=leaf,mf=mf,target=target,transform=transform,interact=interact))

def write(p,obj):
 p.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8')

def load():
 xt,yt,xd,yd=prior.data()
 x=pd.concat([xt,xd]).sort_index().astype('float32')
 y=pd.concat([yt,yd]).sort_index()
 x.index=x.index.as_unit('ns'); y.index=y.index.as_unit('ns')
 y['label_end']=pd.to_datetime(y.label_end,utc=True).astype('datetime64[ns, UTC]')
 assert x.index.equals(y.index) and x.index.is_unique
 assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
 mapping=json.loads((ROOT/'Report/ic_rf_sources/magnitude_feature_mapping.json').read_text())
 report=[r['column'] for r in mapping if r['mapped']]
 assert len(report)==38 and set(report).issubset(x.columns)
 return x,y,xd.index.as_unit('ns'),report

def engineer(x,y,report):
 x=x.replace([np.inf,-np.inf],np.nan); extra={}
 for c in report:
  s=x[c]; extra['ABS__'+c]=s.abs()
  for h in [1,24]:
   lag=s.shift(freq=pd.Timedelta(hours=h)).reindex(s.index)
   extra[f'LAG{h}H__'+c]=lag
   extra[f'DELTA{h}H__'+c]=s-lag
  for h in [6,24,168]:
   r=s.rolling(f'{h}h',closed='left',min_periods=8); m=r.mean(); sd=r.std()
   extra[f'MEAN{h}H__'+c]=m
   extra[f'STD{h}H__'+c]=sd
   extra[f'INNOV{h}H__'+c]=s-m
   extra[f'Z{h}H__'+c]=((s-m)/sd.clip(lower=1e-4)).clip(-20,20)
 order={'5m':5,'15m':15,'1h':60,'4h':240,'12h':720,'1d':1440}
 for stem in sorted({c.rsplit('__',2)[0] for c in report}):
  cols=[c for c in x if c.rsplit('__',2)[0]==stem and c.split('__')[-2] in order]
  cols.sort(key=lambda c:order[c.split('__')[-2]])
  for a,b in zip(cols[:-1],cols[1:]):
   tag=a+'__VS__'+b.split('__')[-2]
   extra['SLOPE__'+tag]=x[a]-x[b]
   extra['BALANCE__'+tag]=(x[a]-x[b])/(x[a].abs()+x[b].abs()+1.)
 for family in sorted({c.split('__')[1] for c in x}):
  group=x[[c for c in x if c.split('__')[1]==family and 'percentile' in c]]
  if len(group.columns):
   extra[f'FAMILY__{family}__mean_percentile']=group.mean(axis=1)
   extra[f'FAMILY__{family}__dispersion']=group.std(axis=1)
   extra[f'FAMILY__{family}__coverage']=group.notna().mean(axis=1)
 for name,phase,period in [('hour',x.index.hour+x.index.minute/60,24),('weekday',x.index.dayofweek,7)]:
  extra[f'TIME__{name}_sin']=pd.Series(np.sin(2*np.pi*phase/period),index=x.index)
  extra[f'TIME__{name}_cos']=pd.Series(np.cos(2*np.pi*phase/period),index=x.index)
 result=pd.concat([x,pd.DataFrame(extra,index=x.index),histories(y,x.index)],axis=1)
 assert result.columns.is_unique
 return result.replace([np.inf,-np.inf],np.nan).astype('float32')

def ic(a,b):
 a=np.asarray(a); b=np.asarray(b); ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()<30 or np.unique(a[ok]).size<2 or np.unique(b[ok]).size<2: return 0.
 return float(spearmanr(a[ok],b[ok]).statistic)

def metrics(p,y):
 p=np.asarray(p); y=np.asarray(y)
 return dict(mae=float(np.mean(abs(p-y))),rmse=float(np.sqrt(np.mean((p-y)**2))),
             rank_ic=ic(p,y),pearson_ic=float(pearsonr(p,y).statistic) if np.unique(p).size>1 else None,rows=len(p))

def screen(frame,response):
 admitted=frame.columns[(frame.notna().sum()>=1000)&(frame.nunique()>=2)]
 blocks=frame.index.strftime('%Y-%m'); records=[]; target=response.to_numpy()
 for c in admitted:
  v=frame[c].to_numpy(); full=ic(v,target)
  by=[ic(v[blocks==month],target[blocks==month]) for month in sorted(set(blocks))]
  mean=float(np.mean(by)); sd=float(np.std(by)); agreement=float(np.mean(np.sign(by)==np.sign(mean)))
  records.append(dict(feature=c,rank_ic=full,block_mean_ic=mean,block_std_ic=sd,sign_agreement=agreement,
                      screen_score=abs(mean)*agreement+.25*abs(full)-.1*sd,observations=int(np.isfinite(v).sum())))
 return pd.DataFrame(records).sort_values(['screen_score','feature'],ascending=[False,True]).reset_index(drop=True)

def preprocessing(train,ranking,cfg,report):
 if cfg['pool']=='raw': allowed={c for c in train if c.startswith('F4Z__')}
 elif cfg['pool']=='report': allowed=set(report)
 elif cfg['pool']=='report_history': allowed=set(report)|{c for c in train if c.startswith('HIST__')}
 else: allowed=set(train)
 ranked=[c for c in ranking.feature if c in allowed]
 if cfg['k']:
  pool=ranked[:max(cfg['k']*4,256)]
  matrix=train[pool].rank().corr().abs().fillna(0); cols=[]
  for c in pool:
   if not cols or matrix.loc[c,cols].max()<.97: cols.append(c)
   if len(cols)>=cfg['k']: break
 else: cols=ranked
 if not cols: raise ValueError('No admitted predictors')
 v=train[cols]
 return dict(columns=cols,median=v.median(),lo=v.quantile(.01),hi=v.quantile(.99),
             scale=(v.quantile(.75)-v.quantile(.25)).clip(lower=1e-4),transform=cfg['transform'],
             sorted_values={c:np.sort(v[c].dropna().to_numpy()) for c in cols} if cfg['transform']=='rank' else {},
             interaction_columns=cols[:8] if cfg['interact'] else [])

def design(frame,prep):
 v=frame[prep['columns']]; missing=v.isna().to_numpy(dtype=np.float32); v=v.fillna(prep['median'])
 if prep['transform']=='rank':
  v=pd.DataFrame({c:2*np.searchsorted(prep['sorted_values'][c],v[c],side='right')/len(prep['sorted_values'][c])-1 for c in v},index=v.index)
 elif prep['transform']=='robust':
  v=((v.clip(prep['lo'],prep['hi'],axis=1)-prep['median'])/prep['scale']).clip(-20,20)
  v=np.sign(v)*np.log1p(abs(v))
 arrays=[v.to_numpy(dtype=np.float32),missing]
 if prep['interaction_columns']:
  z=v[prep['interaction_columns']].to_numpy(dtype=np.float32)
  arrays.append(np.column_stack([z[:,i]*z[:,j] for i,j in itertools.combinations(range(z.shape[1]),2)]))
 a=np.column_stack(arrays); assert np.isfinite(a).all()
 return a

def predict(artifact,frame,reference):
 p=artifact['model'].predict(design(frame,artifact['prep'])); target=artifact['config']['target']
 if target=='log': p=np.expm1(p)
 elif target=='sqrt': p=np.square(np.maximum(p,0))
 elif target=='residual': p=p+reference.loc[frame.index].to_numpy()
 return np.maximum(p,0)

def train_keys(y,start):
 return y.index[(y.index>=start-pd.DateOffset(months=3))&(y.label_end<start)]

def month_keys(index,start):
 return index[(index>=start)&(index<start+pd.DateOffset(months=1))]

class Runner:
 def __init__(self,x,y,report,ref):
  self.x=x; self.y=y; self.report=report; self.ref=ref; self.rankings={}; self.receipts=json.loads((OUT/'fit_ledger.json').read_text()) if (OUT/'fit_ledger.json').exists() else []
 def fit_predict(self,cfg,start,keys):
  tag=start.strftime('%Y-%m')+'__'+cfg['id']; path=OUT/'models'/f'{tag}.joblib'
  if path.exists():
   artifact=joblib.load(path)
   return predict(artifact,self.x.loc[keys],self.ref),path
  tk=train_keys(self.y,start)
  assert self.y.loc[tk,'label_end'].max()<start<=keys.min()
  train=self.x.loc[tk]; response=self.y.loc[tk,'magnitude_log_bps']
  sr=response-self.ref.loc[tk] if cfg['target']=='residual' else response
  rk=(str(start),cfg['target']=='residual')
  if rk not in self.rankings:
   self.rankings[rk]=screen(train,sr)
   self.rankings[rk].to_csv(OUT/'ic'/f'{start:%Y-%m}__{cfg["target"] if rk[1] else "magnitude"}.csv',index=False)
  prep=preprocessing(train,self.rankings[rk],cfg,self.report)
  params=dict(n_estimators=96,criterion='squared_error' if cfg['target'] in ['log','sqrt'] else 'absolute_error',
              max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=cfg['mf'],
              max_samples=min(2048,len(tk)),bootstrap=True,random_state=SEED,n_jobs=6)
  if len(self.receipts)>=70: raise RuntimeError('Registered fit cap exhausted')
  receipt=dict(id=tag,config=cfg,issuance=str(start),training_start=str(tk.min()),training_end=str(tk.max()),
               max_label_end=str(self.y.loc[tk,'label_end'].max()),train_rows=len(tk),features=prep['columns'],parameters=params,status='STARTED')
  self.receipts.append(receipt); write(OUT/'fit_ledger.json',self.receipts)
  print(f'FIT {len(self.receipts)}/70 {tag} n={len(tk)} p={len(prep["columns"])}',flush=True); tick=time.monotonic()
  try:
   fy=sr if cfg['target']=='residual' else response
   if cfg['target']=='log': fy=np.log1p(fy)
   elif cfg['target']=='sqrt': fy=np.sqrt(fy)
   with threadpool_limits(limits=1):
    model=RandomForestRegressor(**params).fit(design(train,prep),fy)
   artifact=dict(model=model,prep=prep,config=cfg,issuance=start,train_keys=tk)
   result=predict(artifact,self.x.loc[keys],self.ref)
   joblib.dump(artifact,path,compress=3)
   receipt.update(status='PASS',seconds=time.monotonic()-tick)
   write(OUT/'fit_ledger.json',self.receipts)
   print(f'DONE {tag} {receipt["seconds"]:.1f}s',flush=True)
   return result,path
  except BaseException as error:
   receipt.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=repr(error)); write(OUT/'fit_ledger.json',self.receipts); raise

def calibrate(p,y):
 options=[]
 for scale,offset in itertools.product([.75,.9,1.,1.1,1.25],[-25.,0.,25.]):
  value=np.maximum(0,scale*p+offset)
  options.append((float(np.mean(abs(value-y))),abs(scale-1)+abs(offset)/1000,scale,offset))
 score,_,scale,offset=min(options)
 return dict(scale=scale,offset=offset,mae=score)

def read_csv(path):
 f=pd.read_csv(path,index_col='decision_time'); f.index=pd.to_datetime(f.index,utc=True).as_unit('ns')
 return f

def register(x,months,report,resume=False):
 if OUT.exists() and resume:
  contract=json.loads((OUT/'CONTRACT.json').read_text())
  assert contract['configs']==CONFIGS and contract['outer_months']==list(map(str,months))
  assert (OUT/'RESUME.json').exists(), 'Resume must be explicitly recorded'
  return
 OUT.mkdir(parents=True,exist_ok=False)
 for name in ['models','ic','inner','outer']: (OUT/name).mkdir()
 write(OUT/'CONTRACT.json',dict(created_utc=pd.Timestamp.now(tz='UTC'),outer_months=list(map(str,months)),
  authorization='User requested IC-guided engineering, nested walk-forward and training/testing scores.',
  protocol='Final 3 represented calendar months as outer tests, monthly refits; each outer selects on 3 preceding monthly inner folds.',
  training='Each fit uses preceding 3 calendar months, label_end strictly before issuance.',
  inner_maturity='Inner validation labels also require label_end strictly before outer issuance.',
  features='All 522 existing masked F4Z candidates plus magnitude-report-derived causal engineering and matured history.',
  feature_selection='Pairwise-complete Spearman IC and monthly sign stability; >=1000 finite varied readings per fit; rank-correlation redundancy cutoff .97.',
  configs=CONFIGS,fit_cap=70,caching='Same config/issuance reuses identical fits across overlapping folds.',
  selection='Inner original-scale pooled MAE, tie by rank IC/name. Calibrate scale/offset and compare top-three standalone RF average inside inner layer.',
  calibration_scales=[.75,.9,1.,1.1,1.25],calibration_offsets=[-25.,0.,25.],
  residual_candidate='Explicit frozen-baseline plus RF residual hybrid, separately labeled from standalone RF.',
  overall_training='Pooled in-sample predictions from monthly fitted models; repeated timestamps across overlapping training windows are retained and disclosed.',
  overall_testing='Unique outer timestamp predictions, pooled once; all comparators scored on identical keys.',
  prior_exposure='Historical development and source feature rankings were previously exposed. Nested selection does not create an untouched holdout.',
  stop='One bounded run; no outer-test-driven retuning or guaranteed win.',report_columns=report,engineered_count=len(x.columns),
  original_leaderboard_eligible=False))
 write(OUT/'environment.json',{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','joblib','duckdb']})
 paths=[Path(__file__),PACKAGE/'MANIFEST.json',ROOT/'Report/ic_rf_sources/magnitude_feature_mapping.json']
 write(OUT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())

def assemble_predictions(predictions,s):
 if 'rf_top3_ensemble' in [s['selected_standalone'],s['selected_forecast']]:
  predictions['rf_top3_ensemble']=np.mean([predictions[name] for name in s['top_three']],axis=0)
 result={'nested_raw_rf':predictions[s['best_raw_rf']]}
 for column,name in [('nested_standalone_rf',s['selected_standalone']),('nested_selected',s['selected_forecast'])]:
  c=s['calibrations'][name]; result[column]=np.maximum(0,c['scale']*predictions[name]+c['offset'])
 return result

def run(resume=False):
 raw,y,dev,report=load()
 last=dev.max().normalize().replace(day=1); months=pd.date_range(last-pd.DateOffset(months=2),last,freq='MS')
 print('OUTER MONTHS',list(map(str,months)),flush=True)
 ref=prior.frozen_predictions(raw).magnitude_log_bps
 x=engineer(raw,y,report); register(x,months,report,resume=resume)
 cutoff=months[0]; changed=y.copy(); changed.loc[changed.label_end>=cutoff,'magnitude_log_bps']+=100000
 earlier=raw.index[raw.index<=cutoff]
 np.testing.assert_allclose(histories(y,earlier),histories(changed,earlier),rtol=0,atol=0,equal_nan=True)
 np.testing.assert_allclose(x.loc[earlier],engineer(raw.loc[earlier],y.loc[earlier],report),rtol=0,atol=0,equal_nan=True)
 write(OUT/'PREFLIGHT.json',dict(status='PASS',future_label_perturbation=True,prefix_invariance=True,features=len(x.columns)))
 runner=Runner(x,y,report,ref); results=[]; train_results=[]; score_rows=[]
 for outer in months:
  print('OUTER',str(outer),flush=True)
  starts=pd.date_range(outer-pd.DateOffset(months=3),outer-pd.DateOffset(months=1),freq='MS'); inner=[]
  for start in starts:
   keys=month_keys(x.index,start); keys=keys[y.loc[keys,'label_end']<outer]
   assert y.loc[keys,'label_end'].max()<outer
   block=pd.DataFrame(index=keys); block['truth']=y.loc[keys,'magnitude_log_bps']
   for cfg in CONFIGS: block[cfg['id']],_=runner.fit_predict(cfg,start,keys)
   inner.append(block)
  val=pd.concat(inner).sort_index(); target=val.truth.to_numpy()
  val.to_csv(OUT/'inner'/f'{outer:%Y-%m}_predictions.csv',index_label='decision_time')
  scores={c['id']:metrics(val[c['id']],target) for c in CONFIGS}
  pure=[c['id'] for c in CONFIGS if c['target']!='residual']
  ranked=sorted(pure,key=lambda c:(scores[c]['mae'],-scores[c]['rank_ic'],c)); top=ranked[:3]
  candidates={name:val[name].to_numpy() for name in pure}
  candidates['rf_top3_ensemble']=val[top].mean(axis=1).to_numpy()
  candidates['residual_ic64']=val.residual_ic64.to_numpy()
  cal={name:calibrate(value,target) for name,value in candidates.items()}
  selected_pure=min(pure+['rf_top3_ensemble'],key=lambda c:(cal[c]['mae'],c))
  selected=min(candidates,key=lambda c:(cal[c]['mae'],c))
  s=dict(outer=str(outer),inner_starts=list(map(str,starts)),inner_rows=len(val),
         latest_inner_label_end=str(y.loc[val.index,'label_end'].max()),scores=scores,calibrations=cal,
         top_three=top,best_raw_rf=ranked[0],selected_standalone=selected_pure,
         selected_forecast=selected,selected_is_hybrid=selected=='residual_ic64',frozen_before_outer_scoring=True)
  write(OUT/'outer'/f'{outer:%Y-%m}_selection.json',s)
  print('SELECTED',outer.strftime('%Y-%m'),selected,'standalone',selected_pure,flush=True)
  keys=month_keys(dev,outer); tk=train_keys(y,outer)
  required={ranked[0],selected_pure,selected}
  if 'rf_top3_ensemble' in required: required.update(top); required.remove('rf_top3_ensemble')
  pp={}; tp={}; paths={}
  for name in sorted(required):
   cfg=next(c for c in CONFIGS if c['id']==name)
   pp[name],path=runner.fit_predict(cfg,outer,keys)
   artifact=joblib.load(path); tp[name]=predict(artifact,x.loc[tk],ref); paths[name]=str(path.relative_to(OUT))
  test=pd.DataFrame(assemble_predictions(pp,s),index=keys)
  train=pd.DataFrame(assemble_predictions(tp,s),index=tk)
  inn=pd.DataFrame(assemble_predictions({name:candidates[name] for name in candidates},s),index=val.index)
  for frame,split,true in [(test,'test',y.loc[keys,'magnitude_log_bps']),(train,'train_in_sample',y.loc[tk,'magnitude_log_bps']),(inn,'inner_validation',val.truth)]:
   for name in frame:
    score_rows.append(dict(month=outer.strftime('%Y-%m'),split=split,model=name,**metrics(frame[name],true)))
  train['truth']=y.loc[tk,'magnitude_log_bps']; train['outer_month']=outer.strftime('%Y-%m'); train_results.append(train)
  test['selected_is_hybrid']=selected=='residual_ic64'; test['month']=outer.strftime('%Y-%m')
  test.to_csv(OUT/'outer'/f'{outer:%Y-%m}_predictions.csv',index_label='decision_time')
  s['model_paths']=paths; write(OUT/'outer'/f'{outer:%Y-%m}_selection.json',s); results.append(test)
 predictions=pd.concat(results).sort_index(); expected=dev[dev>=months[0]]
 assert predictions.index.equals(expected) and predictions.index.is_unique
 predictions.to_csv(OUT/'predictions.csv',index_label='decision_time')
 training=pd.concat(train_results); training.to_csv(OUT/'training_predictions.csv',index_label='decision_time')
 pd.DataFrame(score_rows).to_csv(OUT/'train_validation_test_by_month.csv',index=False)
 overall=[]
 for name in ['nested_raw_rf','nested_standalone_rf','nested_selected']:
  overall.append(dict(split='train_in_sample_pooled',model=name,**metrics(training[name],training.truth)))
  overall.append(dict(split='test_outer_unique',model=name,**metrics(predictions[name],y.loc[expected,'magnitude_log_bps'])))
 pd.DataFrame(overall).to_csv(OUT/'overall_training_testing.csv',index=False)
 evaluate(predictions,y,ref); verify()

def evaluate(p,y,ref):
 keys=p.index; truth=y.loc[keys,'magnitude_log_bps']; values=p[['nested_raw_rf','nested_standalone_rf','nested_selected']].copy()
 values['frozen_pair']=ref.loc[keys]
 values['previous_rf']=read_csv(prior.OUT/'predictions.csv').loc[keys,'magnitude_log_bps']
 values['previous_hybrid']=read_csv(PACKAGE/'submissions/member_02/rf_walk_forward_v2/predictions.csv').loc[keys,'magnitude_log_bps']
 values['fixed_M0']=read_frame(PACKAGE/'benchmarks/development_predictions.parquet').loc[keys,'M0_magnitude']
 for month in p.month.unique():
  start=pd.Timestamp(month,tz='UTC'); tk=train_keys(y,start)
  values.loc[p.month==month,'rolling_median']=float(y.loc[tk,'magnitude_log_bps'].median())
 assert values.notna().all().all()
 loss=values.sub(truth,axis=0).abs()
 scoreboard=pd.DataFrame([dict(model=name,**metrics(values[name],truth)) for name in values]).set_index('model').sort_values('mae')
 scoreboard.to_csv(OUT/'scoreboard.csv')
 loss.groupby(p.month).mean().to_csv(OUT/'monthly_mae.csv')
 pd.DataFrame({month:{name:ic(values.loc[p.month==month,name],truth.loc[p.month==month]) for name in values} for month in p.month.unique()}).T.to_csv(OUT/'monthly_rank_ic.csv')
 values.assign(truth=truth).to_csv(OUT/'all_predictions.csv',index_label='decision_time')
 intervals=[]
 for days in [7,14]:
  groups=((keys-keys.min()).total_seconds()//(days*86400)).astype(int)
  sizes=loss.groupby(groups).size().to_numpy(); draws=np.random.default_rng(SEED+days).integers(0,len(sizes),(3000,len(sizes)))
  for model in ['nested_standalone_rf','nested_selected']:
   for baseline in ['frozen_pair','previous_rf','previous_hybrid','rolling_median']:
    sums=(loss[model]-loss[baseline]).groupby(groups).sum().to_numpy()
    lo,hi=np.quantile(sums[draws].sum(axis=1)/sizes[draws].sum(axis=1),[.025,.975])
    intervals.append(dict(model=model,baseline=baseline,block_days=days,low=float(lo),high=float(hi)))
 write(OUT/'paired_intervals.json',intervals)
 summary=dict(status='EXECUTED_ON_REAL_DATA',rows=len(keys),start=str(keys.min()),end=str(keys.max()),
  fit_count=len(json.loads((OUT/'fit_ledger.json').read_text())),scores=[dict(model=name,**metrics(values[name],truth)) for name in scoreboard.index],
  comparisons={model:{base:dict(mae_reduction=float(loss[base].mean()-loss[model].mean()),improvement_percent=float(100*(1-loss[model].mean()/loss[base].mean())),beats=bool(loss[model].mean()<loss[base].mean())) for base in ['frozen_pair','previous_rf','previous_hybrid']} for model in ['nested_standalone_rf','nested_selected']},
  outer_months_with_hybrid=int(p.groupby('month').selected_is_hybrid.first().sum()),
  interpretation='Nested selection; historically exposed dataset and report priors. Nominal intervals, not untouched confirmation. Latest month partial.')
 write(OUT/'RESULT.json',summary)
 print('FINAL SCOREBOARD\n'+scoreboard.to_string(),flush=True)

def verify():
 raw,y,dev,report=load(); x=engineer(raw,y,report); ref=prior.frozen_predictions(raw).magnitude_log_bps
 saved=read_csv(OUT/'predictions.csv'); contract=json.loads((OUT/'CONTRACT.json').read_text())
 assert saved.index.equals(dev[dev>=pd.Timestamp(contract['outer_months'][0])])
 maximum=0.
 for path in sorted((OUT/'outer').glob('*_selection.json')):
  s=json.loads(path.read_text()); start=pd.Timestamp(s['outer']); keys=month_keys(saved.index,start)
  assert pd.Timestamp(s['latest_inner_label_end'])<start
  pp={}
  for name,modelpath in s['model_paths'].items():
   a=joblib.load(OUT/modelpath); expected=train_keys(y,start); assert a['train_keys'].equals(expected)
   np.testing.assert_allclose(a['prep']['median'],x.loc[expected,a['prep']['columns']].median(),rtol=0,atol=0)
   pp[name]=predict(a,x.loc[keys],ref)
  for column,value in assemble_predictions(pp,s).items():
   error=float(np.max(abs(saved.loc[keys,column].to_numpy()-value))); maximum=max(maximum,error); assert error<1e-9
  val=read_csv(OUT/'inner'/f'{start:%Y-%m}_predictions.csv')
  assert (y.loc[val.index,'label_end']<start).all()
  for name,score in s['scores'].items(): assert abs(float(abs(val[name]-val.truth).mean())-score['mae'])<1e-9
 for r in json.loads((OUT/'fit_ledger.json').read_text()):
  assert r['status'] in ['PASS','INTERRUPTED'] and pd.Timestamp(r['max_label_end'])<pd.Timestamp(r['issuance'])
 for path,digest in json.loads((OUT/'source_hashes.json').read_text()).items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 for path,digest in json.loads((PACKAGE/'MANIFEST.json').read_text()).items(): assert hashlib.sha256((PACKAGE/path).read_bytes()).hexdigest()==digest
 overall=pd.read_csv(OUT/'overall_training_testing.csv'); training=read_csv(OUT/'training_predictions.csv')
 for _,r in overall.iterrows():
  p=training[r.model] if r.split=='train_in_sample_pooled' else saved[r.model]
  t=training.truth if r.split=='train_in_sample_pooled' else y.loc[saved.index,'magnitude_log_bps']
  assert abs(metrics(p,t)['mae']-r.mae)<1e-9
 write(OUT/'VERIFICATION.json',dict(status='PASS',rows=len(saved),max_replay_error=maximum,
  outer_coverage=True,training_windows_and_medians=True,inner_maturity=True,inner_mae_recomputed=True,
  overall_train_test_mae_recomputed=True,original_manifest_unchanged=True,additional_fits=0))
 print('VERIFICATION PASS',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['run','resume','verify']); args=parser.parse_args()
 verify() if args.command=='verify' else run(resume=args.command=='resume')
