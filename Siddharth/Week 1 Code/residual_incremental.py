"""Small OOF residual corrections with individual walk-forward feature admission."""
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import filtered_walkforward as fw
import canonical_return_features as crf
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,ks_2samp
ROOT=fw.ROOT;PREVIOUS=ROOT/'Experiments/filtered_walkforward_v1'
def dump(p,v):fw.nb.dump(p,v)
def price_features(prices,index):
 p=prices.sort_index();assert p.index.is_unique and (p.dropna()>0).all()
 lag=p.shift(freq='30min').reindex(p.index);ret=10000*np.log(p/lag)
 f={}
 for hours in [4,24,168]:
  roll=ret.rolling(f'{hours}h',closed='left',min_periods=int(hours*2*.9))
  f[f'PRICE__realized_vol_{hours}h']=np.sqrt(roll.apply(lambda a:np.sum(a*a),raw=True))
  f[f'PRICE__mean_absolute_return_{hours}h']=ret.abs().rolling(f'{hours}h',closed='left',min_periods=int(hours*2*.9)).mean()
 f['PRICE__vol_ratio_4h_24h']=(f['PRICE__realized_vol_4h']/np.sqrt(4))/(f['PRICE__realized_vol_24h']/np.sqrt(24)).clip(lower=1e-6)
 return pd.DataFrame(f).reindex(index)
def build_frame(raw,y,price=None,returns5=None):
 parents=[c for c in raw if c.split('__')[1] in fw.FAMILIES]
 frame=raw[parents].copy()
 frame=frame.join(fw.engineered(raw,parents))
 hist=fw.nb.old.histories(y,raw.index)
 frame=frame.join(hist[['HIST__median_7d','HIST__median_30d','HIST__mean_30d']])
 if price is not None:frame=frame.join(price_features(price,raw.index))
 if returns5 is not None:frame=frame.join(crf.features(returns5,raw.index))
 return frame.replace([np.inf,-np.inf],np.nan)
def extend_baseline(raw,y,dev,out,hours,prices):
 # Build a chronological prediction stream, including pre-development history for residual targets.
 runner=fw.Runner(raw,y,out,hours,prices)
 if hours==48 and not prices:
  for p in list((PREVIOUS/'models').glob('*.joblib'))+list((ROOT/'Experiments/residual_incremental_v1/models').glob('*.joblib')):
   if not (p.stem.endswith('__pair_exp') or p.stem.endswith('__reference_rf')):continue
   if p.stem in runner.cache:continue
   a=fw.joblib.load(p);start=a['issuance'];tk=fw.keys_for(y,start,'exp')
   if not a['train_keys'].equals(tk):continue
   runner.cache[p.stem]=a
   fw.joblib.dump(a,out/'models'/p.name,compress=3)
   runner.ledger.append(dict(id=p.stem,status='REUSED',source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 pieces=[]
 for start in pd.date_range('2023-01-01','2024-10-01',freq='QS',tz='UTC'):
  keys=y.index[(y.index>=start)&(y.index<start+pd.DateOffset(months=3))]
  if start==pd.Timestamp('2024-10-01',tz='UTC'):keys=y.index[y.index>=start]
  pp=runner.base(start,keys)
  pieces.append(pd.DataFrame(dict(baseline=pp,baseline_issuance=str(start)),index=keys))
 base=pd.concat(pieces).sort_index()
 base['truth']=y.loc[base.index,'magnitude_log_bps'];base['residual']=base.truth-base.baseline
 base.to_csv(out/'oof_baseline.csv',index_label='decision_time')
 assert base.index.is_unique
 for _,g in base.groupby('baseline_issuance'):
  start=pd.Timestamp(g.baseline_issuance.iloc[0])
  assert (g.index>=start).all()
  for kind in ['pair_exp','reference_rf']:
   a=runner.cache[f'{start:%Y-%m}__{kind}'];assert y.loc[a['train_keys'],'label_end'].max()<start
 return base
def residual_keys(y,base,start):
 return base.index[y.loc[base.index,'label_end']<start]
def preliminary(frame,y,base,tk,out,outer):
 records=[];available=frame.loc[tk];r=base.loc[tk,'residual'];truth=y.loc[tk,'magnitude_log_bps']
 for c in frame:
  s=available[c];ok=s.notna()
  if ok.sum()<1000 or ok.mean()<.7 or s.nunique()<3:continue
  rho=fw.ic(s[ok].to_numpy(),r[ok].to_numpy())
  lo,hi=r[ok].quantile([.25,.75]);ks=float(ks_2samp(s[ok & (r<=lo)],s[ok & (r>=hi)],method='asymp').statistic)
  target_ic,target_ks,lift=fw.effect(s.to_numpy(),truth.to_numpy())
  family=next((f for f in fw.FAMILIES if '__'+f+'__' in c),'history' if c.startswith('HIST') else 'price' if c.startswith('PRICE') else 'baseline')
  kind='raw' if c.startswith('F4Z') else 'engineered' if family in fw.FAMILIES else family
  records.append(dict(feature=c,family=family,kind=kind,residual_ic=rho,residual_ks=ks,magnitude_ic=target_ic,magnitude_ks=target_ks,magnitude_lift=lift,
   eligible=abs(rho)>=.015 or ks>=.1,score=abs(rho)+.25*ks))
 ranked=pd.DataFrame(records).sort_values(['score','feature'],ascending=[False,True])
 ranked.to_csv(out/'screening'/f'{outer:%Y-%m}.csv',index=False)
 selected=[]
 for family in fw.FAMILIES:
  for kind in ['raw','engineered']:
   pool=ranked[(ranked.family==family)&(ranked.kind==kind)&ranked.eligible]
   chosen=[]
   for c in pool.feature:
    if chosen and available[chosen].corrwith(available[c],method='spearman').abs().max()>=.9:continue
    chosen.append(c)
    if len(chosen)==2:break
   selected+=chosen
 for kind in ['history','price']:
  selected+=list(ranked[(ranked.kind==kind)&ranked.eligible].feature.head(3))
 if 'BASE__forecast' in frame:selected+=['BASE__forecast']
 return list(dict.fromkeys(selected))[:19]
def fit_bin(s,r,start,keys,feature):
 med=float(s.median());values=s.fillna(med).to_numpy()
 cuts=np.unique(np.quantile(values,[1/3,2/3]));bins=np.searchsorted(cuts,values,side='right')
 global_median=float(np.median(r));levels=[];counts=[]
 for j in range(len(cuts)+1):
  ix=bins==j;n=int(ix.sum());counts.append(n)
  local=float(np.median(np.asarray(r)[ix])) if n else global_median
  strength=n/(n+2048.) if n>=256 else 0.
  levels.append(float(np.clip(global_median+strength*(local-global_median),-50.,50.)))
 return dict(feature=feature,issuance=str(start),train_keys=list(map(str,keys)),median=med,cuts=list(map(float,cuts)),
  levels=levels,counts=counts,global_median=global_median,shrinkage=2048,correction_cap=50)
def predict_bin(a,s):
 values=s.fillna(a['median']).to_numpy();bins=np.searchsorted(a['cuts'],values,side='right')
 return np.asarray(a['levels'])[bins]
def gate(p,truth,baseline,index):
 a=fw.nb.metrics(p,truth);b=fw.nb.metrics(baseline,truth);months=index.strftime('%Y-%m');wins=0;deltas=[]
 for month in sorted(set(months)):
  ix=months==month;x=fw.nb.metrics(np.asarray(p)[ix],np.asarray(truth)[ix]);z=fw.nb.metrics(np.asarray(baseline)[ix],np.asarray(truth)[ix])
  wins+=x['mae']<z['mae'];deltas.append(fw.nb.rankscore(x)-fw.nb.rankscore(z))
 return dict(**a,mae_gain=b['mae']-a['mae'],ic_gain=fw.nb.rankscore(a)-fw.nb.rankscore(b),
  monthly_mae_wins=int(wins),monthly_mean_ic_gain=float(np.mean(deltas)),
  eligible=bool(a['mae']<b['mae']-.01 and fw.nb.rankscore(a)>fw.nb.rankscore(b)+.0001 and wins>=4 and np.mean(deltas)>0))
def load_inputs(hours,prices,returns_path=None):
 if prices and returns_path:raise ValueError('Supply canonical prices or canonical5m returns, not both.')
 returns5=None;validation=None
 if returns_path:
  raw,original,dev,_=fw.nb.old.load()
  returns5=crf.load_returns(returns_path);validation=crf.validate(returns5,original)
  y=original if hours==48 else crf.make_labels(returns5,raw.index,hours)
  raw=raw.loc[y.index];dev=dev.intersection(y.index)
 else:raw,y,dev,_=fw.nb.load_data(hours,prices)
 return raw,y,dev,returns5,validation
def run(hours,prices,out,returns_path=None):
 raw,y,dev,returns5,source_validation=load_inputs(hours,prices,returns_path)
 if prices and hours==48:
  original=fw.nb.old.load()[1];common=y.index.intersection(original.index)
  if len(common)<1000 or np.max(abs(y.loc[common,'signed_return_log_bps']-original.loc[common,'signed_return_log_bps']))>1e-6:
   raise ValueError('Supplied prices do not reproduce canonical48h labels; refusing a changed-target baseline comparison.')
 price=None
 if prices:
  q=pd.read_csv(prices);q['decision_time']=pd.to_datetime(q.decision_time,utc=True);price=q.set_index('decision_time').price
 out.mkdir(parents=True,exist_ok=False)
 for d in ['models','baseline','screening','inner','outer','corrections']: (out/d).mkdir()
 starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 dump(out/'CONTRACT.json',dict(horizon_hours=hours,prices=str(prices) if prices else None,returns_5m=str(returns_path) if returns_path else None,outer_starts=list(map(str,starts)),test_rows=len(dev),
  model='Baseline hybrid plus average of at most3 regularized univariate three-bin median residual corrections; no replacement booster.',
  residual_target='Observed magnitude minus strictly out-of-fold quarterly reference forecast; residual labels must mature before correction issuance.',
  candidate_pool='Selected from data before earliest inner fold; up to2 raw and2 engineered per flow/volatility/funding family,3 matured history,3 absolute price features if supplied, baseline prediction; cap19.',
  inner='Three preceding chronological two-month folds, training all earlier matured OOF residuals from January2023.',
  admission='Each feature individually must improve MAE and IC, win MAE in >=4/6 months and improve mean monthly IC, before joining correction ensemble.',
  combination='At most top3 admitted nonredundant features; compare average correction of top1/2/3 at weights0/.1/.25/.5. Same stability gate; else zero.',
  regularization='Three quantile bins; bin median shrunk towards global residual median with2048 pseudo-observations; correction clipped50bps before weight.',
  residual_fit_cap=260,baseline_fit_cap=140,exposure='Previously inspected history; no untouched confirmation. One bounded run, no outer-score retuning.',
  absolute_price_data='Verified canonical5m returns' if returns_path else 'supplied prices' if prices else 'Unavailable in this run: price-level volatility features not executed.',
  target_mae=181.5238678328609 if hours==48 else None))
 (out/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'Code/filtered_walkforward.py',ROOT/'Code/nested_boost.py']})
 if source_validation:dump(out/'CANONICAL_SOURCE_VERIFICATION.json',source_validation)
 if returns_path:dump(out/'returns_source_hash.json',{str(returns_path):hashlib.sha256(Path(returns_path).read_bytes()).hexdigest(),str(ROOT/'Code/canonical_return_features.py'):hashlib.sha256((ROOT/'Code/canonical_return_features.py').read_bytes()).hexdigest()})
 frame=build_frame(raw,y,price,returns5)
 cut=starts[0];parents=[c for c in raw if c.split('__')[1] in fw.FAMILIES]
 pd.testing.assert_frame_equal(frame.loc[:cut],build_frame(raw.loc[:cut],y.loc[:cut],price.loc[:cut] if price is not None else None,returns5.loc[:cut] if returns5 is not None else None))
 changed=y.copy();changed.loc[changed.label_end>=cut,'magnitude_log_bps']+=100000
 np.testing.assert_allclose(fw.nb.old.histories(y,raw.index[raw.index<=cut]),fw.nb.old.histories(changed,raw.index[raw.index<=cut]),equal_nan=True,rtol=0,atol=0)
 # Price feature invariance, even when real canonical prices are absent.
 ix=pd.date_range('2020-01-01',periods=1000,freq='30min',tz='UTC');toy=pd.Series(np.exp(np.arange(1000)*.0001),index=ix)
 pd.testing.assert_frame_equal(price_features(toy,ix).iloc[:700],price_features(toy.iloc[:700],ix[:700]))
 dump(out/'PREFLIGHT.json',dict(status='PASS',causal_features=True,matured_history=True,price_feature_prefix=True,real_price_features=prices is not None or returns_path is not None))
 base=extend_baseline(raw,y,dev,out,hours,prices);frame['BASE__forecast']=base.baseline
 residual_ledger=[];tests=[];trains=[];validations=[];all_admission=[]
 for no,outer in enumerate(starts):
  inner_starts=[outer-pd.DateOffset(months=m) for m in [6,4,2]]
  earliest=inner_starts[0];screen_keys=residual_keys(y,base,earliest)
  candidates=preliminary(frame,y,base,screen_keys,out,outer)
  blocks=[];artifacts={}
  for start in inner_starts:
   tk=residual_keys(y,base,start)
   vk=base.index[(base.index>=start)&(base.index<start+pd.DateOffset(months=2))&(y.loc[base.index,'label_end']<outer)]
   block=pd.DataFrame(dict(truth=base.loc[vk,'truth'],baseline=base.loc[vk,'baseline']),index=vk)
   for feature in candidates:
    a=fit_bin(frame.loc[tk,feature],base.loc[tk,'residual'],start,tk,feature)
    tag=f'{outer:%Y-%m}__{start:%Y-%m}__'+hashlib.sha256(feature.encode()).hexdigest()[:12]
    if len(residual_ledger)>=260:raise RuntimeError('Residual fit cap exhausted')
    dump(out/'corrections'/f'{tag}.json',a);residual_ledger.append(dict(id=tag,status='PASS',outer=str(outer),issuance=str(start),train_rows=len(tk),max_label_end=str(y.loc[tk,'label_end'].max()),feature=feature,phase='inner'))
    block[feature]=predict_bin(a,frame.loc[vk,feature])
   blocks.append(block)
  val=pd.concat(blocks).sort_index();val.to_csv(out/'inner'/f'{outer:%Y-%m}.csv',index_label='decision_time')
  admission=[]
  for feature in candidates:
   options=[]
   for weight in [.1,.25,.5]:
    pp=np.maximum(0,val.baseline+weight*val[feature]);g=gate(pp,val.truth,val.baseline,val.index)
    options.append(dict(feature=feature,weight=weight,**g))
   eligible=[o for o in options if o['eligible']]
   best=min(eligible or options,key=lambda z:(z['mae'],z['weight']))
   admission.append(best)
  admitted=sorted([a for a in admission if a['eligible']],key=lambda z:(z['mae'],-z['rank_ic'],z['feature']))
  picked=[]
  for a in admitted:
   feature=a['feature']
   if picked and frame.loc[screen_keys,picked].corrwith(frame.loc[screen_keys,feature],method='spearman').abs().max()>=.9:continue
   picked.append(feature)
   if len(picked)==3:break
  ensembles=[]
  for k in range(1,len(picked)+1):
   correction=val[picked[:k]].mean(axis=1)
   for w in [.1,.25,.5]:
    pp=np.maximum(0,val.baseline+w*correction)
    ensembles.append(dict(features=picked[:k],weight=w,**gate(pp,val.truth,val.baseline,val.index)))
  eligible=[a for a in ensembles if a['eligible']]
  chosen=min(eligible,key=lambda z:(z['mae'],len(z['features']),z['weight'])) if eligible else dict(features=[],weight=0.)
  decision=dict(outer=str(outer),screen_max_label_end=str(y.loc[screen_keys,'label_end'].max()),earliest_inner=str(earliest),candidates=candidates,
   admission=admission,ensembles=ensembles,selected=chosen,validation_rows=len(val),latest_validation_label_end=str(y.loc[val.index,'label_end'].max()))
  dump(out/'outer'/f'{outer:%Y-%m}_selection.json',decision)
  all_admission += [dict(outer=str(outer),**a) for a in admission]
  print('SELECT',str(outer),'features',len(chosen['features']),'weight',chosen['weight'],flush=True)
  tk=residual_keys(y,base,outer);end=starts[no+1] if no+1<len(starts) else dev.max()+pd.Timedelta(minutes=30)
  keys=dev[(dev>=outer)&(dev<end)]
  test=pd.DataFrame(dict(truth=base.loc[keys,'truth'],baseline=base.loc[keys,'baseline']),index=keys)
  train=pd.DataFrame(dict(truth=base.loc[tk,'truth'],baseline=base.loc[tk,'baseline']),index=tk)
  test_c=[];train_c=[]
  for feature in chosen['features']:
   a=fit_bin(frame.loc[tk,feature],base.loc[tk,'residual'],outer,tk,feature)
   tag=f'{outer:%Y-%m}__outer__'+hashlib.sha256(feature.encode()).hexdigest()[:12]
   dump(out/'corrections'/f'{tag}.json',a);residual_ledger.append(dict(id=tag,status='PASS',outer=str(outer),issuance=str(outer),train_rows=len(tk),max_label_end=str(y.loc[tk,'label_end'].max()),feature=feature,phase='outer'))
   test_c.append(predict_bin(a,frame.loc[keys,feature]));train_c.append(predict_bin(a,frame.loc[tk,feature]))
  for f,arr in [(test,test_c),(train,train_c)]:
   f['selected']=np.maximum(0,f.baseline+chosen['weight']*(np.mean(arr,axis=0) if arr else 0))
   f['outer']=str(outer)
  val_selected=pd.DataFrame(dict(truth=val.truth,baseline=val.baseline),index=val.index)
  val_selected['selected']=np.maximum(0,val.baseline+chosen['weight']*(val[chosen['features']].mean(axis=1) if chosen['features'] else 0))
  val_selected['outer']=str(outer)
  test.to_csv(out/'outer'/f'{outer:%Y-%m}_predictions.csv',index_label='decision_time')
  tests.append(test);trains.append(train);validations.append(val_selected)
  dump(out/'residual_fit_ledger.json',residual_ledger)
 p=pd.concat(tests).sort_index();train=pd.concat(trains);validation=pd.concat(validations)
 assert p.index.equals(dev) and p.index.is_unique
 p.to_csv(out/'predictions.csv',index_label='decision_time');train.to_csv(out/'training_predictions.csv',index_label='decision_time');validation.to_csv(out/'validation_predictions.csv',index_label='decision_time')
 pd.DataFrame(all_admission).to_csv(out/'feature_admission.csv',index=False)
 scores=[]
 for split,f in [('train_residual_in_sample',train),('validation_selected',validation),('test',p)]:
  for name in ['baseline','selected']:scores.append(dict(split=split,model=name,**fw.nb.metrics(f[name],f.truth)))
 pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
 monthly=[dict(month=m,model=c,**fw.nb.metrics(g[c],g.truth)) for m,g in p.groupby(p.index.strftime('%Y-%m')) for c in ['baseline','selected']]
 pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False)
 dump(out/'uncertainty.json',fw.nb.uncertainty(p.rename(columns={'baseline':'report_hybrid'})))
 a=fw.nb.metrics(p.selected,p.truth);b=fw.nb.metrics(p.baseline,p.truth)
 dump(out/'RESULT.json',dict(status='EXECUTED',scores=scores,test_rows=len(p),residual_fits=len(residual_ledger),beats_both=a['mae']<b['mae'] and a['rank_ic']>b['rank_ic'],
  beats_report_mae=(a['mae']<181.5238678328609) if hours==48 else None,training_model_rows=len(train),training_unique_rows=train.index.nunique(),validation_model_rows=len(validation),validation_unique_rows=validation.index.nunique(),
  real_absolute_price_features=prices is not None or returns_path is not None,fresh_confirmation='NOT_RUN: no unexamined data supplied.',other_horizons='Not trained on real data.'))
 print('FINAL\n'+pd.DataFrame(scores).to_string(index=False),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--horizon-hours',type=int,default=48);ap.add_argument('--prices',type=Path);ap.add_argument('--returns-5m',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'Experiments/residual_incremental_v1')
 a=ap.parse_args();run(a.horizon_hours,a.prices,a.out,a.returns_5m)
