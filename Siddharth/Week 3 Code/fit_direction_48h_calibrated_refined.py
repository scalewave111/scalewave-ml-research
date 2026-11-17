"""Focused48h probability refinement with purged chronological OOF calibration."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import fit_direction_48h_classifiers as base
ROOT=base.ROOT;HERE=base.HERE
SPECS=[dict(model=m,pack=k,config=c) for m,k,c in [('hgb','direction_new_state','control'),('hgb','direction_new_state','regularized'),('extra_trees','direction_core','control'),('extra_trees','direction_core','regularized'),('hgb','direction_core','regularized'),('extra_trees','direction_new_state','regularized')]]
MODES=['raw','offset','platt'];WEIGHTS=[.25,.5,.75,1.]
def logit(p):
 p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7);return np.log(p)-np.log1p(-p)
def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-35,35)))
def fitcal(p,y,mode):
 if mode=='raw' or len(y)<200 or len(np.unique(y))<2:return dict(mode='raw',rows=len(y))
 if mode=='offset':
  # Shrink the calibration intercept change by half,without fitting evaluation data.
  shift=.5*(float(logit(np.mean(y)))-float(logit(np.mean(p))))
  return dict(mode=mode,shift=float(np.clip(shift,-.5,.5)),rows=len(y))
 m=LogisticRegression(C=.01,max_iter=300).fit(logit(p).reshape(-1,1),y)
 return dict(mode=mode,slope=float(m.coef_[0,0]),intercept=float(m.intercept_[0]),rows=len(y))
def calibrate(p,c):
 if c['mode']=='raw':return np.asarray(p)
 q=sigmoid(logit(p)+c['shift']) if c['mode']=='offset' else sigmoid(c['slope']*logit(p)+c['intercept'])
 return np.clip(.5*np.asarray(p)+.5*q,1e-7,1-1e-7)
def fit(x,y,cut,spec,packs,cap):
 keys=y.index[y.label_end<cut];base.check(len(keys)>=1000,'Too few training labels');pp=base.prepare(x,keys,packs[spec['pack']]);sample=keys[np.unique(np.linspace(0,len(keys)-1,min(cap,len(keys))).astype(int))]
 target=y.loc[sample,'y_up'];base.check(target.nunique()==2,'Need binary sample')
 reg=spec['config']=='regularized'
 if spec['model']=='hgb':m=HistGradientBoostingClassifier(loss='log_loss',max_iter=60 if reg else 40,max_depth=2 if reg else 3,max_leaf_nodes=4 if reg else 7,min_samples_leaf=128 if reg else 64,l2_regularization=20 if reg else 10,learning_rate=.03 if reg else .05,early_stopping=False,random_state=48)
 else:m=ExtraTreesClassifier(n_estimators=48 if reg else 32,criterion='log_loss',max_depth=4 if reg else 5,min_samples_leaf=64 if reg else 32,max_features=.75,bootstrap=True,max_samples=min(1024,len(sample)),n_jobs=1,random_state=48)
 m.fit(base.design(x,sample,pp,spec['model']),target)
 return dict(model=m,prep=pp,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())
def choose(trials):
 key=lambda t:(t['score'],t['id'],t['mode'],t['weight']);best=min(trials,key=key);ok=[t for t in trials if t['accepted']]
 return best,min(ok,key=key) if ok else None

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data-dir',type=Path,default=ROOT.parent/'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data')
 parser.add_argument('--direction-template',type=Path,default=ROOT.parent/'Week 2/Code/neural_finetune_v45/neural_finetune_v45/final_pipeline.json')
 parser.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
 parser.add_argument('--canonical-manifest',type=Path,default=ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json')
 parser.add_argument('--row-cap',type=int,default=2000);parser.add_argument('--threads',type=int,default=2);parser.add_argument('--check-inputs',action='store_true')
 args=parser.parse_args();base.check(args.row_cap>=1000 and args.threads>0,'Invalid cap/threads')
 with threadpool_limits(limits=args.threads):
  x,y,dev,packs,defs,pop,files,audit=base.load(args)
  if args.check_inputs:print(json.dumps(audit,indent=2,default=str));return
  ref=HERE/'experiments/direction_48h_classifiers_v1/20261116T130206171832Z';done=json.loads((ref/'COMPLETE.json').read_text());base.check(done['status']=='COMPLETE' and base.sha(ref/'predictions.csv')==done['prediction_sha256'],'Reference hash failed')
  previous=pd.read_csv(ref/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True);base.check(previous.index.is_unique and previous.index.equals(dev.sort_values()),'Reference keys differ')
  stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');out=HERE/'experiments/direction_48h_calibrated_refined_v1'/stamp;report=ROOT/'Report/direction_48h_calibrated_refined_v1'/stamp;out.mkdir(parents=True);report.mkdir(parents=True)
  files += [Path(__file__),Path(base.__file__),ref/'COMPLETE.json',ref/'predictions.csv']
  base.dump(out/'CONTRACT.json',dict(arguments=vars(args),input_audit=audit,specs=SPECS,modes=MODES,weights=WEIGHTS,calibration='For validation blockj,fitcal only previous OOF blocks with label_end<currentcut. Firstblockidentity. Finalcal uses all4OOFblocks maturebeforeouterissue. Offset half logitrate difference,capped.5;PlattL2C.01,raw/cal50-50blend. No outerlabels.',selection='Originaljoint relativelogloss/Brierscore andguard unchanged:.1%gainboth,3/4wins/ties,no>2%windowloss,prevalencefallback. Candidate separate.',input_hashes={str(f):base.sha(f) for f in files},limitations='Exploratory focus motivated by exposed results. No automatic promotion of earlier0.686candidate. No untouchedtest,guaranteedgain orresume.96innerfits,up to8outerfits;calibrationfits additional.'))
  defs.to_csv(out/'new_direction_feature_formulas.csv',index=False);pop.to_csv(out/'population_audit.csv',index_label='decision_time')
  frames=[];plans=[];ledger=[];cal_audit=[]
  for issue in base.ISSUES:
   windows=[]
   for offset in [15,12,9,6]:
    cut=issue-pd.DateOffset(months=offset);end=cut+pd.DateOffset(months=3);va=y.index[(y.index>=cut)&(y.index<end)&(y.label_end<end)&(y.label_end<issue)];base.check(len(va)>=100,'Too few validation rows');windows.append((cut,va))
   ys=[y.loc[va,'y_up'].to_numpy() for cut,va in windows];priors=[float(y.loc[y.label_end<cut,'y_up'].mean()) for cut,va in windows];trials=[];oofs={}
   for i,spec in enumerate(SPECS):
    probs=[]
    for j,(cut,va) in enumerate(windows):
     print(issue.date(),i+1,'/6',spec,'window',j+1,'/4',flush=True)
     a=fit(x,y,cut,spec,packs,args.row_cap);base.check(a['max_label_end']<va.min(),'Purgefailed');probs.append(base.predict(a,x,va))
    oofs[i]=probs
    for mode in MODES:
     preds=[]
     for j,(cut,va) in enumerate(windows):
      if j:
       label_end=pd.to_datetime(np.concatenate([y.loc[v,'label_end'].to_numpy() for _,v in windows[:j]]),utc=True);mask=label_end<cut
       pp=np.concatenate(probs[:j])[mask];yy=np.concatenate(ys[:j])[mask]
       c=fitcal(pp,yy,mode);base.check(not mask.any() or label_end[mask].max()<cut,'Calibrationpurgefailed')
      else:c=dict(mode='raw',rows=0)
      cal_audit.append(dict(issue=str(issue),id=i,requested_mode=mode,window=j,fit_cut=str(cut),calibration=c));preds.append(calibrate(probs[j],c))
     for w in WEIGHTS:
      trial=dict(issue=str(issue),id=i,spec=spec,mode=mode,weight=w,**base.assess(ys,[w*p+(1-w)*b for p,b in zip(preds,priors)],priors));trials.append(trial);ledger.append(trial)
    base.dump(out/'validation_trials.json',ledger);base.dump(out/'calibration_audit.json',cal_audit)
   best,chosen=choose(trials);plan=dict(issue=str(issue),candidate=best,selected=chosen,models=[],calibrators={});plans.append(plan);base.dump(out/'selections.json',plans)
   end=base.ISSUES[list(base.ISSUES).index(issue)+1] if issue!=base.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1);te=dev[(dev>=issue)&(dev<end)];prior=float(y.loc[y.label_end<issue,'y_up'].mean())
   f=pd.DataFrame(dict(y_up=y.loc[te,'y_up'],signed_return_log_bps=y.loc[te,'signed_return_log_bps'],issue=issue,prevalence=prior,fair=.5),index=te);fitted={}
   for t in [best]+([] if chosen is None else [chosen]):
    i=t['id']
    if i not in fitted:
     a=fit(x,y,issue,SPECS[i],packs,args.row_cap);path=out/f'model_{issue:%Y%m}_{i}.joblib';joblib.dump(a,path,compress=3);q=base.predict(a,x,te);np.testing.assert_allclose(q,base.predict(joblib.load(path),x,te),rtol=0,atol=1e-8);fitted[i]=q;plan['models'].append(dict(artifact=path.name,sha256=base.sha(path),features=a['prep']['cols'],spec=SPECS[i],max_label_end=a['max_label_end']))
    # Earlier OOF labels only; every source block was purged at its own end.
    base.check(all(y.loc[v,'label_end'].max()<issue for _,v in windows),'Finalcalibrationmaturityfailed')
    c=fitcal(np.concatenate(oofs[i]),np.concatenate(ys),t['mode']);plan['calibrators'][str(i)+'_'+t['mode']]=c
   def forecast(t):
    c=plan['calibrators'][str(t['id'])+'_'+t['mode']];return t['weight']*calibrate(fitted[t['id']],c)+(1-t['weight'])*prior
   f['candidate']=forecast(best);f['p_up']=prior if chosen is None else forecast(chosen);f['p_down']=1-f.p_up;f['predicted_direction']=np.where(f.p_up>=.5,1,-1);f['actual_direction']=np.sign(f.signed_return_log_bps).astype(int)
   frames.append(f);base.dump(out/'selections.json',plans)
  result=pd.concat(frames).sort_index();base.check(result.index.equals(dev.sort_values()),'Coveragemismatch')
  for col in ['y_up','signed_return_log_bps','prevalence']:np.testing.assert_allclose(result[col],previous[col],atol=1e-8,rtol=0)
  result['previous_selected']=previous.p_up;result['previous_candidate']=previous.candidate;result.to_csv(out/'predictions.csv',index_label='decision_time')
  rows=[]
  for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
   for col in ['prevalence','fair','previous_selected','previous_candidate','p_up','candidate']:rows.append(dict(period=period,model=col,**base.score(g.y_up,g[col])))
  metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False);metrics.to_csv(report/'metrics.csv',index=False);text='48h direction focusedcalibration;guardedp_up vs diagnosticcandidate;historicalexposed evaluation.\n'+metrics.to_string(index=False);(report/'SUMMARY.txt').write_text(text,encoding='utf-8');base.dump(report/'selections.json',plans)
  for p in [Path(__file__),Path(base.__file__)]: (out/p.name).write_bytes(p.read_bytes())
  base.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True));print(text,'\nOutputs:',out,'\nReport:',report,flush=True)
if __name__=='__main__':main()
