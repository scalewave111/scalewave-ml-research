"""Run locally: two MAE learners, standalone and pair blend, fixed 48h protocol."""
from pathlib import Path
import hashlib,json,os,sys,time,threading
import joblib,numpy as np,pandas as pd
from scipy.stats import spearmanr
from feature_experiment import load,splits,transform,anchors,pair_prediction
from model_trial_helpers import fit,predict,choose,SETTINGS,SEEDS

HERE=Path(__file__).resolve().parent;SUB=HERE.parent;ROOT=SUB.parent
OUT=HERE/'model_trial_run';CURRENT=SUB/'magnitude_current/run_001'

def read(p):return json.loads(p.read_text())
def write(p,value):
 if p.exists():p.read_text()
 tmp=p.with_suffix(p.suffix+'.tmp')
 if tmp.exists():tmp.read_text()
 tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def dump(p,value):
 if p.exists():joblib.load(p)
 tmp=p.with_suffix('.partial');joblib.dump(value,tmp);tmp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inputs(c):
 files=[Path(__file__),HERE/'model_trial_helpers.py',HERE/'feature_experiment.py',CURRENT/'config.json',CURRENT/'predictions.parquet']
 files += [ROOT/part[k] for part in c['data_parts'] for k in ['X','y']]
 files += [ROOT/c['anchor_profile']/f'fold_{f}_inner.json' for f in range(1,5)]
 files += [ROOT/c['outer_pair_profile']/f'fold_{f}_pair.json' for f in range(1,5)]
 return {str(p.relative_to(ROOT)):digest(p) for p in files}
def unchanged(hashes):
 for name,h in hashes.items():assert digest(ROOT/name)==h,f'Input changed: {name}'
def preflight():
 c=read(CURRENT/'config.json');reg=read(SUB/'magnitude_benchmark_registry.json')
 assert c['horizon_minutes']==2880 and c['seeds']==SEEDS
 assert len(c['anchor_features'])+len(c['additions'])==7
 X,y=load(c,True);base=pd.read_parquet(CURRENT/'predictions.parquet')
 assert digest(CURRENT/'predictions.parquet')==reg['current']['prediction_sha256']
 np.testing.assert_array_equal(base.actual,y.loc[base.index,c['target_column']])
 assert abs(float((base.actual-base.prediction).abs().mean())-reg['current']['MAE'])<1e-12
 indices=[]
 for fold,ot,ov,it,iv in splits(c,y):
  indices.extend(ov)
  for idx in [ot,it]:assert all(X.loc[idx,f].count()>=1000 and X.loc[idx,f].nunique()>1 for f in X)
  _,outer=anchors(c,fold)
  np.testing.assert_allclose(pair_prediction(X.loc[ov,c['anchor_features']],outer),base.loc[ov,'pair'],atol=1e-10,rtol=0)
 assert pd.DatetimeIndex(indices).equals(base.index)
 print(f"SETUP PASSED: {len(base)} rows; seven features; benchmark MAE {reg['current']['MAE']:.9f}",flush=True)
 return c,reg,X,y,base
def metric(y,p):return {'MAE':float(np.abs(y-p).mean()),'Spearman_IC':float(spearmanr(y,p).statistic)}
def cleanup(result):
 keep=set(result['retained_model_files'])
 assert OUT.resolve().parent==HERE.resolve() and not OUT.is_symlink()
 for p in OUT.iterdir():
  assert p.is_file() and not p.is_symlink() and p.resolve().parent==OUT.resolve()
  if (p.suffix in ['.joblib','.partial'] and p.name not in keep):p.unlink()
 if not result['beats_incumbent'] and (OUT/'predictions.parquet').exists():(OUT/'predictions.parquet').unlink()
def replay_result(result,c,X,y):
 if not result['beats_incumbent']:return
 saved=pd.read_parquet(OUT/'predictions.parquet');allpred=[]
 family,mode=result['best_candidate'].split(':')
 for fold,ot,ov,it,iv in splits(c,y):
  card=result['folds'][f'{family}:{fold}'];sel=card['selection'][mode]
  B,_=transform(X.loc[ov],card['outer_preprocessing'])
  models=[joblib.load(OUT/f'{family}_f{fold}_s{sel["setting"]}_outer_seed{seed}.joblib')['bundle'] for seed in SEEDS]
  estimate=np.mean([predict(m,B) for m in models],axis=0)
  _,pair=anchors(c,fold);p=pair_prediction(X.loc[ov,c['anchor_features']],pair)
  allpred.append(pd.Series((1-sel['weight'])*p+sel['weight']*estimate,index=ov))
 replayed=pd.concat(allpred);assert replayed.index.equals(saved.index)
 np.testing.assert_array_equal(saved.actual,y.loc[saved.index,c['target_column']])
 np.testing.assert_allclose(replayed,saved.prediction,atol=1e-9,rtol=0)
 assert abs(metric(saved.actual,replayed)['MAE']-result['scores'][result['best_candidate']]['MAE'])<1e-9
def main():
 c,reg,X,y,base=preflight();hashes=inputs(c)
 if '--check' in sys.argv:print('No real-data training started.');return
 OUT.mkdir(exist_ok=True)
 if (OUT/'RESULT.json').exists():
  result=read(OUT/'RESULT.json');assert result['status']=='VERIFIED';unchanged(result['input_hashes'])
  replay_result(result,c,X,y);cleanup(result);print((OUT/'MODEL_RESULTS.txt').read_text());return
 if (OUT/'contract.json').exists():
  contract=read(OUT/'contract.json');assert contract['input_hashes']==hashes
 else:
  assert not list(OUT.iterdir()),'Unexpected output exists; request review instead of deleting it.'
  contract={'input_hashes':hashes,'baseline':reg['current'],'config':c,'settings':SETTINGS,'seeds':SEEDS,
   'NN':'Two ReLU hidden layers; Softplus output; MAE loss; train-only median target scale; Adam lr=.001 weight_decay=.001; batch512; maximum80 epochs; patience10 using inner validation; outer epochs fixed from inner.',
   'HGB':'absolute_error; max_iter200; lr=.05; leaves7 or15; min_leaf64; l2=1; no internal random validation.',
   'selection':'Within each family, select standalone setting by inner MAE; separately select blend setting and weight from 0,.1,.25,.5. Mean of seeds48,17,91. Never select using outer labels.',
   'scope':'Fixed original four folds and seven features. Fresh data untouched. Current incumbent is preserved; a verified winning challenger is retained for review, not automatically promoted.'}
  write(OUT/'contract.json',contract)
 receipt=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'folds':{},'fits':{}}
 frames=[]
 def fitted(family,setting,seed,fold,role,A,ty,prep,B=None,vy=None,epochs=None):
  name=f'{family}_f{fold}_s{setting}_{role}_seed{seed}.joblib';path=OUT/name
  if path.exists():
   payload=joblib.load(path);assert payload['preprocessing']==prep
   model=payload['bundle'];assert model['seed']==seed and model['settings']==SETTINGS[family][setting] and model['family']==family
   if epochs is not None:assert model['epochs']==epochs
   print('Reusing saved fit:',name,flush=True)
  else:
   print(f'FIT {family}, period {fold}/4, setting {setting+1}/2, {role}, seed {seed}',flush=True)
   began=time.monotonic();model=fit(family,SETTINGS[family][setting],seed,A,ty,B,vy,epochs)
   payload={'bundle':model,'preprocessing':prep,'fit_seconds':time.monotonic()-began,'training_rows':len(ty)};dump(path,payload)
  receipt['fits'][name]={'seconds':payload['fit_seconds'],'rows':payload['training_rows'],'epochs':model.get('epochs'),'epochs_run':model.get('epochs_run')}
  write(OUT/'progress.json',receipt)
  return joblib.load(path)['bundle']
 for fold,ot,ov,it,iv in splits(c,y):
  target=c['target_column'];ia,iprep=transform(X.loc[it]);ib,_=transform(X.loc[iv],iprep)
  oa,oprep=transform(X.loc[ot]);ob,_=transform(X.loc[ov],oprep)
  ic,oc=anchors(c,fold);pair_i=pair_prediction(X.loc[iv,c['anchor_features']],ic);pair_o=pair_prediction(X.loc[ov,c['anchor_features']],oc)
  frame=pd.DataFrame({'actual':y.loc[ov,target],'incumbent':base.loc[ov,'prediction'],'pair':pair_o,'fold':fold},index=ov)
  for family in SETTINGS:
   inner=[];models={}
   for setting in range(2):
    ms=[fitted(family,setting,seed,fold,'inner',ia,y.loc[it,target].to_numpy(),iprep,ib,y.loc[iv,target].to_numpy()) for seed in SEEDS]
    models[setting]=ms;inner.append(np.mean([predict(m,ib) for m in ms],axis=0))
   selection,scores=choose(inner,y.loc[iv,target].to_numpy(),pair_i)
   estimates={}
   for setting in sorted({v['setting'] for v in selection.values()}):
    ms=[fitted(family,setting,seed,fold,'outer',oa,y.loc[ot,target].to_numpy(),oprep,
     epochs=models[setting][k].get('epochs')) for k,seed in enumerate(SEEDS)]
    estimates[setting]=np.mean([predict(m,ob) for m in ms],axis=0)
   receipt['folds'][f'{family}:{fold}']={'selection':selection,'inner_scores':scores,'outer_preprocessing':oprep,
    'outer_training_rows':len(ot),'inner_training_rows':len(it),'outer_max_label_end':str(y.loc[ot,c['label_end_column']].max()),'inner_max_label_end':str(y.loc[it,c['label_end_column']].max()),
    'inner_validation_start':str(iv.min()),'outer_validation_start':str(ov.min())}
   for mode,sel in selection.items():frame[f'{family}:{mode}']=(1-sel['weight'])*pair_o+sel['weight']*estimates[sel['setting']]
   write(OUT/'progress.json',receipt)
  frames.append(frame);unchanged(hashes);print(f'PERIOD {fold}/4 COMPLETE',flush=True)
 pred=pd.concat(frames);assert pred.index.equals(base.index);assert np.isfinite(pred.to_numpy()).all()
 cols=[f'{f}:{mode}' for f in SETTINGS for mode in ['standalone','blend']]
 scores={n:metric(pred.actual,pred[n]) for n in ['incumbent']+cols}
 best=min(cols,key=lambda n:(scores[n]['MAE'],n));beats=scores[best]['MAE']<scores['incumbent']['MAE']-1e-12
 family,mode=best.split(':');retained=[]
 if beats:
  for fold in range(1,5):
   sel=receipt['folds'][f'{family}:{fold}']['selection'][mode]
   retained += [f'{family}_f{fold}_s{sel["setting"]}_outer_seed{seed}.joblib' for seed in SEEDS]
  p=pred[['actual','incumbent','pair','fold',best]].rename(columns={best:'prediction'})
  p.to_parquet(OUT/'predictions.partial');(OUT/'predictions.partial').replace(OUT/'predictions.parquet')
 fold_scores={str(f):{n:metric(g.actual,g[n]) for n in ['incumbent']+cols} for f,g in pred.groupby('fold')}
 result={'status':'VERIFIED','input_hashes':hashes,'scores':scores,'fold_scores':fold_scores,'folds':receipt['folds'],
  'best_candidate':best,'beats_incumbent':beats,'retained_model_files':retained,'rows':len(pred),'completed_fits':len(receipt['fits']),
  'verification':'Train-only preprocessing and strict label-end purges; inner-only parameter, epoch and blend selection; saved outer models reloaded; exact timestamps and targets; immutable source/data hashes.',
  'interpretation':'Adaptive development comparison, not fresh confirmation. Incumbent and its registry are unchanged. Retained winning challenger needs review/integration before promotion.'}
 delta=(pred.actual-pred[best]).abs()-(pred.actual-pred.incumbent).abs()
 result['nominal_block_intervals']=[]
 for days in [7,14]:
  groups=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int)
  sums=delta.groupby(groups).sum().to_numpy();sizes=delta.groupby(groups).size().to_numpy()
  draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)))
  values=sums[draws].sum(1)/sizes[draws].sum(1)
  result['nominal_block_intervals'].append({'days':days,'delta_MAE':float(delta.mean()),'interval':np.quantile(values,[.025,.975]).tolist(),'caution':'Not adjusted for repeated research selection.'})
 # Independently replay all four scored candidates from disk before pruning.
 for fold,ot,ov,it,iv in splits(c,y):
  for family in SETTINGS:
   card=receipt['folds'][f'{family}:{fold}'];B,_=transform(X.loc[ov],card['outer_preprocessing'])
   for mode,sel in card['selection'].items():
    models=[joblib.load(OUT/f'{family}_f{fold}_s{sel["setting"]}_outer_seed{seed}.joblib')['bundle'] for seed in SEEDS]
    p=np.mean([predict(m,B) for m in models],axis=0)
    np.testing.assert_allclose((1-sel['weight'])*pred.loc[ov,'pair']+sel['weight']*p,pred.loc[ov,f'{family}:{mode}'],atol=1e-9,rtol=0)
 replay_result(result,c,X,y);unchanged(hashes)
 lines=['MODEL COMPARISON COMPLETE - VERIFIED','']
 for n,m in scores.items():lines.append(f"{n}: MAE {m['MAE']:.9f}; Spearman {m['Spearman_IC']:.9f}; improvement {scores['incumbent']['MAE']-m['MAE']:+.9f} log bps")
 lines+=['',f'Best challenger: {best}',f'Beats incumbent: {beats}',f'Completed fits: {len(receipt["fits"])}',
  'Incumbent model and registry remain unchanged. Winning challenger retained for review if better.',
  'Unused model files pruned; compact evidence retained. Fresh data untouched.',
  'This is development performance; further independent confirmation is needed.']
 text='\n'.join(lines);p=OUT/'MODEL_RESULTS.txt'
 if p.exists():p.read_text()
 p.write_text(text);write(OUT/'RESULT.json',result);cleanup(result);print(text,flush=True)

if __name__=='__main__':
 import msvcrt
 with (HERE/'model_trial.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0)
  try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  except OSError:raise SystemExit('Another model comparison is already running.')
  stop=threading.Event()
  def heartbeat():
   while not stop.wait(60):print('Still running; completed fits are saved.',flush=True)
  threading.Thread(target=heartbeat,daemon=True).start()
  try:main()
  except KeyboardInterrupt:print('\nStopped. Run the same command to resume. An unfinished fit restarts.');sys.exit(130)
  except Exception as e:print(f'\nSTOPPED: {type(e).__name__}: {e}. Keep files and share this error.');raise
  finally:stop.set()
