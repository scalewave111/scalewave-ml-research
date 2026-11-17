"""User-run 120h net magnitude: focused ExtraTrees loss and shrinkage refinement."""
from pathlib import Path
import argparse, json, sys, importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import fit_net_magnitude_120h as base

CONFIGS={'small':([32,16],40,.01),'medium':([64,32],60,.01),'regularized':([32],60,.1)}
WEIGHTS=[.05,.1,.25,.4,.5,.6,.75,1.]
FAST=False
REFERENCES={'hgb':'20261116T101322630960Z','extra':'20261116T101506990782Z','mlp':'20261116T101737342721Z','gru':'20261116T101825668821Z'}

class MAEMLP:
    def __init__(self,config):self.config=config
    def fit(self,x,y):
        import torch
        torch.manual_seed(48);torch.use_deterministic_algorithms(True)
        hidden,epochs,decay=CONFIGS[self.config];dims=[x.shape[1]]+hidden+[1];layers=[]
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2:layers.append(torch.nn.ReLU())
        model=torch.nn.Sequential(*layers)
        with torch.no_grad():model[-1].bias.fill_(float(np.median(y)))
        xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32).reshape(-1,1)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=decay)
        generator=torch.Generator().manual_seed(48)
        model.train()
        for epoch in range(epochs):
            for batch in torch.randperm(len(y),generator=generator).split(256):
                opt.zero_grad(set_to_none=True);loss=torch.nn.functional.l1_loss(model(xx[batch]),yy[batch])
                base.check(bool(torch.isfinite(loss)),'Nonfinite MLP loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        self.weights=[(m.weight.detach().numpy().copy(),m.bias.detach().numpy().copy()) for m in model if isinstance(m,torch.nn.Linear)]
        return self
    def predict(self,x):
        import torch
        z=torch.tensor(x,dtype=torch.float32)
        with torch.no_grad():
            for i,(w,b) in enumerate(self.weights):
                z=torch.nn.functional.linear(z,torch.from_numpy(w),torch.from_numpy(b))
                if i<len(self.weights)-1:z=torch.relu(z)
        return z.numpy().ravel().astype(float)

class FastGRU(base.GRURegressor):
    def fit(self,x,y):
        import torch
        torch.manual_seed(48)
        self.inputs=x.shape[-1];net=self.network(self.inputs)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
        xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32)
        net.train()
        for epoch in range(12):
            for start in range(0,len(xx),256):
                optimizer.zero_grad()
                loss=torch.nn.functional.l1_loss(net(xx[start:start+256]),yy[start:start+256])
                base.check(bool(torch.isfinite(loss)),'Nonfinite GRU loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        self.state={k:v.detach().clone() for k,v in net.state_dict().items()}
        return self

def configure_fast(args):
    global FAST,CONFIGS
    FAST=args.fast
    if FAST:
        CONFIGS={'regularized':([32],16,.1)}
        args.row_cap=min(args.row_cap,2000)


def fit(x,y,cut,spec,packs,cap):
    if not FAST and spec['model'] in ['hgb','gru']:return base.fit(x,y,cut,spec,packs,cap)
    keys=y.index[y.label_end<cut];base.check(len(keys)>=100,'Insufficient mature training rows')
    pp=base.prepare(x,keys,packs[spec['pack']]);sample=keys[np.unique(np.linspace(0,len(keys)-1,min(cap,len(keys))).astype(int))]
    scale=max(float(y.loc[keys,base.TARGET].median()),1.)
    center=0.
    if FAST and spec['model'] in ['hgb','gru']:
        center=float(y.loc[keys,base.TARGET].median());scale=max(float(y.loc[keys,base.TARGET].std()),1.)
        model=(FastGRU() if spec['model']=='gru' else HistGradientBoostingRegressor(loss='absolute_error',max_iter=40,max_leaf_nodes=7,max_depth=3,min_samples_leaf=64,l2_regularization=5,early_stopping=False,random_state=48))
    else:
        model=(MAEMLP(spec['config']) if spec['model']=='mlp' else ExtraTreesRegressor(n_estimators=24 if FAST else 64,criterion=('squared_error' if spec['config']=='squared' else 'absolute_error'),max_depth=4 if FAST else 6,min_samples_leaf=32,max_features=.75,bootstrap=True,max_samples=min(1024 if FAST else 4096,len(sample)),random_state=48,n_jobs=1))
    model.fit(base.model_design(x,sample,pp,spec['model']),(y.loc[sample,base.TARGET].to_numpy()-center)/scale)
    return dict(prep=pp,model=model,center=center,scale=scale,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())

def blend(pred,anchor,weight):return weight*np.asarray(pred)+(1-weight)*anchor

def accept(errors,refs):
    # Both baseline MAEs, plus mean-baseline RMSE, using earlier windows only.
    if not all(base.gate(errors,refs[b]) for b in refs):return False
    mse=np.array([np.mean(e**2) for e in errors])
    baseline=np.array([np.mean(e**2) for e in refs['mean']])
    pooled=np.mean(np.concatenate(errors)**2)
    pooled_ref=np.mean(np.concatenate(refs['mean'])**2)
    return bool(pooled<=(.999**2)*pooled_ref and (mse<=baseline).sum()>=2 and (mse<=1.02**2*baseline).all())


def choose(trials):
    best=min(trials,key=lambda t:(t['mae'],t['id'],t['anchor'],t['weight']))
    accepted=[t for t in trials if t['accepted']]
    return best,min(accepted,key=lambda t:(t['mae'],t['id'],t['anchor'],t['weight'])) if accepted else None

def run_family(x,y,dev,packs,family,args,files,audit,population):
    stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    name='net_magnitude_120h_extra_refined_v1'
    out=base.HERE/'experiments'/name/stamp;report=base.ROOT/'Report'/name/stamp
    out.mkdir(parents=True);report.mkdir(parents=True)
    ref=base.HERE/'experiments'/f'net_magnitude_120h_{family}_only_v1'/REFERENCES[family]
    done=json.loads((ref/'COMPLETE.json').read_text());base.check(done['status']=='COMPLETE' and base.sha(ref/'predictions.csv')==done['prediction_sha256'],'Reference integrity failure')
    previous=pd.read_csv(ref/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True)
    base.check(previous.index.is_unique and previous.index.equals(dev.sort_values()),'Reference timestamp mismatch')
    pd.read_csv(ref/'new_feature_formulas.csv').to_csv(out/'new_feature_formulas.csv',index=False)
    specs=[dict(model=family,pack=pack,config=config) for pack in packs for config in ['control','squared']]
    base.dump(out/'CONTRACT.json',dict(horizon_minutes=7200,target=base.TARGET,formula=base.FORMULA,family=family,arguments=vars(args),input_audit=audit,packs=packs,specs=specs,weights=WEIGHTS,anchors=['median','mean'],fallback='mean',loss_configs=['control_absolute_error','squared_error'],seed=48,configs=CONFIGS,fast_settings=(dict(packs=list(packs),row_cap=args.row_cap,hgb_iterations=40,extra_trees=24,extra_depth=4,extra_bootstrap_cap=1024,mlp_epochs=16,gru_epochs=12) if FAST else None),
        selection='Three earlier two-month windows at offsets9/6/3months. Pooled MAE rank. Require >=0.1% MAE improvement against BOTH baseline MAEs; >=2/3 window wins/ties and no >2% window loss against EACH baseline. Otherwise training mean. Diagnostic candidate separate. Also require >=0.1% pooled RMSE improvement vs mean, 2/3 RMSE wins/ties and no >2% RMSE window loss. Rank accepted choices by MAE.',
        adaptation='Focused ExtraTrees control absolute-error versus squared-error splits; finer shrinkage to training mean/median. Two packs, 2000 cap and fast budgets retained; 48 inner fits. All preprocessing fitted on mature training only.',
        versions={v:importlib.metadata.version(v) for v in ['numpy','pandas','scikit-learn','torch','joblib']},input_hashes={str(f):base.sha(f) for f in sorted(set(files))},limitations='Historical adaptive exposed evaluation. Not exact UP reproduction; no guaranteed win or fresh test. No resume.'))
    population.to_csv(out/'population_audit.csv',index_label='decision_time')
    frames=[];selections=[];ledger=[]
    for issue in base.ISSUES:
        windows=[]
        for offset in [9,6,3]:
            cut=issue-pd.DateOffset(months=offset)
            va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&(y.label_end<issue)]
            base.check(len(va)>=100,'Too few validation rows')
            tr=y.loc[y.label_end<cut,base.TARGET]
            windows.append((cut,va,float(tr.median()),float(tr.mean())))
        truths=[y.loc[va,base.TARGET].to_numpy() for cut,va,md,mn in windows]
        refs={anchor:[np.full(len(va),md if anchor=='median' else mn)-truth for (cut,va,md,mn),truth in zip(windows,truths)] for anchor in ['median','mean']}
        trials=[]
        for i,spec in enumerate(specs):
            predictions=[]
            for window_id,(cut,va,md,mn) in enumerate(windows,1):
                print(family,issue.date(),f'candidate {i+1}/{len(specs)}, window {window_id}/3: fitting',flush=True)
                a=fit(x,y,cut,spec,packs,args.row_cap)
                base.check(a['max_label_end']<va.min(),'Training purge failed')
                predictions.append(base.predict(a,x,va))
            for anchor in ['median','mean']:
                for weight in WEIGHTS:
                    errors=[blend(pred,md if anchor=='median' else mn,weight)-truth for pred,(cut,va,md,mn),truth in zip(predictions,windows,truths)]
                    record=dict(issue=str(issue),id=i,spec=spec,anchor=anchor,weight=weight,mae=float(np.abs(np.concatenate(errors)).mean()),accepted=accept(errors,refs),window_rmse=[float(np.sqrt(np.mean(e**2))) for e in errors],window_mae=[float(np.abs(e).mean()) for e in errors])
                    trials.append(record);ledger.append(record)
            print(family,issue.date(),i+1,'/',len(specs),spec,flush=True)
            base.dump(out/'validation_trials.json',ledger)
        best,chosen=choose(trials)
        # Freeze choices before evaluating any outer outcomes.
        selection=dict(issue=str(issue),candidate=best,selected=chosen,fallback=chosen is None,models=[])
        selections.append(selection);base.dump(out/'selections.json',selections)
        end=base.ISSUES[list(base.ISSUES).index(issue)+1] if issue!=base.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
        te=dev[(dev>=issue)&(dev<end)];base.check(len(te)>0,'Empty outer period')
        tr=y.loc[y.label_end<issue,base.TARGET];anchors={'median':float(tr.median()),'mean':float(tr.mean())}
        f=pd.DataFrame({'actual':y.loc[te,base.TARGET],'issue':issue,**anchors},index=te)
        fitted={}
        for record in [best]+([] if chosen is None else [chosen]):
            i=record['id']
            if i not in fitted:
                a=fit(x,y,issue,specs[i],packs,args.row_cap);path=out/f'model_{issue:%Y%m}_{i}.joblib';joblib.dump(a,path,compress=3)
                pred=base.predict(a,x,te);np.testing.assert_allclose(pred,base.predict(joblib.load(path),x,te),atol=1e-8,rtol=0)
                fitted[i]=pred;selection['models'].append(dict(artifact=path.name,sha256=base.sha(path),features=a['prep']['cols'],spec=specs[i],training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end']))
        f['candidate']=blend(fitted[best['id']],anchors[best['anchor']],best['weight'])
        f['selected']=anchors['mean'] if chosen is None else blend(fitted[chosen['id']],anchors[chosen['anchor']],chosen['weight'])
        frames.append(f);base.dump(out/'selections.json',selections)
    result=pd.concat(frames).sort_index();base.check(result.index.equals(dev.sort_values()),'Coverage mismatch')
    # Original saved forecasts are comparisons only; never inputs to selection.
    for col in ['actual','mean','median']:np.testing.assert_allclose(result[col],previous[col],atol=1e-8,rtol=0)
    control=base.HERE/'experiments/net_magnitude_120h_extra_blended_fast_v1/20261116T104801609018Z'
    receipt=json.loads((control/'COMPLETE.json').read_text())
    base.check(base.sha(control/'predictions.csv')==receipt['prediction_sha256'],'Fast control hash mismatch')
    fast=pd.read_csv(control/'predictions.csv',index_col='decision_time');fast.index=pd.to_datetime(fast.index,utc=True)
    base.check(fast.index.is_unique and fast.index.equals(result.index),'Fast control alignment mismatch')
    for col in ['actual','mean','median']:np.testing.assert_allclose(result[col],fast[col],atol=1e-8,rtol=0)
    result['previous_fast']=fast['selected']
    result['previous']=previous[family]
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    rows=[]
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for model in ['median','mean','previous','previous_fast','selected','candidate']:rows.append(dict(period=period,model=model,**base.metrics(g.actual,g[model])))
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False);table.to_csv(report/'metrics.csv',index=False)
    text='120h net magnitude: '+family+' independent blended search\n'+table.to_string(index=False)+'\nCandidate is diagnostic; selected may fall back to mean. Historical exposed evaluation.\n'
    (report/'SUMMARY.txt').write_text(text);base.dump(report/'selections.json',selections)
    for file in [Path(__file__),Path(base.__file__)]: (out/file.name).write_bytes(file.read_bytes())
    base.dump(out/'COMPLETE.json',dict(status='COMPLETE',target=base.TARGET,rows=len(result),prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True,reference_prediction_sha256=base.sha(ref/'predictions.csv')))
    print(text,'\nOutputs:',out,'\nReport:',report,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=['extra'],default=['extra'])
    parser.add_argument('--week2-code',type=Path,default=base.ROOT.parent/'Week 2/Code')
    parser.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    parser.add_argument('--canonical-manifest',type=Path,default=base.ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json',help='Alternate canonical checksum manifest; defaults to verified local receipt')
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--row-cap',type=int,default=6000)
    parser.add_argument('--fast',action='store_true',default=True,help='2 packs, one regularized MLP, 2000-row cap and reduced estimator budgets')
    parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args();base.check(args.threads>0 and args.row_cap>=256,'Invalid thread/cap')
    configure_fast(args)
    import torch
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        x,y,dev,population,packs,definitions,files,audit=base.load(args)
        if FAST:packs={k:packs[k] for k in ['market_new','all']}
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));return
        files += [Path(__file__),Path(base.__file__),base.ROOT/'PROJECT_CONTEXT.md']
        control=base.HERE/'experiments/net_magnitude_120h_extra_blended_fast_v1/20261116T104801609018Z'
        files += [control/'COMPLETE.json',control/'predictions.csv',control/'CONTRACT.json']
        for family in args.families:
            ref=base.HERE/'experiments'/f'net_magnitude_120h_{family}_only_v1'/REFERENCES[family]
            files += [ref/'COMPLETE.json',ref/'predictions.csv',ref/'CONTRACT.json']
        framework=base.ROOT.parents[1]/'ml_framework'
        files += [framework/v for v in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        # Population is written separately, not serialized into CLI arguments.
        for family in args.families:
            run_family(x,y,dev,packs,family,args,files,audit,population)

if __name__=='__main__':main()
