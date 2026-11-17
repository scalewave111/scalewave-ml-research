"""Focused bounded120h refinement: fixed small-network controls plus nearby alternatives."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from threadpoolctl import threadpool_limits
import fit_net_magnitude_120h_distribution_search as engine
base=engine.base
import fit_net_magnitude_120h_controlled as control
SETTINGS={'tiny8':dict(hidden=8,decay=.5,epochs=12,bound=.25),
          'strong16':dict(hidden=16,decay=1.,epochs=12,bound=.25),
          'tight16':dict(hidden=16,decay=1.,epochs=12,bound=.10),
          'wide16':dict(hidden=16,decay=1.,epochs=12,bound=.40),
          'long16':dict(hidden=16,decay=1.,epochs=24,bound=.25)}
BUDGETS=SETTINGS
WEIGHTS=[.025,.05,.1,.2,.25,.35,.5,.75,1.]

class SmallCorrection:
    def __init__(self,family,config):self.family=family;self.config=config
    def network(self,inputs):
        import torch
        h=self.config['hidden']
        if self.family=='gru':
            class Net(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.gru=torch.nn.GRU(inputs,h,batch_first=True);self.head=torch.nn.Linear(h,1)
                def forward(self,z):return self.head(self.gru(z)[0][:,-1]).squeeze(-1)
            net=Net();head=net.head
        else:
            net=torch.nn.Sequential(torch.nn.Linear(inputs,h),torch.nn.ReLU(),torch.nn.Linear(h,1),torch.nn.Flatten(start_dim=0));head=net[2]
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        return net
    def fit(self,x,y):
        import torch
        torch.manual_seed(48);torch.use_deterministic_algorithms(True)
        self.inputs=x.shape[-1];net=self.network(self.inputs)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=self.config['decay'])
        xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32)
        generator=torch.Generator().manual_seed(48)
        for epoch in range(self.config['epochs']):
            for batch in torch.randperm(len(y),generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                # Bounded residual is converted to target units before L1 loss.
                loss=torch.nn.functional.l1_loss(torch.tanh(net(xx[batch])),yy[batch])
                base.check(bool(torch.isfinite(loss)),'Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
        self.state={k:v.detach().clone() for k,v in net.state_dict().items()}
        return self
    def predict(self,x):
        import torch
        net=self.network(self.inputs);net.load_state_dict(self.state);net.eval()
        with torch.no_grad():return np.concatenate([torch.tanh(net(torch.tensor(x[i:i+512],dtype=torch.float32))).numpy() for i in range(0,len(x),512)])

def fit(x,y,cut,family,spec,cols):
    if spec['config']=='control':
        return control.fit(x,y,cut,family,dict(budget='small',target='residual'),cols)
    keys=y.index[y.label_end<cut];base.check(len(keys)>=1000,'Too few mature rows')
    pp=base.prepare(x,keys,cols)
    sample=keys[np.unique(np.linspace(0,len(keys)-1,min(2000,len(keys))).astype(int))]
    center=float(y.loc[keys,base.TARGET].mean());scale=max(SETTINGS[spec['config']]['bound']*center,1e-6)
    target=(y.loc[sample,base.TARGET].to_numpy()-center)/scale
    model=SmallCorrection(family,SETTINGS[spec['config']])
    model.fit(engine.design(x,sample,pp,family,'signedlog'),target)
    return dict(model=model,prep=pp,center=center,scale=scale,family=family,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())

def predict(a,x,keys):
    if isinstance(a['model'],control.ControlledNeural):return control.predict(a,x,keys)
    return np.maximum(0,a['center']+a['scale']*a['model'].predict(engine.design(x,keys,a['prep'],a['family'],'signedlog')))


def accept(errors,mean_errors):
    # MAE is the prespecified primary objective. RMSE is reported, not a gate.
    gains=np.array([np.abs(e).mean()/max(np.abs(b).mean(),1e-12) for e,b in zip(errors,mean_errors)])
    return bool(base.gate(errors,mean_errors) and (gains<=1).sum()>=3 and (gains<=1.01).all())

def choose(trials):
    order=lambda t:(t['stability_score'],t['mae'],t['cell'],t['weight'])
    best=min(trials,key=order);accepted=[t for t in trials if t['accepted']]
    return best,min(accepted,key=order) if accepted else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=['mlp','gru'],default=['mlp','gru'])
    parser.add_argument('--week2-code',type=Path,default=base.ROOT.parent/'Week 2/Code')
    parser.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    parser.add_argument('--canonical-manifest',type=Path,default=base.ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args();base.check(args.threads>0,'Invalid threads')
    import torch
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        x,y,dev,population,packs,definitions,files,audit=engine.load(args)
        # Sixteen fixed horizon-specific inputs; no feature search in this experiment.
        cols=[f'LONG__{n}__{h}h' for h in [60,120,240] for n in ['normalized_momentum','range_position','trend_efficiency','up_variance_share']]
        cols += [f'LONG__drawdown_vol_units__168h',f'LONG__jump_variance_share__336h','END120__past_endpoint_mean_720h','END120__past_endpoint_cv_720h']
        base.check(set(cols)<=set(x),'Missing fixed features');x=x[cols]
        audit.update(features=len(cols),fixed_features=cols)
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));return
        specs={name:dict(config=name) for name in SETTINGS}
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        files += [Path(__file__),Path(engine.__file__),Path(control.__file__),Path(base.__file__),base.ROOT/'PROJECT_CONTEXT.md']
        for family in args.families:
            previous_path=base.HERE/'experiments'/f'net_magnitude_120h_{family}_small_correction_v1'/'20261116T121607555821Z'
            receipt=json.loads((previous_path/'COMPLETE.json').read_text())
            base.check(base.sha(previous_path/'predictions.csv')==receipt['prediction_sha256'],'Previous reference integrity failed')
            previous=pd.read_csv(previous_path/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True)
            name=f'net_magnitude_120h_{family}_bounded_refined_v1'
            out=base.HERE/'experiments'/name/stamp;report=base.ROOT/'Report'/name/stamp
            out.mkdir(parents=True);report.mkdir(parents=True)
            base.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=7200,target=base.TARGET,family=family,input_audit=audit,fixed_features=cols,budgets=BUDGETS,cells=specs,weights=WEIGHTS,objective='MAE',gate='>=0.1% pooled MAE gain vs mean,3/4wins/ties,no>1%windowloss;rank mean(windowMAE/baselineMAE)+.25std ratios;fallback mean;RMSEdiagnostic.',selection='Four earlier3monthwindows15/12/9/6months;train/validation label maturity strict120h. Choices frozen before outer scoring.',target_note='Previous tiny8/strong16 controls retained. All residual nets startzero; bounds .1/.25/.4trainingmean via tanh;L1 loss,no target clipping.',limitations='Historical adaptive exposed evaluation. Previous fixedconfig gains motivate search,not independent confirmation. Controls retained;no untouched test,guaranteed improvement or resume.',input_hashes={str(f):base.sha(f) for f in sorted(set(files+[previous_path/'COMPLETE.json',previous_path/'predictions.csv']))}))
            population.to_csv(out/'population_audit.csv',index_label='decision_time');definitions.to_csv(out/'feature_formulas.csv',index=False)
            ledger=[];diagnostics=[];paired=[];frames=[];selections=[]
            for issue in base.ISSUES:
                windows=[]
                for offset in [15,12,9,6]:
                    cut=issue-pd.DateOffset(months=offset);end=cut+pd.DateOffset(months=3)
                    va=y.index[(y.index>=cut)&(y.index<end)&(y.label_end<end)&(y.label_end<issue)]
                    base.check(len(va)>=100,'Too few validation rows');windows.append((cut,va))
                refs=[np.full(len(va),y.loc[y.label_end<cut,base.TARGET].mean())-y.loc[va,base.TARGET].to_numpy() for cut,va in windows]
                trials=[]
                for cell,spec in specs.items():
                    predictions=[];truths=[];anchors=[]
                    for j,(cut,va) in enumerate(windows):
                        print(family,issue.date(),cell,'validation',j+1,'/4',flush=True)
                        a=fit(x,y,cut,family,spec,cols);base.check(a['max_label_end']<va.min(),'Purge failed')
                        predictions.append(predict(a,x,va));truths.append(y.loc[va,base.TARGET].to_numpy());anchors.append(float(y.loc[y.label_end<cut,base.TARGET].mean()))
                        tr=y.index[y.label_end<cut];dk=tr[np.unique(np.linspace(0,len(tr)-1,min(1000,len(tr))).astype(int))]
                        for split,keys in [('training_sample',dk),('validation',va)]:
                            pred=predict(a,x,keys);actual=y.loc[keys,base.TARGET]
                            diagnostics.append(dict(issue=str(issue),cell=cell,window=j,split=split,**base.metrics(actual,pred),actual_mean=float(actual.mean()),predicted_mean=float(pred.mean())))
                    for weight in WEIGHTS:
                        errors=[weight*p+(1-weight)*anchor-truth for p,anchor,truth in zip(predictions,anchors,truths)]
                        ratios=np.array([np.abs(e).mean()/max(np.abs(ref).mean(),1e-12) for e,ref in zip(errors,refs)])
                        trials.append(dict(issue=str(issue),cell=cell,weight=weight,stability_score=float(ratios.mean()+.25*ratios.std()),window_mae_ratios=ratios.tolist(),accepted=accept(errors,refs),**base.metrics(np.zeros(sum(map(len,errors))),np.concatenate(errors))))
                    ledger.extend(trials[-len(WEIGHTS):]);base.dump(out/'validation_trials.json',ledger)
                best,chosen=choose(trials)
                # Save the selection before calculating later-period errors.
                selections.append(dict(issue=str(issue),candidate=best,selected=chosen,models=[]));base.dump(out/'selections.json',selections)
                end=base.ISSUES[list(base.ISSUES).index(issue)+1] if issue!=base.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
                te=dev[(dev>=issue)&(dev<end)];tr=y.loc[y.label_end<issue,base.TARGET]
                mean=float(tr.mean());median=float(tr.median())
                f=pd.DataFrame(dict(actual=y.loc[te,base.TARGET],issue=issue,mean=mean,median=median),index=te)
                for cell,spec in specs.items():
                    a=fit(x,y,issue,family,spec,cols);path=out/f'model_{issue:%Y%m}_{cell}.joblib';joblib.dump(a,path,compress=3)
                    pred=predict(a,x,te);np.testing.assert_allclose(pred,predict(joblib.load(path),x,te),rtol=0,atol=1e-8)
                    f[cell]=pred;selections[-1]['models'].append(dict(artifact=path.name,sha256=base.sha(path),features=a['prep']['cols'],spec=spec,training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end']))
                    for group,mask in [('all',np.ones(len(te),dtype=bool)),('above_training_p90',f.actual.to_numpy()>float(tr.quantile(.9)))]:
                        if mask.any():paired.append(dict(issue=str(issue),cell=cell,group=group,**base.metrics(f.actual.to_numpy()[mask],pred[mask])))
                f['candidate']=best['weight']*f[best['cell']]+(1-best['weight'])*mean
                f['selected']=mean if chosen is None else chosen['weight']*f[chosen['cell']]+(1-chosen['weight'])*mean
                frames.append(f);base.dump(out/'selections.json',selections)
                pd.DataFrame(diagnostics).to_csv(out/'training_validation_diagnostics.csv',index=False)
                pd.DataFrame(paired).to_csv(out/'outer_cell_diagnostics.csv',index=False)
            result=pd.concat(frames).sort_index();base.check(result.index.equals(dev.sort_values()),'Coverage mismatch')
            base.check(np.isfinite(result.drop(columns='issue')).all().all(),'Nonfinite forecasts')
            base.check(previous.index.is_unique and previous.index.equals(result.index),'Reference alignment failed')
            for col in ['actual','mean','median']:np.testing.assert_allclose(result[col],previous[col],atol=1e-8,rtol=0)
            for col in ['tiny8','strong16']:result['previous_'+col]=previous[col]
            result.to_csv(out/'predictions.csv',index_label='decision_time')
            scores=[]
            for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
                for model in ['mean','median',*specs,'previous_tiny8','previous_strong16','selected','candidate']:scores.append(dict(period=period,model=model,**base.metrics(g.actual,g[model])))
            scores=pd.DataFrame(scores);scores.to_csv(out/'metrics.csv',index=False);scores.to_csv(report/'metrics.csv',index=False)
            text='120h controlled '+family+';primary MAE;RMSE diagnostic;historical exposed evaluation.\n'+scores.to_string(index=False)
            (report/'SUMMARY.txt').write_text(text,encoding='utf-8');base.dump(report/'selections.json',selections)
            (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
            base.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True))
            print(text,'\nOutputs:',out,'\nReport:',report,flush=True)
if __name__=='__main__':main()
