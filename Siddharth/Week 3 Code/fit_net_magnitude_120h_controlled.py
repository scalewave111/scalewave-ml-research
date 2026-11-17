"""Controlled 120h experiment: fixed features, budget and mean-residual ablations."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from threadpoolctl import threadpool_limits
import fit_net_magnitude_120h_distribution_search as engine
base=engine.base
BUDGETS={'small':dict(cap=2000,epochs=16,gru_epochs=12,rounds=40,trees=24),
         'larger':dict(cap=6000,epochs=48,gru_epochs=24,rounds=80,trees=64)}
WEIGHTS=[.05,.1,.25,.5,.75,1.]

class ControlledNeural(engine.NeuralHead):
    def __init__(self,kind,config,residual):
        super().__init__(kind,config,'l1');self.residual=residual
    def network(self,inputs):
        import torch
        net=super().network(inputs)
        if self.residual:
            final=net.head if self.kind=='gru' else [m for m in net if isinstance(m,torch.nn.Linear)][-1]
            with torch.no_grad():final.weight.zero_();final.bias.zero_()
        return net

def fit(x,y,cut,family,spec,cols):
    b=BUDGETS[spec['budget']];keys=y.index[y.label_end<cut]
    base.check(len(keys)>=1000,'Too few mature training rows')
    pp=base.prepare(x,keys,cols)
    sample=keys[np.unique(np.linspace(0,len(keys)-1,min(b['cap'],len(keys))).astype(int))]
    values=y.loc[keys,base.TARGET]
    center=float(values.mean() if spec['target']=='residual' else values.median())
    scale=max(float(values.std()),1.)
    target=(y.loc[sample,base.TARGET].to_numpy()-center)/scale
    if family=='hgb':
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=b['rounds'],max_depth=3,max_leaf_nodes=7,min_samples_leaf=128,l2_regularization=10,learning_rate=.05,max_bins=63,early_stopping=False,random_state=48)
    elif family=='extra':
        model=ExtraTreesRegressor(n_estimators=b['trees'],criterion='absolute_error',max_depth=4,min_samples_leaf=32,max_features=.75,bootstrap=True,max_samples=min(1024,len(sample)),n_jobs=1,random_state=48)
    else:
        engine.FAST=spec['budget']=='small'
        engine.CONFIGS={'small_budget':([32],16,.1),'larger_budget':([32],48,.1)}
        model=ControlledNeural(family,spec['budget']+'_budget',spec['target']=='residual')
    a=engine.design(x,sample,pp,family,'signedlog');model.fit(a,target)
    return dict(model=model,prep=pp,center=center,scale=scale,family=family,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())

def predict(a,x,keys):
    # CONFIGS controls architecture on replay; both cells have the same hidden width.
    engine.CONFIGS={'small_budget':([32],16,.1),'larger_budget':([32],48,.1)}
    return np.maximum(0,a['center']+a['scale']*a['model'].predict(engine.design(x,keys,a['prep'],a['family'],'signedlog')))

def accept(errors,mean_errors):
    # MAE is the prespecified primary objective. RMSE is reported, not a gate.
    return base.gate(errors,mean_errors)

def choose(trials):
    order=lambda t:(t['mae'],t['cell'],t['weight'])
    best=min(trials,key=order);accepted=[t for t in trials if t['accepted']]
    return best,min(accepted,key=order) if accepted else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=['hgb','extra','mlp','gru'],default=['hgb','extra','mlp','gru'])
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
        specs={b+'_'+t:dict(budget=b,target=t) for b in BUDGETS for t in ['direct','residual']}
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        files += [Path(__file__),Path(engine.__file__),Path(base.__file__),base.ROOT/'PROJECT_CONTEXT.md']
        for family in args.families:
            name=f'net_magnitude_120h_{family}_controlled_v1'
            out=base.HERE/'experiments'/name/stamp;report=base.ROOT/'Report'/name/stamp
            out.mkdir(parents=True);report.mkdir(parents=True)
            base.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=7200,target=base.TARGET,family=family,input_audit=audit,fixed_features=cols,budgets=BUDGETS,cells=specs,weights=WEIGHTS,objective='MAE',gate='>=0.1% pooled MAE gain vs training mean;2/4 wins/ties,no>2%windowloss. Otherwise mean. RMSE diagnostic only.',selection='Four earlier3monthwindows15/12/9/6months;train/validation label maturity strict120h. Choices frozen before outer scoring.',target_note='Direct is median-centered/std-scaled; residual is mean-centered/std-scaled. Neural residual readout starts zero. Constant target shifts are translation-equivalent for tree losses, not a new source of signal.',limitations='Historical exposed adaptive evaluation; larger budget jointly changes row cap and iteration budget. Not an exact replay of previous fast model. No untouched test,guaranteed improvement or resume.',input_hashes={str(f):base.sha(f) for f in sorted(set(files))}))
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
                        trials.append(dict(issue=str(issue),cell=cell,weight=weight,accepted=accept(errors,refs),**base.metrics(np.zeros(sum(map(len,errors))),np.concatenate(errors))))
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
            result.to_csv(out/'predictions.csv',index_label='decision_time')
            scores=[]
            for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
                for model in ['mean','median',*specs,'selected','candidate']:scores.append(dict(period=period,model=model,**base.metrics(g.actual,g[model])))
            scores=pd.DataFrame(scores);scores.to_csv(out/'metrics.csv',index=False);scores.to_csv(report/'metrics.csv',index=False)
            text='120h controlled '+family+';primary MAE;RMSE diagnostic;historical exposed evaluation.\n'+scores.to_string(index=False)
            (report/'SUMMARY.txt').write_text(text,encoding='utf-8');base.dump(report/'selections.json',selections)
            (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
            base.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True))
            print(text,'\nOutputs:',out,'\nReport:',report,flush=True)
if __name__=='__main__':main()
