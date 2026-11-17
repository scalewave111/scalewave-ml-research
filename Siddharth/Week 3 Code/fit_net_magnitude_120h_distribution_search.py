"""User-run 120h net magnitude: four independent paired-objective models with 120h endpoint-specific features and baseline blending."""
from pathlib import Path
import argparse, json, sys, importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import fit_net_magnitude_120h as base

CONFIGS={'small':([32,16],40,.01),'medium':([64,32],60,.01),'regularized':([32],60,.1)}
WEIGHTS=[.25,.5,.75,1.]
FAST=False


import net_120h_long_features as price_features
import net_120h_market_features as state_features
from itertools import product
from net_120h_endpoint_features import endpoint_features, endpoint_columns
import net_120h_refinement_features as refinement

HEAD_WEIGHTS=[0.,.25,.5,.75,1.]
WEIGHTS=[.05,.1,.2,.25,.35,.5,.65,.75,.9,1.]

def configure_fast(args):
    global FAST,CONFIGS
    FAST=args.fast
    if FAST:
        CONFIGS={'regularized':([32],16,.1)}
        args.row_cap=min(args.row_cap,2000)

def load(args):
    _,y,dev,population,_,_,files,audit=base.load(args)
    # The loader imports this established read-only source dependency.
    import fit_excursions_48h as prior
    timeline,_,_,_,_=prior.e2.old.load()
    market,market_files=state_features.load_sources(prior.e2.old.data.DATA,timeline.index)
    files += market_files
    returns_path=prior.e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=prior.e2.price.canonical.load_returns(returns_path)
    price,pdict=price_features.build(r,market.index,120)
    state,sdict=state_features.build(market,price,market.index)
    x=price.join(state)
    # Prespecified compact groups; no feature selected using outer scores.
    price,extra_dictionary=endpoint_features(r,market.index,price)
    x=price.join(state)
    volatility=[f'LONG__vol_per_sqrt_hour__{h}h' for h in [24,60,120,240,168,336,720,1440]]
    proxies=endpoint_columns()
    trend=[f'LONG__{name}__{h}h' for h in [60,120,240] for name in ['normalized_momentum','range_position','trend_efficiency','up_variance_share']]
    trend += [f'LONG__{name}__{h}h' for name,h in [('drawdown_vol_units',168),('jump_variance_share',336)]]
    persistence=[f'END120__{name}_{h}h' for h in [168,720] for name in ['past_endpoint_mean','past_endpoint_rms','past_endpoint_cv']]
    proxies += persistence
    slow=[f'STATE__{f}_{stat}__{h}h' for f,stat,h in product(['funding_level','basis_level','oi_position','oi_unwind','flow_pressure','liq_history'],['mean','deviation'],[24,120])]
    packs={'volatility_endpoint':volatility+proxies,'trend_range':trend+proxies,'slow_market':volatility+proxies+slow}
    fresh,fresh_dictionary=refinement.build(r,market.index)
    x=x.join(fresh)
    packs['endpoint_distribution_search']=packs['trend_range']+list(fresh)
    packs={k:packs[k] for k in ['trend_range','endpoint_distribution_search']}
    used=sorted(set(c for cols in packs.values() for c in cols))
    base.check(set(used)<=set(x),'Missing 120h-specific feature columns')
    x=x[used]
    # Causal checks on both price and market features; no outcome eligibility mask.
    cut=market.index[len(market)//2];keys=market.index[market.index<=cut][-10:]
    changed=r.copy();changed.loc[changed.index>=cut]=1e9
    pp,_=price_features.build(changed,market.index,120)
    mm=market.copy();mm.loc[mm.index>=cut]=1e9
    ss,_=state_features.build(mm,pp,market.index)
    pp,_=endpoint_features(changed,market.index,pp)
    fresh_check,_=refinement.build(changed,market.index)
    pd.testing.assert_frame_equal(x.loc[keys],pp.join(ss).join(fresh_check)[used].loc[keys])
    prefix=market.loc[market.index<=cut]
    pp,_=price_features.build(r.loc[r.index<=cut],prefix.index,120)
    ss,_=state_features.build(prefix,pp,prefix.index)
    pp,_=endpoint_features(r.loc[r.index<=cut],prefix.index,pp)
    fresh_check,_=refinement.build(r.loc[r.index<=cut],prefix.index)
    pd.testing.assert_frame_equal(x.loc[keys],pp.join(ss).join(fresh_check)[used].loc[keys])
    dictionary=pd.concat([pdict,sdict,extra_dictionary,fresh_dictionary],ignore_index=True).fillna('')
    dictionary=dictionary[dictionary.feature.isin(used)]
    audit.update(features=len(x.columns),packs={k:len(v) for k,v in packs.items()},feature_dictionary=dictionary.to_dict('records'),new_feature_causality='price and market prefix/future perturbation passed',gru_snapshots_hours=[36,24,12,0],market_source='Projected required columns from original masked X_train/X_development; aligned to inherited full timeline',market_source_columns=state_features.SOURCES)
    files += [Path(price_features.__file__),Path(state_features.__file__),Path(__file__).with_name('net_120h_endpoint_features.py'),Path(refinement.__file__)]
    return x,y,dev,population,packs,dictionary,files,audit

def basis(a,kind):
    # x already contains training-only standardized values and missing flags.
    if kind=='raw':return a
    n=a.shape[-1]//2
    values=a[...,:n];flags=a[...,n:]
    z=np.sign(values)*np.log1p(np.abs(values))
    return np.concatenate([z,z*z,flags] if kind=='quadratic' else [z,flags],axis=-1)

def design(x,keys,pp,kind,transform):
    if kind!='gru':return basis(base.transform(x.reindex(keys),pp),transform)
    return np.stack([basis(base.transform(x.reindex(keys-pd.Timedelta(hours=h)),pp),transform) for h in [36,24,12,0]],axis=1)

class NeuralHead:
    def __init__(self,kind,config,objective):
        self.kind=kind;self.config=config;self.objective=objective
    def network(self,inputs):
        import torch
        if self.kind=='gru':
            class Net(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.gru=torch.nn.GRU(inputs,16,batch_first=True);self.head=torch.nn.Linear(16,1)
                def forward(self,z):return self.head(self.gru(z)[0][:,-1]).squeeze(-1)
            return Net()
        dims=[inputs]+CONFIGS[self.config][0]+[1];layers=[]
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2:layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers,torch.nn.Flatten(start_dim=0))
    def fit(self,x,y):
        import torch
        torch.manual_seed(48);torch.use_deterministic_algorithms(True)
        self.inputs=x.shape[-1];net=self.network(self.inputs)
        epochs=(12 if FAST else 24) if self.kind=='gru' else CONFIGS[self.config][1]
        decay=.03 if self.kind=='gru' else CONFIGS[self.config][2]
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=decay)
        xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32)
        generator=torch.Generator().manual_seed(48)
        lossfn=torch.nn.functional.l1_loss if self.objective=='l1' else torch.nn.functional.mse_loss
        for epoch in range(epochs):
            for batch in torch.randperm(len(y),generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True);loss=lossfn(net(xx[batch]),yy[batch])
                base.check(bool(torch.isfinite(loss)),'Nonfinite neural loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        self.state={k:v.detach().clone() for k,v in net.state_dict().items()}
        return self
    def predict(self,x):
        import torch
        net=self.network(self.inputs);net.load_state_dict(self.state);net.eval()
        with torch.no_grad():return np.concatenate([net(torch.tensor(x[i:i+512],dtype=torch.float32)).numpy() for i in range(0,len(x),512)])

def fit(x,y,cut,spec,packs,cap):
    keys=y.index[y.label_end<cut];base.check(len(keys)>=1000,'Insufficient mature training rows')
    cols=[c for c in packs[spec['pack']] if np.isfinite(x.loc[keys,c]).sum()>=1000]
    pp=base.prepare(x,keys,cols)
    sample=keys[np.unique(np.linspace(0,len(keys)-1,min(cap,len(keys))).astype(int))]
    a=design(x,sample,pp,spec['model'],spec['transform'])
    center=float(y.loc[keys,base.TARGET].median());scale=max(float(y.loc[keys,base.TARGET].std()),1.)
    target=(y.loc[sample,base.TARGET].to_numpy()-center)/scale
    heads=[]
    for objective in ['l1','l2']:
        print('  fitting',objective,'head;',len(sample),'estimator rows',flush=True)
        if spec['model']=='hgb':
            model=HistGradientBoostingRegressor(loss='absolute_error' if objective=='l1' else 'squared_error',max_iter=40 if FAST else 60,max_leaf_nodes=7,max_depth=3,min_samples_leaf=128,l2_regularization=10,learning_rate=.05,max_bins=63,early_stopping=False,random_state=48)
        elif spec['model']=='extra':
            model=ExtraTreesRegressor(n_estimators=24 if FAST else 64,criterion='absolute_error' if objective=='l1' else 'squared_error',max_depth=4 if FAST else 6,min_samples_leaf=32,max_features=.75,bootstrap=True,max_samples=min(1024 if FAST else 4096,len(sample)),random_state=48,n_jobs=1)
        else:model=NeuralHead(spec['model'],spec['config'],objective)
        model.fit(a,target);heads.append(model)
    return dict(prep=pp,heads=heads,center=center,scale=scale,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())

def predict_pair(bundle,x,keys):
    spec=bundle['spec'];pp=bundle['prep']
    a=design(x,keys,pp,spec['model'],spec['transform'])
    # Require sufficient observed inputs in every GRU snapshot, as for UP120h.
    snapshots=[keys-pd.Timedelta(hours=h) for h in [36,24,12,0]] if spec['model']=='gru' else [keys]
    good=np.logical_and.reduce([np.isfinite(x.reindex(k)[pp['cols']]).mean(axis=1).to_numpy()>=.8 for k in snapshots])
    return np.stack([np.where(good,np.maximum(0,model.predict(a)*bundle['scale']+bundle['center']),bundle['center']) for model in bundle['heads']])

def assess(errors,refs):
    def metric(e,rmse=False):
        a=np.concatenate(e)
        return float(np.sqrt(np.mean(a*a)) if rmse else np.abs(a).mean())
    ratios=[];accepted=True
    for reference,rmse in [('median',False),('mean',False),('mean',True)]:
        reference_errors=refs[reference]
        value=metric(errors,rmse);baseline=metric(reference_errors,rmse)
        ratios.append(value/max(baseline,1e-12))
        window=np.array([metric([e],rmse) for e in errors]);control=np.array([metric([e],rmse) for e in reference_errors])
        accepted=accepted and value<=.999*baseline and (window<=control).sum()>=3 and bool((window<=1.02*control).all())
    return max(ratios),bool(accepted)

def blend(pred,anchor,weight):return weight*np.asarray(pred)+(1-weight)*anchor

def choose(trials):
    best=min(trials,key=lambda t:(t['mae'],t['score'],t['id'],t['anchor'],t['weight'],t['head_weight']))
    accepted=[t for t in trials if t['accepted']]
    return best,min(accepted,key=lambda t:(t['mae'],t['score'],t['id'],t['anchor'],t['weight'],t['head_weight'])) if accepted else None

def run_family(x,y,dev,packs,family,args,files,audit,population):
    stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    name=f'net_magnitude_120h_{family}_distribution_search'+('_fast_v1' if FAST else '_v1')
    out=base.HERE/'experiments'/name/stamp;report=base.ROOT/'Report'/name/stamp
    out.mkdir(parents=True);report.mkdir(parents=True)
    pd.DataFrame(audit['feature_dictionary']).to_csv(out/'feature_dictionary.csv',index=False)
    specs=[dict(model=family,pack=pack,config=config,transform=transform) for pack in packs for transform in args.transforms for config in (CONFIGS if family=='mlp' else ['default'])]
    base.dump(out/'CONTRACT.json',dict(horizon_minutes=7200,target=base.TARGET,formula=base.FORMULA,family=family,arguments=vars(args),input_audit=audit,packs=packs,specs=specs,weights=WEIGHTS,head_weights=HEAD_WEIGHTS,transforms=args.transforms,anchors=['median','mean'],fallback='mean',seed=48,configs=CONFIGS,fast_settings=(dict(packs=list(packs),row_cap=args.row_cap,hgb_iterations=40,extra_trees=24,extra_depth=4,extra_bootstrap_cap=1024,mlp_epochs=16,gru_epochs=12) if FAST else None),
        preprocessing='Training-only finite coverage >=80%, >=1000 finite observations, nonconstant; median/IQR, clip +/-8, missing flags; optional signedlog or signedlog plus squares. Target median-centered/std-scaled, no target log. Inference requires >=80% finite retained inputs at every snapshot, otherwise training median.',
        head_models=dict(hgb='absolute/squared error; depth3,7leaves,leaf128,L2=10,rate.05,bins63,40fast/60full iterations',extra='absolute/squared error;24fast/64full trees,depth4fast/6full,minleaf32,maxfeatures.75,bootstrap1024fast/4096full',mlp='L1/L2 ReLU AdamW lr.001; configs recorded;batch256,clip1',gru='L1/L2 GRU16 AdamW lr.001 decay.03,12fast/24full epochs,batch256,clip1; snapshots36/24/12/0h'),
        selection='Four earlier three-month windows at offsets15/12/9/6months; validation labels mature before window end. Rank pooled MAE; worst relative MAE/RMSE score retained as tie-break/diagnostic. Require >=0.1% MAE gain versus EACH mean/median and RMSE gain versus mean; >=3/4 window wins/ties and no >2% window loss for each metric/comparator. Otherwise training mean. Diagnostic candidate separate.',
        adaptation='MAE-ranked refinement;12new historical endpoint-quantile/persistence/regime inputs;paired controltrend36 versus augmented48input pack; original four families independent, no family mixing;120h-specific 5min volatility, drift/absolute endpoint proxies, historical endpoint persistence, trend/range and slow market packs; four windows, 36/24/12/0h GRU snapshots, paired L1/L2 heads and baseline blending; new net target and freshly fitted models. No saved UP model anchors or prior net runs required.',
        versions={v:importlib.metadata.version(v) for v in ['numpy','pandas','scikit-learn','torch','joblib']},input_hashes={str(f):base.sha(f) for f in sorted(set(files))},limitations='Historical adaptive exposed evaluation. Not exact UP reproduction; no guaranteed win or fresh test. No resume.'))
    population.to_csv(out/'population_audit.csv',index_label='decision_time')
    frames=[];selections=[];ledger=[]
    for issue in base.ISSUES:
        windows=[]
        for offset in [15,12,9,6]:
            cut=issue-pd.DateOffset(months=offset)
            window_end=cut+pd.DateOffset(months=3)
            va=y.index[(y.index>=cut)&(y.index<window_end)&(y.label_end<window_end)&(y.label_end<issue)]
            base.check(len(va)>=100,'Too few validation rows')
            tr=y.loc[y.label_end<cut,base.TARGET]
            windows.append((cut,va,float(tr.median()),float(tr.mean())))
        truths=[y.loc[va,base.TARGET].to_numpy() for cut,va,md,mn in windows]
        refs={anchor:[np.full(len(va),md if anchor=='median' else mn)-truth for (cut,va,md,mn),truth in zip(windows,truths)] for anchor in ['median','mean']}
        trials=[]
        for i,spec in enumerate(specs):
            predictions=[]
            for window_id,(cut,va,md,mn) in enumerate(windows,1):
                print(family,issue.date(),f'candidate {i+1}/{len(specs)}, window {window_id}/4: fitting',flush=True)
                a=fit(x,y,cut,spec,packs,args.row_cap)
                base.check(a['max_label_end']<va.min(),'Training purge failed')
                predictions.append(predict_pair(a,x,va))
            for head_weight in HEAD_WEIGHTS:
                head_predictions=[blend(pair[1],pair[0],head_weight) for pair in predictions]
                for anchor in ['median','mean']:
                    for weight in WEIGHTS:
                        errors=[blend(pred,md if anchor=='median' else mn,weight)-truth for pred,(cut,va,md,mn),truth in zip(head_predictions,windows,truths)]
                        score,accepted=assess(errors,refs)
                        record=dict(issue=str(issue),id=i,spec=spec,anchor=anchor,weight=weight,head_weight=head_weight,score=score,mae=float(np.abs(np.concatenate(errors)).mean()),rmse=float(np.sqrt(np.mean(np.concatenate(errors)**2))),accepted=accepted,window_mae=[float(np.abs(e).mean()) for e in errors])
                        trials.append(record);ledger.append(record)
            print(family,issue.date(),i+1,'/',len(specs),spec,flush=True)
            base.dump(out/'validation_trials.json',ledger)
        control=min((t for t in trials if t['spec']['pack']=='trend_range'),key=lambda t:(t['mae'],t['score']))
        augmented=min((t for t in trials if t['spec']['pack']=='endpoint_distribution_search'),key=lambda t:(t['mae'],t['score']))
        ablation=dict(issue=str(issue),family=family,control_mae=control['mae'],augmented_mae=augmented['mae'],validation_mae_reduction=control['mae']-augmented['mae'],control_rmse=control['rmse'],augmented_rmse=augmented['rmse'],validation_rmse_reduction=control['rmse']-augmented['rmse'],augmented_accepted=augmented['accepted'])
        path=out/'feature_group_ablation.csv'
        pd.DataFrame([ablation]).to_csv(path,mode='a',header=not path.exists(),index=False)
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
                pred=predict_pair(a,x,te);np.testing.assert_allclose(pred,predict_pair(joblib.load(path),x,te),atol=1e-8,rtol=0)
                fitted[i]=pred;selection['models'].append(dict(artifact=path.name,sha256=base.sha(path),features=a['prep']['cols'],spec=specs[i],training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end']))
        f['candidate']=blend(blend(fitted[best['id']][1],fitted[best['id']][0],best['head_weight']),anchors[best['anchor']],best['weight'])
        f['selected']=anchors['mean'] if chosen is None else blend(blend(fitted[chosen['id']][1],fitted[chosen['id']][0],chosen['head_weight']),anchors[chosen['anchor']],chosen['weight'])
        frames.append(f);base.dump(out/'selections.json',selections)
    result=pd.concat(frames).sort_index();base.check(result.index.equals(dev.sort_values()),'Coverage mismatch')
    base.check(np.isfinite(result[['actual','mean','median','selected','candidate']]).all().all(),'Nonfinite final forecasts')
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    rows=[]
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for model in ['median','mean','selected','candidate']:rows.append(dict(period=period,model=model,**base.metrics(g.actual,g[model])))
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False);table.to_csv(report/'metrics.csv',index=False)
    text='120h net magnitude: '+family+' independent blended search\n'+table.to_string(index=False)+'\nCandidate is diagnostic; selected may fall back to mean. Historical exposed evaluation.\n'
    (report/'SUMMARY.txt').write_text(text);base.dump(report/'selections.json',selections)
    for file in [Path(__file__),Path(base.__file__),Path(price_features.__file__),Path(state_features.__file__),Path(__file__).with_name('net_120h_endpoint_features.py'),Path(refinement.__file__)]: (out/file.name).write_bytes(file.read_bytes())
    base.dump(out/'COMPLETE.json',dict(status='COMPLETE',target=base.TARGET,rows=len(result),prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True))
    print(text,'\nOutputs:',out,'\nReport:',report,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=['hgb','extra','mlp','gru'],default=['hgb','extra','mlp','gru'])
    parser.add_argument('--week2-code',type=Path,default=base.ROOT.parent/'Week 2/Code')
    parser.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    parser.add_argument('--canonical-manifest',type=Path,default=base.ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json',help='Alternate canonical checksum manifest; defaults to verified local receipt')
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--row-cap',type=int,default=6000)
    parser.add_argument('--fast',action='store_true',default=True,help='2 paired endpoint-distribution packs, one regularized MLP, 2000-row cap and reduced estimator budgets')
    parser.add_argument('--full-search',dest='fast',action='store_false',help='Restore original full search and estimator budgets')
    parser.add_argument('--transforms',nargs='+',choices=['raw','signedlog','quadratic'],default=['signedlog'])
    parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args();base.check(args.threads>0 and args.row_cap>=256,'Invalid thread/cap')
    base.check(len(set(args.families))==len(args.families) and len(set(args.transforms))==len(args.transforms),'Duplicate families/transforms')
    configure_fast(args)
    import torch
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        x,y,dev,population,packs,definitions,files,audit=load(args)
        for issue in base.ISSUES:
            for offset in [15,12,9,6]:
                cut=issue-pd.DateOffset(months=offset);end=cut+pd.DateOffset(months=3)
                base.check((y.label_end<cut).sum()>=1000,'Insufficient mature training history')
                base.check(((y.index>=cut)&(y.label_end<end)&(y.label_end<issue)).sum()>=100,'Insufficient mature validation history')
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));return
        files += [Path(__file__),Path(base.__file__),base.ROOT/'PROJECT_CONTEXT.md']
        framework=base.ROOT.parents[1]/'ml_framework'
        files += [framework/v for v in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        # Population is written separately, not serialized into CLI arguments.
        for family in args.families:
            run_family(x,y,dev,packs,family,args,files,audit,population)

if __name__=='__main__':main()
