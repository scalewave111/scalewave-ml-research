"""CPU experiments for conditional BTC 48h ordinary-bps magnitudes.

Save in Week 2/Code. Run --model garch or --model gru separately.
NEW TRAINING occurs only when you run this file. Existing inputs are read-only.
Dependencies: numpy pandas scipy joblib torch threadpoolctl; GARCH: duckdb.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
      'PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h',
      'F4Z__liquidation__venue_breadth_percentile__1h__value',
      'F4Z__funding__funding_persistence_percentile__5m__value',
      'F4Z__basis__basis_level_rz__1h__value']
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
STEPS=48
SEED=48
MAX_EPOCHS=24
PATIENCE=5


def check(ok,msg):
    if not ok:
        raise ValueError(msg)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid scoring arrays')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def prepare(x,keys):
    a=x.loc[keys,COLS]; med=a.median().fillna(0); filled=a.fillna(med)
    scale=filled.quantile(.75)-filled.quantile(.25)
    scale=scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1).clip(lower=1e-6)
    return dict(columns=COLS,median=med,scale=scale)


def gru_fit_predict(x,target,train,valid,test,epochs=None):
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    torch.manual_seed(SEED)
    prep=prepare(x,train)
    z=((x[COLS].fillna(prep['median'])-prep['median'])/prep['scale']).clip(-8,8)
    values=np.column_stack([z.to_numpy(np.float32),x[COLS].isna().to_numpy(np.float32)])
    # Missing timestamps get median-imputed values and all missing flags.
    values=np.vstack([values,np.r_[np.zeros(8),np.ones(8)].astype(np.float32)])
    check(np.isfinite(values).all(),'Nonfinite GRU design')
    offsets=np.arange(STEPS-1,-1,-1)*pd.Timedelta(minutes=30).value

    class Sequences(Dataset):
        def __init__(self,keys):
            times=keys.asi8[:,None]-offsets[None,:]
            self.ids=x.index.get_indexer(pd.to_datetime(times.ravel(),utc=True)).reshape(len(keys),STEPS)
            self.ids[self.ids<0]=len(values)-1
            self.y=(target.loc[keys].to_numpy(np.float32)/100.)
        def __len__(self): return len(self.y)
        def __getitem__(self,i): return torch.from_numpy(values[self.ids[i]]),torch.tensor(self.y[i])

    class GRU(nn.Module):
        def __init__(self,median):
            super().__init__()
            self.gru=nn.GRU(16,16,batch_first=True)
            self.head=nn.Linear(16,1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(median/100.,1e-6)))))
        def forward(self,a):
            _,h=self.gru(a)
            return nn.functional.softplus(self.head(h[-1])).squeeze(1)

    loaders={name:DataLoader(Sequences(keys),batch_size=256,shuffle=name=='train',num_workers=0)
             for name,keys in [('train',train),('valid',valid),('test',test)]}
    model=GRU(float(target.loc[train].median()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    def predict(loader):
        model.eval(); result=[]
        with torch.no_grad():
            for a,_ in loader: result.append(model(a).numpy().astype(float)*100.)
        return np.concatenate(result)
    history=[]
    initial=predict(loaders['valid'])
    best=metrics(target.loc[valid],initial)['mae_bps']
    best_epoch=0; best_state=copy.deepcopy(model.state_dict()); stale=0
    limit=MAX_EPOCHS if epochs is None else epochs
    for ep in range(1,limit+1):
        model.train()
        for a,b in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss=(model(a)-b).abs().mean()
            check(bool(torch.isfinite(loss)),'Nonfinite GRU loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if epochs is None:
            met=metrics(target.loc[valid],predict(loaders['valid']))
            history.append(dict(epoch=ep,**met))
            if met['mae_bps']<best-1e-4:
                best=met['mae_bps']; best_epoch=ep; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            print(f'  GRU epoch {ep}: validation MAE {met["mae_bps"]:.3f}',flush=True)
            if stale>=PATIENCE: break
        else:
            best_epoch=ep; best_state=copy.deepcopy(model.state_dict())
            if ep==1 or ep%4==0: print(f'  GRU outer epoch {ep}/{limit}',flush=True)
    model.load_state_dict(best_state)
    vp=predict(loaders['valid']); tp=predict(loaders['test'])
    artifact=dict(state={k:v.cpu().numpy() for k,v in best_state.items()},prep=prep,epochs=best_epoch,
                  seed=SEED,steps=STEPS,hidden=16,history=history,train_first=str(train.min()),train_last=str(train.max()))
    return tp,vp,artifact


def garch_filter_step(h, value, omega, alpha, beta):
    # Missing innovation: integrate its square using E[r_t^2 | past] = h.
    # This is an expected-variance approximation, not an observed zero return.
    if np.isnan(value):
        return omega+(alpha+beta)*h
    check(np.isfinite(value),'Infinite GARCH return encountered')
    return omega+alpha*value**2+beta*h


def garch_runs(values):
    """Keep missing intervals in place; batch consecutive observed intervals."""
    values=np.asarray(values,float)
    check(not np.isinf(values).any(),'Infinite return in GARCH parameter sample')
    missing=np.isnan(values)
    bounds=np.r_[0,np.flatnonzero(missing[1:]!=missing[:-1])+1,len(values)]
    return [(bool(missing[a]), values[a:b]) for a,b in zip(bounds[:-1],bounds[1:])]


def garch_likelihood(params,runs,initial_variance=1.):
    """Approximate Gaussian QML with expected-variance propagation over gaps.
    Missing observations contribute no likelihood. This is not exact integration
    over latent missing shocks. All variances remain conditional on past data.
    Returns average objective and variance for the next interval.
    """
    from scipy.signal import lfilter
    omega,alpha,beta=params
    h=float(initial_variance); total=0.; count=0
    for missing,block in runs:
        if missing:
            for _ in range(len(block)):
                h=omega+(alpha+beta)*h
        else:
            squared=block**2
            forcing=omega+alpha*np.r_[0.,squared[:-1]]
            variances,_=lfilter([1.],[1.,-beta],forcing,zi=[h-omega])
            if not np.isfinite(variances).all() or (variances<=0).any():
                return 1e20, float('nan')
            total+=float(np.sum(np.log(variances)+squared/variances))
            count+=len(block)
            h=omega+alpha*squared[-1]+beta*variances[-1]
    return .5*total/max(count,1),float(h)


def fit_garch_with_gaps(fit):
    from scipy.optimize import minimize
    a=fit.to_numpy(float)
    check(not np.isinf(a).any() and np.isfinite(a).sum()>=5000,
          'GARCH needs at least 5000 observed returns in its fixed parameter window')
    scale2=float(np.nanmean(a*a))
    check(np.isfinite(scale2) and scale2>0,'Invalid GARCH return scale')
    runs=garch_runs(a/np.sqrt(scale2))
    # Fixed numerical starting points; choose by TRAINING likelihood only.
    starts=[(.05,.05,.9),(.1,.1,.8),(.02,.15,.83)]
    candidates=[]; attempts=[]
    for start in starts:
        result=minimize(lambda par:garch_likelihood(par,runs)[0],start,
            method='SLSQP',bounds=[(1e-8,10.),(0.,.999999),(0.,.999999)],
            constraints=[{'type':'ineq','fun':lambda par:.999999-par[1]-par[2]}],
            options={'maxiter':300,'ftol':1e-9})
        good=bool(result.success and np.isfinite(result.fun) and result.fun<1e19
                  and result.x[0]>0 and min(result.x[1:])>=0 and sum(result.x[1:])<=1.)
        attempts.append(dict(success=good,message=str(result.message),iterations=int(result.nit)))
        if good: candidates.append(result)
    check(bool(candidates),f'GARCH optimization failed: {attempts}')
    best=min(candidates,key=lambda res:res.fun)
    omega,alpha,beta=map(float,best.x)
    _,next_h=garch_likelihood(best.x,runs)
    gap_lengths=[len(block) for missing,block in runs if missing]
    info=dict(estimator='normalized Gaussian approximate QML; expected variance across missing observations',
        missing_fit_returns=int(np.isnan(a).sum()),longest_missing_fit_gap_minutes=5*max(gap_lengths,default=0),
        observed_fit_returns=int(np.isfinite(a).sum()),objective=float(best.fun),attempts=attempts,
        initialization='mean squared observed return in parameter-fit sample; sample strictly precedes mapping origins')
    return omega*scale2,alpha,beta,next_h*scale2,info


def garch_predict(r,y,target,side,cutoff,keys):
    """GARCH params before mapping window; MAE scale from later matured labels.
    Zero-mean Gaussian GARCH(1,1), no direction probability model.
    Forecast endpoints strictly after origin; return observations strictly before.
    """
    mapping_start=cutoff-pd.Timedelta(days=90)
    fit_start=mapping_start-pd.Timedelta(days=180)
    fit_grid=pd.date_range(fit_start,mapping_start-pd.Timedelta(minutes=5),freq='5min')
    fit=r.reindex(fit_grid)/100. # log-bps -> log-percent
    omega,alpha,beta,h,fit_info=fit_garch_with_gaps(fit)
    print(f'  GARCH {cutoff.date()}: fitted on {fit_info["observed_fit_returns"]} observed returns; {fit_info["missing_fit_returns"]} missing intervals preserved',flush=True)
    phi=alpha+beta
    check(omega>0 and alpha>=0 and beta>=0 and phi<=1.000001,'Invalid GARCH coefficients')
    # h is the conditional variance after the parameter-fit sample, including gaps.
    mapping=y.index[(y.index>=mapping_start)&(y.label_end<cutoff)]
    check(len(mapping)>0,'Empty GARCH mapping window')
    origins=mapping.union(keys).sort_values()
    grid=pd.date_range(fit.index[-1]+pd.Timedelta(minutes=5),origins.max()-pd.Timedelta(minutes=5),freq='5min')
    after=r.reindex(grid)/100.
    check(not np.isinf(after.to_numpy()).any(),'Infinite 5m returns during filtering')
    missing=after.isna()
    missing_count=int(missing.sum())
    longest_gap=0; streak=0
    for absent in missing.to_numpy():
        streak=streak+1 if absent else 0
        longest_gap=max(longest_gap,streak)
    if missing_count:
        print(f'  GARCH {cutoff.date()}: expected-variance propagation across {missing_count} missing 5m returns; longest gap {longest_gap*5} minutes',flush=True)
    endpoints=after.index; vals=after.to_numpy(); j=0
    last=fit.index[-1]; sigmas=[]
    # Forecast weights: sum_{k=0}^{575} phi^k and omega contribution.
    powers=phi**np.arange(576); weight=float(powers.sum())
    intercept=float(omega*np.cumsum(np.r_[0.,powers[:-1]]).sum())
    for origin in origins:
        while j<len(vals) and endpoints[j]<origin:
            h=garch_filter_step(h,vals[j],omega,alpha,beta)
            last=endpoints[j]; j+=1
        check(origin-last==pd.Timedelta(minutes=5),'GARCH origin must be on complete 5m grid')
        # h forecasts return ending at origin (excluded from known information).
        # Advance expectation once to return ending at origin+5m.
        first=omega+phi*h
        total=weight*first+intercept
        check(np.isfinite(total) and total>0,'Invalid forecast variance')
        sigmas.append(np.sqrt(total)*100.) # 48h volatility in log-bps
    sigma=pd.Series(sigmas,index=origins)
    output={}; scales={}
    for direction in ['UP','DOWN']:
        mk=mapping[side.loc[mapping].eq(direction)]
        check(len(mk)>=100,'Too few direction-specific mapping labels')
        s=sigma.loc[mk].to_numpy(); ratios=target.loc[mk].to_numpy()/s
        order=np.argsort(ratios); cumulative=np.cumsum(s[order])
        # Weighted median minimizes sum |ordinary_magnitude - scale*volatility|.
        scale=float(ratios[order[np.searchsorted(cumulative,.5*cumulative[-1])]])
        scales[direction]=dict(scale=scale,rows=len(mk),max_label_end=str(y.loc[mk,'label_end'].max()))
        output[direction]=sigma.loc[keys].to_numpy()*scale
    return output,dict(omega=omega,alpha=alpha,beta=beta,parameter_first=str(fit.index.min()),
        parameter_last=str(fit.index.max()),mapping_start=str(mapping_start),cutoff=str(cutoff),scales=scales,
        parameter_fit=fit_info,missing_filter_returns=missing_count,longest_missing_filter_gap_minutes=longest_gap*5,
        missing_filter_policy='Expected-variance propagation in fitting and filtering; approximate QML, no future fill, zero-return substitution or time compression.',
        variance_unit='log-percent squared',mapping='MAE-optimal nonnegative scalar to ordinary-bps magnitude')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=['garch','gru'],required=True)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args(); workspace=args.workspace.resolve(); week1=workspace.parent/'Week 1'
    out=args.out or workspace/f'Code/experiments/{args.model}_conditional_mae_v1'
    check(not out.exists(),f'Preserving existing output: {out}; use --out with a new directory')
    data=week1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    oldpath=week1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    x,y,dev,ref=joblib.load(data)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid prepared index')
    check(len(dev)==18481 and dev.equals(ref.index),'Unexpected evaluation universe')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected label horizon')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    check(np.isfinite(target).all(),'Nonfinite targets')
    old=pd.read_csv(oldpath,index_col='decision_time'); old.index=pd.to_datetime(old.index,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamps mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference direction mismatch')
    x=x[COLS].replace([np.inf,-np.inf],np.nan)
    packages=['numpy','pandas','scipy','joblib','threadpoolctl']
    if args.model=='gru':
        import torch
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        packages+=['torch']
        r=None
    else:
        packages+=['duckdb']
        module_path=week1/'Code/canonical_return_features.py'
        spec=importlib.util.spec_from_file_location('canonical_returns_for_experiment',module_path)
        cr=importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
        r=cr.load_returns(week1/'Dataset/canonical_returns_5m.parquet')
        lab=cr.make_labels(r,y.index,48)
        check(lab.index.equals(y.index),'Canonical labels missing')
        np.testing.assert_allclose(lab.signed_return_log_bps,y.signed_return_log_bps,rtol=0,atol=1e-8)
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(model=args.model,target='positive ordinary-bps 48h endpoint magnitude conditional on direction',
        starts=list(STARTS),purge='label_end strictly before relevant cutoff',features=COLS if args.model=='gru' else 'canonical 5m log returns',
        gru=dict(steps=STEPS,stride_minutes=30,hidden=16,seed=SEED,max_epochs=MAX_EPOCHS,patience=PATIENCE,
            selection='earlier validation MAE only; IC reported, not optimized; one fixed recipe, no grid search'),
        garch=dict(parameter_days=180,mapping_days=90,model='zero-mean Gaussian GARCH(1,1)',
            mapping='separate side-specific MAE-optimal positive scalar; params estimated strictly before mapping origins',
            selection='fixed recipe; validation descriptive; state updates causally inside test, parameters/scales frozen',
            estimator='SciPy approximate Gaussian QML for all folds, three fixed training-likelihood starting points',
            missing_filter_returns='propagate expected variance across missing 5m intervals in fitting and filtering; skip missing likelihood terms; log counts and longest gap'),
        limitation='New fits on repeatedly inspected historical periods; exploratory, not an untouched holdout. No direction model.',
        versions={p:importlib.metadata.version(p) for p in packages},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    frames=[]; validation=[]
    for issue in STARTS:
        inner=issue-pd.DateOffset(months=3)
        ik=y.index[y.label_end<inner]; ot=y.index[y.label_end<issue]
        # Same first-two-validation-month selection period as original raw models.
        vk=y.index[(y.index>=inner)&(y.index<issue-pd.DateOffset(months=1))&(y.label_end<issue)]
        end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(minutes=30)
        te=dev[(dev>=issue)&(dev<end)]
        check(len(te)>0 and len(vk)>0 and y.loc[ik,'label_end'].max()<inner and y.loc[ot,'label_end'].max()<issue,'Invalid purged windows')
        check(pd.to_datetime(old.loc[te,'issue'],utc=True).eq(issue).all(),'Reference folds differ')
        print(f'START {args.model.upper()} {issue.date()}',flush=True)
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        if args.model=='garch':
            vp,ia=garch_predict(r,y,target,side,inner,vk)
            tp,oa=garch_predict(r,y,target,side,issue,te)
            dump(out/f'{issue:%Y-%m}_garch.json',dict(inner=ia,outer=oa))
        for direction in ['UP','DOWN']:
            tr=ot[side.loc[ot].eq(direction)]; ti=ik[side.loc[ik].eq(direction)]; va=vk[side.loc[vk].eq(direction)]
            check(min(len(tr),len(ti),len(va))>=100,'Insufficient side-specific samples')
            if args.model=='gru':
                _,v,inner_art=gru_fit_predict(x,target,ti,va,va)
                epochs=inner_art['epochs']
                print(f'  {direction}: selected {epochs} epochs; outer fit begins',flush=True)
                # Outer fit fixed epochs: validation loader uses training rows only,
                # and its outputs are never used for selection or reported as validation.
                t,_,outer_art=gru_fit_predict(x,target,tr,tr[:min(256,len(tr))],te,epochs=epochs)
                joblib.dump(dict(inner=inner_art,outer=outer_art),out/f'{issue:%Y-%m}_{direction}_gru.joblib',compress=3)
                del inner_art,outer_art; gc.collect()
            else:
                v=vp[direction][side.loc[vk].eq(direction).to_numpy()]; t=tp[direction]
            validation.append(dict(issue=str(issue),side=direction,model=args.model,**metrics(target.loc[va],v)))
            frame[f'{direction}_{args.model}']=t
            for kind,value in [('mean',target.loc[tr].mean()),('median',target.loc[tr].median())]:
                frame[f'{direction}_baseline_{kind}']=float(value)
                np.testing.assert_allclose(old.loc[te,f'{direction.lower()}_train_{kind}_baseline_bps'],value,rtol=0,atol=1e-8)
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}_raw']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time'); frames.append(frame)
        pd.DataFrame(validation).to_csv(out/'validation_scores.csv',index=False)
        print(f'DONE {issue.date()}',flush=True)
    pred=pd.concat(frames).sort_index(); check(pred.index.equals(old.index),'Final index mismatch')
    rows=[]
    for fold,f in [('POOLED',pred)]+[(k,g) for k,g in pred.groupby('issue')]:
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)]; actual=g.actual_magnitude_bps
            bm=metrics(actual,g[f'{direction}_baseline_mean'])['mae_bps']
            bd=metrics(actual,g[f'{direction}_baseline_median'])['mae_bps']
            for col in [c for c in pred if c.startswith(direction+'_')]:
                met=metrics(actual,g[col]); rows.append(dict(issue=fold,side=direction,model=col[len(direction)+1:],**met,
                    mae_gain_vs_mean_bps=bm-met['mae_bps'],mae_gain_vs_median_bps=bd-met['mae_bps']))
    scores=pd.DataFrame(rows)
    scores.to_csv(out/'scores.csv',index=False); pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('MAE lower is better; Rank IC higher positive is better.\n'
        'Positive MAE gains mean improvement. Constant within-fold baseline IC is undefined.\n'
        'These are separate side models evaluated on matching realized directions, not a direction predictor.\n'
        'Historical exposed evaluation; no significance claims or trading results.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',model=args.model,rows=len(pred),folds=4))
    print(summary); print(f'Outputs: {out}')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
