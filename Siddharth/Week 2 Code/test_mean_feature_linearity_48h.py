r"""Chronological linear/spline and optional pair-interaction diagnostics.
python .\Code\test_mean_feature_linearity_48h.py --interactions
UP3/DOWN7 current Ridge features. Conditional mean magnitudes, ordinary bps.
"""
from pathlib import Path
import argparse
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as b

HERE=Path(__file__).resolve().parent
IDS=['liq_15m','liq_5m','liq_1h','vol_ratio_4h_24h','vol_168h','liq_breadth_1h','funding_5m','basis_1h']
NAMES=dict(zip(b.COLS,IDS))
FEATURES={'UP':[b.COLS[i] for i in [0,1,3]],'DOWN':[b.COLS[i] for i in [0,1,3,4,5,6,7]]}
ALPHAS=[.001,.01,.1,1.]


def raw_design(x,a):
    z=b.transform(x,a['prep']);k=len(a['prep']['columns'])
    if a['kind']=='spline' and a['spline'] is not None:
        i=a['indices'][0]
        return np.column_stack([np.delete(z,i,axis=1),a['spline'].transform(z[:,[i]])])
    if a['kind']=='interaction':
        i,j=a['indices'];return np.column_stack([z,z[:,i]*z[:,j]])
    return z


def fit(x,target,keys,columns,kind,indices,alpha,knots):
    f=x.loc[keys,columns];median=f.median().fillna(0.);filled=f.fillna(median)
    prep=dict(columns=columns,median=median,center=filled.mean(),
              scale=filled.std(ddof=0).where(lambda s:s>1e-8,1.).fillna(1.))
    a=dict(prep=prep,kind=kind,indices=indices,spline=None,alpha_per_row=alpha,requested_knots=knots)
    if kind=='spline':
        v=b.transform(x.loc[keys],prep)[:,indices[0]]
        locations=np.unique(np.quantile(v,np.linspace(0,1,knots)))
        if len(locations)>=2:
            a['spline']=SplineTransformer(knots=locations.reshape(-1,1),degree=3,
                include_bias=False,extrapolation='linear').fit(v.reshape(-1,1))
        a['actual_knots']=len(locations)
    design=raw_design(x.loc[keys],a)
    a['design_center']=design.mean(axis=0)
    a['design_scale']=np.maximum(design.std(axis=0),1e-8)
    yy=target.loc[keys].to_numpy(float);a['target_center']=float(yy.mean());a['target_scale']=max(float(yy.std()),1.)
    a['model']=Ridge(alpha=alpha*len(keys)).fit((design-a['design_center'])/a['design_scale'],
                                             (yy-a['target_center'])/a['target_scale'])
    return a


def predict(a,x):
    z=(raw_design(x,a)-a['design_center'])/a['design_scale']
    pred=a['model'].predict(z)*a['target_scale']+a['target_center']
    b.check(np.isfinite(pred).all(),'Nonfinite predictions')
    return np.maximum(pred,0.)


def block_interval(frame,days,repeats,seed):
    """Percentile CI for a row-weighted mean, resampling calendar blocks within folds."""
    rng=np.random.default_rng(seed);numer=np.zeros(repeats);denom=np.zeros(repeats);blocks=0
    for _,g in frame.groupby('issue'):
        stamp=pd.DatetimeIndex(g['time']).as_unit('ns').asi8
        group=stamp//pd.Timedelta(days=days).value
        q=pd.DataFrame({'block':group,'value':g.value.to_numpy(float)}).groupby('block').value.agg(['sum','count'])
        n=len(q);blocks+=n
        if not n:continue
        draw=rng.integers(n,size=(repeats,n))
        numer+=q['sum'].to_numpy()[draw].sum(axis=1);denom+=q['count'].to_numpy()[draw].sum(axis=1)
    if blocks<3:return np.nan,np.nan,blocks
    lo,hi=np.quantile(numer/denom,[.025,.975]);return float(lo),float(hi),blocks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interactions',action='store_true',help='Also compare each pair product against the additive linear model')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--block-days',type=int,default=7)
    p.add_argument('--prepared',type=Path,default=HERE.parent.parent/'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    args=p.parse_args()
    b.check(args.threads>0 and args.bootstrap>=200 and args.block_days>=2,'Use positive threads, >=200 bootstrap draws and >=2-day blocks')
    x,y,dev,_=joblib.load(args.prepared)
    b.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid input index')
    x=x[b.COLS].replace([np.inf,-np.inf],np.nan);y=y.copy();y['label_end']=pd.to_datetime(y.label_end,utc=True)
    b.check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Invalid horizon')
    signed=pd.Series(10000*np.expm1(y.signed_return_log_bps/10000),index=y.index);target=signed.abs()
    b.check(np.isfinite(target).all(),'Nonfinite targets')
    dev=pd.DatetimeIndex(dev).sort_values();b.check(dev.is_unique and dev.isin(x.index).all(),'Invalid evaluation origins')
    out=b.EXP/'mean_feature_linearity_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False);(out/'plots').mkdir()
    b.dump(out/'CONTRACT.json',dict(features=FEATURES,feature_labels=NAMES,alpha_per_row=ALPHAS,spline_knots=[3,5],
        target='Conditional mean ordinary-bps positive endpoint magnitudes, separately UP/DOWN',
        method='One feature spline replacing its linear term; others remain linear. Pair products are optional additions to full linear design.',
        selection='Tune linear and each challenger independently on the same3earlier windows, row-weighted MSE',
        inference='Nonnegative projected magnitude; standardization/imputation/knots fit on training only',
        uncertainty=f'{args.bootstrap} calendar-block bootstrap replicates, {args.block_days} days, stratified by outer fit period; pointwise95percent intervals, not multiplicity-adjusted',
        conclusion='Curvature gain requires pooledCI>0 and >=3/4 period improvements; negativeCI implies linear preferred; otherwise inconclusive. These do not prove true linearity.',
        bins='Training-side quantile boundaries; actual later-side means plotted with pointwise block uncertainty, marginal not partial effects',
        limitations='Exposed historical evaluation; correlated inputs and overlapping outcomes; no causal interpretation, no proof of exact linearity',
        interactions=args.interactions,prepared_sha256=b.sha(args.prepared),source_sha256=b.sha(Path(__file__))))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'helper_snapshot.py').write_bytes(Path(b.__file__).read_bytes())
    print('Outputs:',out,flush=True)
    validation=[];selections=[];frames=[];bins=[];audit=[]
    with threadpool_limits(limits=args.threads):
        for fold,issue in enumerate(b.ISSUES):
            stop=b.ISSUES[fold+1] if fold+1<len(b.ISSUES) else dev.max()+pd.Timedelta(nanoseconds=1)
            origins=dev[(dev>=issue)&(dev<stop)]
            for side,columns in FEATURES.items():
                mask=signed>0 if side=='UP' else signed<0
                te=origins[mask.loc[origins]];outer=b.train_keys(y,mask,issue,None)
                specs=[('linear','linear',[])]
                specs += [('spline_'+NAMES[c],'spline',[i]) for i,c in enumerate(columns)]
                if args.interactions:
                    specs += [('interaction_'+NAMES[columns[i]]+'__'+NAMES[columns[j]],'interaction',[i,j])
                              for i,j in itertools.combinations(range(len(columns)),2)]
                f=pd.DataFrame({'time':te,'issue':str(issue),'side':side,'actual':target.loc[te].to_numpy()})
                for name,kind,indices in specs:
                    best=None
                    for knots in ([3,5] if kind=='spline' else [0]):
                        for alpha in ALPHAS:
                            loss,count=0.,0
                            for offset in [9,6,3]:
                                cut=issue-pd.DateOffset(months=offset)
                                tr=b.train_keys(y,mask,cut,None)
                                va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))
                                           &(y.label_end<cut+pd.DateOffset(months=3))&mask]
                                b.check(len(va)>=100 and y.loc[va,'label_end'].max()<issue,'Validation purge failed')
                                a=fit(x,target,tr,columns,kind,indices,alpha,knots)
                                m=b.metric(target.loc[va],predict(a,x.loc[va]));loss+=m['mse_bps2']*len(va);count+=len(va)
                                validation.append(dict(issue=issue,side=side,model=name,alpha=alpha,knots=knots,
                                    offset=offset,max_training_label_end=y.loc[tr,'label_end'].max(),**m))
                            candidate=dict(alpha=alpha,knots=knots,mse=loss/count)
                            if best is None or (candidate['mse'],knots,-alpha)<(best['mse'],best['knots'],-best['alpha']):best=candidate
                    selections.append(dict(issue=issue,side=side,model=name,**best));b.dump(out/'SELECTIONS.json',selections)
                    a=fit(x,target,outer,columns,kind,indices,best['alpha'],best['knots'])
                    a.update(issue=issue,side=side,max_training_label_end=y.loc[outer,'label_end'].max())
                    pred=predict(a,x.loc[te]);f[name]=pred
                    path=out/f'{issue:%Y-%m}_{side}_{name}.joblib';joblib.dump(a,path,compress=3)
                    np.testing.assert_allclose(pred,predict(joblib.load(path),x.loc[te]),rtol=0,atol=1e-6)
                    audit.append(dict(issue=issue,side=side,model=name,training_rows=len(outer),max_training_label_end=a['max_training_label_end'],
                                      artifact=path.name,sha256=b.sha(path),spline_has_variation=a['spline'] is not None if kind=='spline' else None))
                    print(f'DONE {issue.date()} {side} {name}',flush=True)
                frames.append(f)
                # Marginal plots: raw feature bins defined exclusively using training data.
                for c in columns:
                    finite=x.loc[outer,c].dropna();cutpoints=np.unique(finite.quantile(np.arange(.1,1.,.1)).to_numpy())
                    xx=x.loc[te,c];labels=np.searchsorted(cutpoints,xx.to_numpy(),side='right')
                    for label in np.unique(labels[xx.notna()]):
                        keys=te[(labels==label)&xx.notna().to_numpy()]
                        h=pd.DataFrame(dict(time=keys,issue=str(issue),value=target.loc[keys].to_numpy()))
                        lo,hi,nblocks=block_interval(h,args.block_days,args.bootstrap,48)
                        bins.append(dict(issue=str(issue),side=side,feature=NAMES[c],bin=int(label),rows=len(keys),blocks=nblocks,
                            lower_bound=float(cutpoints[label-1]) if label else None,
                            upper_bound=float(cutpoints[label]) if label<len(cutpoints) else None,
                            feature_mean=float(x.loc[keys,c].mean()),actual_mean_bps=float(target.loc[keys].mean()),ci_low=lo,ci_high=hi))
                pd.DataFrame(validation).to_csv(out/'validation_metrics.csv',index=False)
                b.dump(out/'FIT_AUDIT.json',audit)
    results=pd.concat(frames,ignore_index=True);results.to_csv(out/'conditional_predictions.csv',index=False)
    summary=[];fold_metrics=[]
    for side,group in results.groupby('side'):
        names=[c for c in group if (c.startswith('spline_') or c.startswith('interaction_')) and group[c].notna().any()]
        for name in names:
            g=group.loc[group[name].notna()].copy()
            base_error=(g.linear-g.actual)**2;new_error=(g[name]-g.actual)**2
            g['value']=base_error-new_error
            lo,hi,blocks=block_interval(g[['time','issue','value']],args.block_days,args.bootstrap,48)
            wins=0
            for issue,h in g.groupby('issue'):
                baseline=b.metric(h.actual,h.linear);challenger=b.metric(h.actual,h[name])
                gain=baseline['mse_bps2']-challenger['mse_bps2'];wins+=int(gain>0)
                fold_metrics.append(dict(side=side,issue=issue,model=name,mse_gain=gain,
                    linear_rmse=baseline['rmse_bps'],challenger_rmse=challenger['rmse_bps']))
            conclusion=('evidence_of_useful_nonlinearity' if name.startswith('spline') else 'evidence_of_useful_interaction') if lo>0 and wins>=3 else (
                'linear_preferred_in_this_test' if hi<0 else 'inconclusive')
            summary.append(dict(side=side,model=name,rows=len(g),linear_rmse=float(np.sqrt(base_error.mean())),
                challenger_rmse=float(np.sqrt(new_error.mean())),mse_gain=float(g.value.mean()),
                ci_low=lo,ci_high=hi,blocks=blocks,periods_improved=wins,periods=g.issue.nunique(),conclusion=conclusion))
    table=pd.DataFrame(summary);table.to_csv(out/'linearity_summary.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'comparison_by_period.csv',index=False)
    bin_table=pd.DataFrame(bins);bin_table.to_csv(out/'binned_relationships.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (side,feature),g in bin_table.groupby(['side','feature']):
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,(issue,h) in zip(axes.flat,g.groupby('issue')):
            h=h.sort_values('feature_mean');ax.plot(h.feature_mean,h.actual_mean_bps,'o-')
            ax.fill_between(h.feature_mean,h.ci_low,h.ci_high,alpha=.2)
            for row in h.itertuples():ax.annotate(str(row.rows),(row.feature_mean,row.actual_mean_bps),fontsize=6)
            ax.set_title(issue[:10]);ax.set_xlabel('Raw feature value (bin mean)');ax.set_ylabel('Mean actual magnitude, bps');ax.grid(alpha=.2)
        fig.suptitle(f'{side}: {feature} | training-defined bins, later outcomes\nMarginal relationship; labels=count; shading=pointwise block-bootstrap95% CI')
        fig.tight_layout();fig.savefig(out/'plots'/f'{side}_{feature}.png',dpi=150);plt.close(fig)
    text=('LINEARITY DIAGNOSTIC: conditional mean magnitudes, ordinarybps\n'
          'Positive MSE gain means challenger beat additive linear model. Pointwise intervals; historical exploratory evidence.\n'
          'No detected improvement does NOT prove the underlying relationship is linear.\n'+table.to_string(index=False))
    (out/'SUMMARY.txt').write_text(text,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',training_performed=True,model_replay_passed=True,
        rows=len(results),comparisons=len(table),predictions_sha256=b.sha(out/'conditional_predictions.csv'),
        contract_sha256=b.sha(out/'CONTRACT.json')))
    print(text);print('Outputs saved:',out)


if __name__=='__main__':main()
