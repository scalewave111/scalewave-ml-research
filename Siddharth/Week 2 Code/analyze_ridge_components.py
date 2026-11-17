"""Replay saved conditional-magnitude models and ablate nonlinear components.

Place in Week 2/Code. No fitting, tuning, or modification of original results.
Requires the same numpy/pandas/scipy/sklearn/torch/joblib environment as training.
Only load your own trusted joblib artifacts. Run without Python's -O option.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits


# Same forward implementation as the supplied training source. Unpickling
# restores layers; __init__ is not invoked. Handles CLI and imported training.
class MagnitudeNet(nn.Module):
    def forward(self, x):
        return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0.)


sys.modules.setdefault('fit_conditional_48h_mean_features', sys.modules[__name__])


def check(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def score(y, p):
    y=np.asarray(y, dtype=float); p=np.asarray(p, dtype=float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(), 'Invalid scoring arrays')
    e=p-y
    r=pd.Series(p).corr(pd.Series(y), method='spearman') if np.ptp(p)>0 else np.nan
    return dict(rows=len(y), mse_bps2=float(np.mean(e**2)), rmse_bps=float(np.sqrt(np.mean(e**2))),
                mae_bps=float(np.mean(abs(e))), mean_predicted_bps=float(p.mean()),
                mean_actual_bps=float(y.mean()), mean_bias_bps=float(e.mean()),
                rank_ic=float(r) if np.isfinite(r) else None)


def mlp_predictions(artifacts, x):
    full=[]; ridge=[]
    for a in artifacts:
        p=a['prep']; raw=x[p['columns']]
        z=((raw.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
        check(np.isfinite(z).all(), 'Nonfinite MLP design')
        model=a['model'].cpu().eval(); f=[]; r=[]
        with torch.no_grad():
            for start in range(0,len(z),4096):
                t=torch.from_numpy(z[start:start+4096])
                f.append(model(t).numpy()*100.)
                r.append(model.linear(t).squeeze(1).clamp(min=0.).numpy()*100.)
        full.append(np.concatenate(f).astype(float))
        ridge.append(np.concatenate(r).astype(float))
    return np.mean(full,axis=0), np.mean(ridge,axis=0)


def hybrid_predictions(saved, x):
    a=saved['artifact']; p=a['prep']; raw=x[p['columns']]
    # Exact design from targeted_feature_extension.py: standardized values
    # followed by missing indicators; no clipping of standardized inputs.
    v=(raw.fillna(p['median'])-p['center'])/p['scale']
    z=np.column_stack([v.to_numpy(dtype=float),raw.isna().to_numpy(dtype=float)])
    check(np.isfinite(z).all(), 'Nonfinite hybrid design')
    r=np.asarray(a['models']['pair'].predict(z),dtype=float)
    a['models']['rf'].n_jobs=1
    f=np.asarray(a['models']['rf'].predict(z),dtype=float)
    w=float(saved['weight'])
    return np.maximum(0,(1-w)*r+w*f), np.maximum(0,r), w


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,help='Optional new output directory')
    args=parser.parse_args(); workspace=args.workspace.resolve()
    src=workspace/'Code/experiments/conditional_48h_mean_feature_comparison_v1'
    data=workspace.parent/'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    out=(args.out or workspace/'Code/experiments/ridge_component_review_v1').resolve()
    check(not out.exists(), f'Output already exists; preserving it: {out}')
    check((src/'COMPLETE.json').is_file(), 'Original experiment is not marked complete')
    required=[data,src/'predictions.csv',src/'SELECTIONS.json']
    for month in ['2024-01','2024-04','2024-07','2024-10']:
        for side in ['UP','DOWN']:
            for pack in ['four','five','eight']:
                for suffix in ['mlp.joblib','hybrid.joblib','settings.json']:
                    required.append(src/f'{month}_{side}_{pack}_{suffix}')
    missing=[str(p) for p in required if not p.is_file()]
    check(not missing, 'Missing inputs:\n'+'\n'.join(missing))
    print('Read-only replay and Ridge ablation. No fitting or tuning.',flush=True)
    x,y,dev,ref=joblib.load(data)
    check(x.index.equals(y.index) and x.index.is_unique, 'Dataset index mismatch')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(), 'Unexpected label horizon')
    pred=pd.read_csv(src/'predictions.csv',index_col='decision_time')
    pred.index=pd.to_datetime(pred.index,utc=True)
    pred['issue']=pd.to_datetime(pred['issue'],utc=True)
    check(pred.index.equals(dev.sort_values()) and len(pred)==18481, 'Saved forecast timestamps mismatch')
    signed=y.loc[pred.index,'signed_return_log_bps'].astype(float)
    truth=np.abs(10000*np.expm1(signed.to_numpy()/10000))
    np.testing.assert_allclose(pred.actual_magnitude_bps,truth,rtol=0,atol=1e-8)
    expected=np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT'))
    check(np.array_equal(pred.actual_direction,expected), 'Direction mismatch')
    selections=json.loads((src/'SELECTIONS.json').read_text())
    x=x.replace([np.inf,-np.inf],np.nan)
    starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
    check(set(pred.issue)==set(starts), 'Unexpected fold starts')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(status='STARTED',training=False,selection=False,
        description='Clipped saved Ridge components; selected variants inherit original full-model feature choices, not Ridge-optimal choices.',
        limitation='Exposed historical data; descriptive comparisons, no significance claims or new uncertainty intervals.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=[str(p) for p in required]))
    audits=[]; weights=[]
    for issue in starts:
        te=pred.index[pred.issue.eq(issue)]
        end=issue+pd.DateOffset(months=3) if issue!=starts[-1] else dev.max()+pd.Timedelta(minutes=30)
        check(te.equals(dev[(dev>=issue)&(dev<end)]), 'Fold membership mismatch')
        for side in ['UP','DOWN']:
            tr=y.index[(y.label_end<issue)&((y.signed_return_log_bps>0) if side=='UP' else (y.signed_return_log_bps<0))]
            mean=float(np.abs(10000*np.expm1(y.loc[tr,'signed_return_log_bps'].to_numpy()/10000)).mean())
            np.testing.assert_allclose(pred.loc[te,f'{side}_baseline_mean'],mean,rtol=0,atol=1e-8)
            for pack in ['four','five','eight']:
                tag=f'{issue:%Y-%m}_{side}_{pack}'
                settings=json.loads((src/f'{tag}_settings.json').read_text())
                check(pd.Timestamp(settings['outer_max_label_end'])<issue, 'Saved purge metadata failed')
                check(pd.Timestamp(settings['inner_max_label_end'])<issue-pd.DateOffset(months=3), 'Inner purge metadata failed')
                for family in ['mlp','hybrid']:
                    a=joblib.load(src/f'{tag}_{family}.joblib')
                    if family=='mlp':
                        check(len(a)==3 and all(z['prep']['columns']==settings['columns'] for z in a), 'MLP feature/seed mismatch')
                        full,ridge=mlp_predictions(a,x.loc[te])
                    else:
                        check(a['artifact']['prep']['columns']==settings['columns'], 'Hybrid feature mismatch')
                        full,ridge,w=hybrid_predictions(a,x.loc[te])
                        check(w==float(settings['rf_weight']), 'Blend weight mismatch')
                        weights.append(dict(issue=str(issue),side=side,pack=pack,rf_weight=w,epochs=settings['epochs']))
                    saved=pred.loc[te,f'{side}_{family}_{pack}'].to_numpy()
                    error=float(np.max(np.abs(full-saved)))
                    # MLP float32 replay may vary slightly across CPU libraries.
                    tolerance=1e-3 if family=='mlp' else 1e-7
                    np.testing.assert_allclose(full,saved,rtol=0,atol=tolerance,
                        err_msg=f'Replay failed: {tag} {family}; do not interpret this run')
                    audits.append(dict(issue=str(issue),side=side,pack=pack,family=family,max_abs_error_bps=error,tolerance_bps=tolerance))
                    pred.loc[te,f'{side}_ridge_{family}_{pack}']=ridge
                    del a; gc.collect()
                print(f'Replayed {tag}',flush=True)
            for family in ['mlp','hybrid']:
                chosen=[s for s in selections if pd.Timestamp(s['issue'])==issue and s['side']==side and s['model']==family]
                check(len(chosen)==1,'Missing/duplicate feature selection')
                pack=chosen[0]['pack']
                np.testing.assert_allclose(pred.loc[te,f'{side}_{family}_selected'],pred.loc[te,f'{side}_{family}_{pack}'],rtol=0,atol=1e-8)
                pred.loc[te,f'{side}_ridge_{family}_selected']=pred.loc[te,f'{side}_ridge_{family}_{pack}']
    rows=[]; comparisons=[]; calibration=[]
    for side in ['UP','DOWN']:
        for fold,g in [('POOLED',pred)]+[(str(k),v) for k,v in pred.groupby('issue')]:
            mask=g.actual_direction.eq(side); actual=g.loc[mask,'actual_magnitude_bps']
            base=score(actual,g.loc[mask,f'{side}_baseline_mean'])
            rows.append(dict(issue=fold,side=side,model='baseline_mean',**base))
            for family in ['mlp','hybrid']:
                for pack in ['four','five','eight','selected']:
                    stats={}
                    for variant,prefix in [('full',family),('ridge','ridge_'+family)]:
                        col=f'{side}_{prefix}_{pack}'; p=g.loc[mask,col]
                        s=score(actual,p); stats[variant]=s
                        rows.append(dict(issue=fold,side=side,model=prefix+'_'+pack,
                                         mse_gain_vs_mean=base['mse_bps2']-s['mse_bps2'],**s))
                        if fold!='POOLED':
                            # Cutpoints use all test forecasts, then evaluate matching direction.
                            cuts=g[col].quantile([1/3,2/3]).to_numpy()
                            labels=np.where(p<=cuts[0],'low',np.where(p<=cuts[1],'medium','high'))
                            for label in ['low','medium','high']:
                                m=labels==label
                                if m.any():
                                    calibration.append(dict(issue=fold,side=side,model=prefix+'_'+pack,group=label,**score(actual.to_numpy()[m],p.to_numpy()[m])))
                    comparisons.append(dict(issue=fold,side=side,family=family,pack=pack,
                        full_mse=stats['full']['mse_bps2'],ridge_mse=stats['ridge']['mse_bps2'],
                        mse_gain_from_nonlinear=stats['ridge']['mse_bps2']-stats['full']['mse_bps2'],
                        ridge_gain_vs_mean=base['mse_bps2']-stats['ridge']['mse_bps2']))
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    for name,records in [('scores',rows),('component_comparison',comparisons),('calibration_by_fold',calibration),('replay_audit',audits),('saved_weights',weights)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    table=pd.DataFrame(comparisons); pooled=table[table.issue.eq('POOLED')]
    summary=('Positive mse_gain_from_nonlinear means full model beats its own Ridge component.\n'
             'Positive ridge_gain_vs_mean means Ridge beats the training-mean baseline.\n'
             'Selected Ridge inherits full-model selections; not independently selected.\n'
             'Descriptive exposed-test results; replay verified, no fits or new statistical intervals.\n\n'+pooled.to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',new_fits=0,replay_checks=len(audits),rows=len(pred),
        limitation='Purge metadata and baseline checked; not a full audit of original fitting or feature causality.'))
    print(summary); print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
