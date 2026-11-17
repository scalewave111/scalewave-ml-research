"""Read-only saved-forecast review by 168h volatility availability.
Save in Week 2/Code. No training, replay, fitting, tuning, or input changes.
Dependencies: numpy pandas scipy joblib. No PyTorch required.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib

FEATURE='PRICE__realized_vol_168h'
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)


def check(ok,msg):
    if not ok: raise ValueError(msg)


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid score arrays')
    ic=pd.Series(y).corr(pd.Series(p),method='spearman') if len(y)>=3 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=ic,
                mean_actual_bps=float(y.mean()),median_actual_bps=float(np.median(y)),
                mean_prediction_bps=float(p.mean()),mean_bias_bps=float((p-y).mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    out=args.out or ws/'Code/experiments/volatility_missing_error_review_v1'
    check(not out.exists(),f'Preserving {out}; use --out with a new folder')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid dataset index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected test population')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected target horizon')
    check(FEATURE in x,'Missing volatility feature')
    source=w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    p=pd.read_csv(source,index_col='decision_time');p.index=pd.to_datetime(p.index,utc=True)
    check(p.index.equals(dev.sort_values()),'Saved forecast timestamp mismatch')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    np.testing.assert_allclose(p.actual_magnitude_bps,target.loc[p.index],rtol=0,atol=1e-8)
    check(np.array_equal(p.actual_direction,side.loc[p.index]),'Direction mismatch')
    p['issue']=pd.to_datetime(p.issue,utc=True)
    check(set(p.issue.unique())==set(STARTS),'Unexpected folds')
    check(p.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    p['volatility_status']=np.where(np.isfinite(x.loc[p.index,FEATURE].to_numpy(float)),'available','missing')
    records=[];contributions=[];daily=[];row_frames=[]
    print('Reading saved forecasts only. No fitting or training.',flush=True)
    for direction,model in [('UP','mlp'),('DOWN','hybrid')]:
        s=p[p.actual_direction.eq(direction)].copy();prefix=direction.lower()
        s['prediction']=s[f'{model}_{prefix}_magnitude_bps']
        for issue,g in s.groupby('issue'):
            expected_end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(nanoseconds=1)
            check(((g.index>=issue)&(g.index<expected_end)).all(),'Fold bounds mismatch')
            tr=y.index[(y.label_end<issue)&side.eq(direction)]
            for base in ['mean','median']:
                value=float(getattr(target.loc[tr],base)())
                np.testing.assert_allclose(g[f'{prefix}_train_{base}_baseline_bps'],value,rtol=0,atol=1e-8)
        s['absolute_error_bps']=(s.prediction-s.actual_magnitude_bps).abs()
        for base in ['mean','median']:
            s[f'{base}_absolute_error_bps']=(s[f'{prefix}_train_{base}_baseline_bps']-s.actual_magnitude_bps).abs()
        row_frames.append(s[['issue','actual_direction','volatility_status','actual_magnitude_bps','prediction',
                             'absolute_error_bps','mean_absolute_error_bps','median_absolute_error_bps']])
        for issue,f in [('POOLED',s)]+list(s.groupby('issue')):
            issue=str(issue);total_error=float(f.absolute_error_bps.sum())
            groups=[('all',f)]+list(f.groupby('volatility_status'))
            for status,g in groups:
                # Empty groups are omitted; IC for constant forecasts is undefined.
                basemae={b:float(g[f'{b}_absolute_error_bps'].mean()) for b in ['mean','median']}
                for name,col in [(model,'prediction'),('baseline_mean',f'{prefix}_train_mean_baseline_bps'),
                                 ('baseline_median',f'{prefix}_train_median_baseline_bps')]:
                    m=metrics(g.actual_magnitude_bps,g[col])
                    records.append(dict(issue=issue,side=direction,status=status,model=name,**m,
                        mae_gain_vs_mean_bps=basemae['mean']-m['mae_bps'],
                        mae_gain_vs_median_bps=basemae['median']-m['mae_bps']))
                if status!='all':
                    contributions.append(dict(issue=issue,side=direction,model=model,status=status,rows=len(g),
                        fraction_of_rows=len(g)/len(f),mae_bps=float(g.absolute_error_bps.mean()),
                        fraction_of_total_absolute_error=float(g.absolute_error_bps.sum()/total_error) if total_error>0 else np.nan,
                        contribution_to_overall_mae_bps=float(g.absolute_error_bps.sum()/len(f)),
                        contribution_to_gain_vs_mean_bps=float((g.mean_absolute_error_bps-g.absolute_error_bps).sum()/len(f)),
                        contribution_to_gain_vs_median_bps=float((g.median_absolute_error_bps-g.absolute_error_bps).sum()/len(f))))
            parts=[r for r in contributions if r['issue']==issue and r['side']==direction]
            np.testing.assert_allclose(sum(r['contribution_to_overall_mae_bps'] for r in parts),f.absolute_error_bps.mean(),rtol=0,atol=1e-8)
        for (issue,day,status),g in s.groupby(['issue',s.index.floor('D'),'volatility_status']):
            daily.append(dict(issue=str(issue),day=str(day),side=direction,status=status,model=model,
                              **metrics(g.actual_magnitude_bps,g.prediction)))
        print(f'Reviewed {direction}: {len(s)} rows; {int(s.volatility_status.eq("missing").sum())} missing volatility',flush=True)
    scores=pd.DataFrame(records);contrib=pd.DataFrame(contributions)
    summary=('VOLATILITY AVAILABILITY AND SAVED-FORECAST ERRORS\n'
        'Original raw UP MLP and DOWN hybrid; same targets, rows, and fold-specific training baselines.\n'
        'Positive MAE gain means model beats the baseline on the SAME subgroup.\n'
        'Contribution columns add across missing/available groups to reproduce overall MAE and baseline gains.\n'
        'High absolute error on missing rows alone does not prove imputation caused it: dates and outcomes also differ.\n'
        'Dropping missing rows changes the evaluation population; available-only MAE is NOT an improved full-test result.\n'
        'IC is descriptive; constant forecasts have undefined IC. Pooled baseline IC is not within-fold ranking.\n'
        'Empty subgroups omitted. Exposed historical data; no significance tests, training, selection, or repairs.\n\n'
        +scores[scores.model.isin(['mlp','hybrid'])].to_string(index=False)
        +'\n\nERROR CONTRIBUTIONS\n'+contrib.to_string(index=False))
    out.mkdir(parents=True)
    scores.to_csv(out/'scores.csv',index=False)
    contrib.to_csv(out/'error_contributions.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily_scores.csv',index=False)
    pd.concat(row_frames).sort_index().to_csv(out/'row_errors.csv',index_label='decision_time')
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETED',rows=len(p))),encoding='utf-8')
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__': main()
