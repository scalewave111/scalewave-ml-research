"""Read saved MAE-model forecasts; compare mean/median baselines and Rank IC.
Place in Week 2/Code. No fitting, tuning, model loading, or input modification.
Dependencies: numpy, pandas, scipy. Historical results remain exploratory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(actual, predicted):
    y=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float)
    require(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(), 'Invalid scoring values')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(p-y).mean()),rank_ic=ic)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--source',type=Path,help='Override original diagnostics folder')
    ap.add_argument('--out',type=Path,help='Override new output folder')
    args=ap.parse_args(); workspace=args.workspace.resolve()
    source=args.source or workspace.parent/'Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1'
    out=args.out or workspace/'Code/experiments/mae_ic_baseline_review_v1'
    require(not out.exists(),f'Preserving existing output folder: {out}')
    path=source/'predictions.csv'
    df=pd.read_csv(path,index_col='decision_time')
    df.index=pd.to_datetime(df.index,utc=True)
    df['issue']=pd.to_datetime(df['issue'],utc=True)
    starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
    require(df.index.is_unique and df.index.is_monotonic_increasing,'Duplicate/unsorted timestamps')
    require(len(df)==18481 and set(df.issue)==set(starts),'Unexpected rows or fold starts')
    require(df.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected direction counts')
    for i,issue in enumerate(starts):
        g=df[df.issue.eq(issue)]
        require((g.index>=issue).all(),'Test timestamp before issuance')
        if i<len(starts)-1:
            require((g.index<starts[i+1]).all(),'Test timestamp outside fold')
    records=[]; baseline_checks=[]
    for side in ['UP','DOWN']:
        s=side.lower()
        columns={'mean_baseline':f'{s}_train_mean_baseline_bps',
                 'median_baseline':f'{s}_train_median_baseline_bps',
                 'mlp_raw':f'mlp_{s}_magnitude_bps',
                 'hybrid_raw':f'hybrid_{s}_magnitude_bps',
                 'mlp_scaled':f'mlp_{s}_calibrated_magnitude_bps',
                 'hybrid_scaled':f'hybrid_{s}_calibrated_magnitude_bps'}
        require(set(columns.values()).issubset(df.columns),'Missing forecast/baseline columns')
        require(np.isfinite(df[list(columns.values())].to_numpy(dtype=float)).all(),'Nonfinite forecasts')
        for issue,g in df.groupby('issue'):
            for model in ['mean_baseline','median_baseline']:
                require(g[columns[model]].nunique()==1,'Baseline must be constant within fold')
                baseline_checks.append(dict(issue=str(issue),side=side,baseline=model,value_bps=float(g[columns[model]].iloc[0])))
        for issue,g in [('POOLED',df)]+[(str(k),v) for k,v in df.groupby('issue')]:
            h=g[g.actual_direction.eq(side)]; y=h.actual_magnitude_bps
            m={name:metrics(y,h[col]) for name,col in columns.items()}
            for name,met in m.items():
                records.append(dict(issue=issue,side=side,model=name,**met,
                    mae_gain_vs_mean_bps=m['mean_baseline']['mae_bps']-met['mae_bps'],
                    mae_gain_vs_median_bps=m['median_baseline']['mae_bps']-met['mae_bps']))
    scores=pd.DataFrame(records)
    # Cross-check pooled calculations against the original saved score table.
    original=pd.read_csv(source/'scores.csv')
    names={'mean_baseline':'Training mean baseline','median_baseline':'Training median baseline',
           'mlp_raw':'MLP raw','hybrid_raw':'Hybrid raw','mlp_scaled':'MLP validation-scaled',
           'hybrid_scaled':'Hybrid validation-scaled'}
    for row in scores[scores.issue.eq('POOLED')].to_dict('records'):
        old=original[(original.side==row['side'])&(original.model==names[row['model']])]
        require(len(old)==1,'Missing/duplicate original score')
        np.testing.assert_allclose(row['mae_bps'],old.mae_ordinary_bps.iloc[0],rtol=0,atol=1e-7)
        np.testing.assert_allclose(row['rank_ic'],old.rank_ic.iloc[0],rtol=0,atol=1e-7,equal_nan=True)
    stability=[]
    folds=scores[~scores.issue.eq('POOLED')]
    for (side,model),g in folds.groupby(['side','model']):
        valid=g.rank_ic.dropna()
        stability.append(dict(side=side,model=model,folds=len(g),
            mae_wins_vs_mean=int((g.mae_gain_vs_mean_bps>0).sum()),
            mae_wins_vs_median=int((g.mae_gain_vs_median_bps>0).sum()),
            positive_ic_folds=int((valid>0).sum()),defined_ic_folds=len(valid),
            median_fold_ic=float(valid.median()) if len(valid) else np.nan,
            min_fold_ic=float(valid.min()) if len(valid) else np.nan,
            max_fold_ic=float(valid.max()) if len(valid) else np.nan))
    pooled=scores[scores.issue.eq('POOLED')].drop(columns='issue')
    stability=pd.DataFrame(stability)
    out.mkdir(parents=True)
    pooled.to_csv(out/'pooled_scores.csv',index=False)
    folds.to_csv(out/'fold_scores.csv',index=False)
    stability.to_csv(out/'fold_stability.csv',index=False)
    pd.DataFrame(baseline_checks).to_csv(out/'baseline_values.csv',index=False)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    summary=('Positive MAE gain means lower error than the named baseline.\n'
             'Rank IC is Spearman correlation, not an accuracy percentage.\n'
             'Constant within-fold forecasts have undefined IC (blank/NaN), not zero.\n'
             'Pooled baseline IC reflects changes between folds, not within-fold ranking.\n'
             'Fold median IC is descriptive, not pooled IC or a significance test.\n'
             'Raw and scaled predictions are reported; no selection or scaling is fitted.\n'
             'Previously exposed historical data; no uncertainty intervals computed.\n\n'
             +pooled.to_string(index=False)+'\n\n'+stability.to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETED',new_fits=0,
        rows=len(df),pooled_scores_match_original=True,input=str(path.resolve()),
        prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        limitation='Uses saved baseline columns. Does not independently audit training membership or recompute training statistics.'),indent=2),encoding='utf-8')
    print(summary); print(f'\nOutputs: {out}')


if __name__=='__main__':
    main()
