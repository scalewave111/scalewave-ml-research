"""Read-only paired calendar-block bootstrap of saved conditional forecasts.
Save in Week 2/Code. Dependencies: numpy pandas scipy. No fitting or tuning.
Resamples nonoverlapping calendar blocks within each fold. Fold contributions
retain their original matching-direction row weights. Partial blocks retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def require(ok,message):
    if not ok: raise ValueError(message)


def load(path):
    f=pd.read_csv(path,index_col='decision_time')
    f.index=pd.to_datetime(f.index,utc=True)
    f['issue']=pd.to_datetime(f['issue'],utc=True)
    require(f.index.is_unique and f.index.is_monotonic_increasing,'Duplicate or unsorted timestamps')
    return f


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    require(len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid scores')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=ic)


def bootstrap_fold(sums,counts,draws,rng):
    """Paired loss-difference sums and side-row counts travel together."""
    sums=np.asarray(sums,float); counts=np.asarray(counts,int)
    result=np.empty(draws); invalid=0
    for start in range(0,draws,250):
        size=min(250,draws-start)
        sample=rng.integers(0,len(counts),size=(size,len(counts)))
        den=counts[sample].sum(axis=1); num=sums[sample].sum(axis=1)
        bad=den==0
        # Conditional sample has no side rows: undefined; never replace with zero.
        invalid+=int(bad.sum())
        result[start:start+size]=np.divide(num,den,out=np.full(size,np.nan),where=~bad)
    return result,invalid


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--gru',type=Path,help='Override GRU predictions.csv')
    ap.add_argument('--original',type=Path,help='Override original predictions.csv')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--draws',type=int,default=5000)
    args=ap.parse_args(); workspace=args.workspace.resolve()
    require(args.draws>=1000,'Use at least 1000 bootstrap draws')
    gp=args.gru or workspace/'Code/experiments/gru_conditional_mae_v1/predictions.csv'
    op=args.original or workspace.parent/'Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    out=args.out or workspace/'Code/experiments/mae_uncertainty_review_v1'
    require(not out.exists(),f'Output already exists: {out}. Use --out with a new folder.')
    g=load(gp); o=load(op)
    require(g.index.equals(o.index) and len(g)==18481,'Forecast timestamps or row counts differ')
    require(g.issue.equals(o.issue) and g.actual_direction.equals(o.actual_direction),'Fold or direction mismatch')
    require(g.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    np.testing.assert_allclose(g.actual_magnitude_bps,o.actual_magnitude_bps,rtol=0,atol=1e-8)
    starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
    require(set(g.issue)==set(starts),'Unexpected folds')
    for i,issue in enumerate(starts):
        ix=g.index[g.issue.eq(issue)]
        require((ix>=issue).all(),'Invalid fold origin')
        if i+1<len(starts): require((ix<starts[i+1]).all(),'Fold overlaps next fold')
    modelcols={}
    for side in ['UP','DOWN']:
        s=side.lower(); cols={'gru':f'{side}_gru'}
        for model in ['mlp','hybrid']:
            col=f'{side}_existing_{model}_raw'
            np.testing.assert_allclose(g[col],o[f'{model}_{s}_magnitude_bps'],rtol=0,atol=1e-8)
            cols[model]=col
        for base in ['mean','median']:
            col=f'{side}_baseline_{base}'
            np.testing.assert_allclose(g[col],o[f'{s}_train_{base}_baseline_bps'],rtol=0,atol=1e-8)
            cols[base]=col
            require(all(f[col].nunique()==1 for _,f in g.groupby('issue')),'Baseline not constant in fold')
        require(np.isfinite(g[list(cols.values())].to_numpy(float)).all(),'Nonfinite predictions')
        modelcols[side]=cols
    require(np.isfinite(g.actual_magnitude_bps).all(),'Nonfinite target')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    comparisons=[('UP','gru','mlp'),('UP','gru','mean'),('UP','gru','median'),
                 ('UP','mlp','mean'),('UP','mlp','median'),
                 ('DOWN','hybrid','mean'),('DOWN','hybrid','median'),('DOWN','hybrid','mlp')]
    score_rows=[]; fold_gain=[]; intervals=[]; block_rows=[]
    for side in ['UP','DOWN']:
        for fold,f in [('POOLED',g)]+[(str(k),v) for k,v in g.groupby('issue')]:
            f=f[f.actual_direction.eq(side)]
            for name,col in modelcols[side].items():
                score_rows.append(dict(issue=fold,side=side,model=name,**metrics(f.actual_magnitude_bps,f[col])))
    for side,candidate,reference in comparisons:
        a=modelcols[side][candidate]; b=modelcols[side][reference]
        mask=g.actual_direction.eq(side)
        losses=np.abs(g[b]-g.actual_magnitude_bps)-np.abs(g[a]-g.actual_magnitude_bps)
        gain=float(losses[mask].mean())
        folds=[]
        for issue,f in g.groupby('issue'):
            m=f.actual_direction.eq(side)
            row=dict(issue=str(issue),side=side,candidate=candidate,reference=reference,
                     rows=int(m.sum()),mae_gain_bps=float(losses.loc[f.index][m].mean()))
            fold_gain.append(row); folds.append((issue,f,m))
        for days in [7,14,28]:
            # Reset gives paired draws across comparisons with the same calendar.
            rng=np.random.default_rng(48149+days)
            combined=np.zeros(args.draws); zero_draws=0
            for issue,f,m in folds:
                block=((f.index-issue)//pd.Timedelta(days=days)).astype(int)
                temp=pd.DataFrame({'block':block,'count':m.to_numpy().astype(int),
                    'loss_sum':np.where(m.to_numpy(),losses.loc[f.index].to_numpy(),0.)})
                agg=temp.groupby('block')[['count','loss_sum']].sum()
                samples,invalid=bootstrap_fold(agg.loss_sum,agg['count'],args.draws,rng)
                combined+=samples*(int(m.sum())/int(mask.sum()))
                zero_draws+=invalid
                block_rows.append(dict(issue=str(issue),side=side,candidate=candidate,reference=reference,
                    block_days=days,calendar_blocks=len(agg),blocks_with_side_rows=int((agg['count']>0).sum()),
                    side_rows=int(m.sum()),smallest_block_side_rows=int(agg['count'].min()),
                    largest_block_side_rows=int(agg['count'].max())))
            valid=combined[np.isfinite(combined)]
            require(len(valid)>=.95*args.draws,'Too many undefined bootstrap replicates')
            lo,hi=np.quantile(valid,[.025,.975])
            intervals.append(dict(side=side,candidate=candidate,reference=reference,block_days=days,
                mae_gain_bps=gain,ci95_low_bps=float(lo),ci95_high_bps=float(hi),
                valid_draws=len(valid),undefined_fold_draws=zero_draws,
                interpretation='improvement interval above zero' if lo>0 else 'deterioration interval below zero' if hi<0 else 'interval crosses zero'))
        print(f'Reviewed {side}: {candidate} versus {reference}',flush=True)
    tables={'scores':score_rows,'fold_gains':fold_gain,'paired_block_intervals':intervals,'block_counts':block_rows}
    for name,rows in tables.items(): pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    summary=('PAIRED MAE UNCERTAINTY REVIEW\n'
        'Gain = reference MAE minus candidate MAE; positive means candidate improves.\n'
        '95% paired percentile intervals; 7/14/28-day calendar blocks resampled within folds.\n'
        'Fold weights fixed at original direction-specific row counts; partial end blocks retained.\n'
        'Same resampled blocks carry candidate/reference errors and side membership together.\n'
        '28-day results use few blocks per fold and may be unstable. Longer dependence can remain.\n'
        'Conditional on saved fits: no refitting uncertainty or model-selection correction.\n'
        'No multiple-comparison correction. Historical tests already informed model choices.\n'
        'An interval above zero is exploratory evidence, not independent confirmation.\n'
        'Rank IC is descriptive in scores.csv; no IC confidence intervals are computed.\n'
        'No direction classifier or trading simulation. No training or input changes.\n\n'
        +pd.DataFrame(intervals).to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump=dict(status='COMPLETED',new_fits=0,rows=len(g),draws=args.draws,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [gp,op]})
    (out/'COMPLETE.json').write_text(json.dumps(dump,indent=2),encoding='utf-8')
    print(summary); print(f'\nOutputs: {out}')


if __name__=='__main__': main()
