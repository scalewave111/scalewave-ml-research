"""Read-only analysis of saved forecasts. No fitting or training.
Run from Week 2: python .\Code\analyze_conditional_mean_results.py
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'Code/experiments/conditional_48h_mean_feature_comparison_v1'
OUT=ROOT/'Code/experiments/conditional_mean_review_v1'
REPEATS=5000
SEED=20261110
MODELS=[f'{family}_{pack}' for family in ['mlp','hybrid'] for pack in ['four','five','eight','selected']]

def measure(a,p,b):
    e=p-a;be=b-a
    return dict(rows=len(a),mae_bps=float(np.abs(e).mean()),rmse_bps=float(np.sqrt(np.mean(e**2))),
        mse_bps2=float(np.mean(e**2)),baseline_mae_bps=float(np.abs(be).mean()),
        baseline_rmse_bps=float(np.sqrt(np.mean(be**2))),
        mse_gain_bps2=float(np.mean(be**2-e**2)),mae_gain_bps=float(np.mean(np.abs(be)-np.abs(e))),
        mean_predicted_bps=float(p.mean()),mean_actual_bps=float(a.mean()),mean_bias_bps=float(e.mean()))

def bootstrap(frame,side,model,days):
    # Keep calendar adjacency before filtering direction. Resample paired losses
    # in nonoverlapping calendar blocks, independently within each outer fold.
    rng=np.random.default_rng(SEED+days)
    totals=np.zeros((REPEATS,3));block_counts=[]
    for issue,g in frame.groupby('issue',sort=True):
        mask=g.actual_direction.eq(side).to_numpy()
        a=g.actual_magnitude_bps.to_numpy(float)
        p=g[f'{side}_{model}'].to_numpy(float);b=g[f'{side}_baseline_mean'].to_numpy(float)
        block=((g.index-pd.Timestamp(issue))//pd.Timedelta(days=days)).to_numpy()
        d=pd.DataFrame({'block':block,'n':mask.astype(int),
            'mse_gain':np.where(mask,(b-a)**2-(p-a)**2,0.),
            'mae_gain':np.where(mask,np.abs(b-a)-np.abs(p-a),0.)})
        aggregates=d.groupby('block')[['n','mse_gain','mae_gain']].sum().to_numpy(float)
        n=len(aggregates);block_counts.append(n)
        draws=rng.integers(0,n,size=(REPEATS,n))
        totals+=aggregates[draws].sum(axis=1)
    assert (totals[:,0]>0).all()
    mse=totals[:,1]/totals[:,0];mae=totals[:,2]/totals[:,0]
    return dict(side=side,model=model,block_days=days,blocks_by_fold=json.dumps(block_counts),
        mse_gain_lower95=float(np.quantile(mse,.025)),mse_gain_upper95=float(np.quantile(mse,.975)),
        mae_gain_lower95=float(np.quantile(mae,.025)),mae_gain_upper95=float(np.quantile(mae,.975)))

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving nonempty output folder: {OUT}')
    frame=pd.read_csv(SRC/'predictions.csv',index_col='decision_time',parse_dates=True)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert len(frame)==18481 and frame.issue.nunique()==4
    assert frame.actual_direction.isin(['UP','DOWN','FLAT']).all()
    numeric=['actual_magnitude_bps']+[f'{s}_{m}' for s in ['UP','DOWN'] for m in MODELS+['baseline_mean']]
    assert np.isfinite(frame[numeric].to_numpy(float)).all()
    assert (frame[numeric].to_numpy(float)>=0).all()
    saved=pd.read_csv(SRC/'scores.csv')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    folds=[];pooled=[];calibration=[];intervals=[];bin_rows=[]
    for side in ['UP','DOWN']:
        for model in MODELS:
            col=f'{side}_{model}'
            g=frame[frame.actual_direction.eq(side)]
            a=g.actual_magnitude_bps.to_numpy(float);p=g[col].to_numpy(float)
            b=g[f'{side}_baseline_mean'].to_numpy(float)
            result=measure(a,p,b)
            ref=saved[(saved.side==side)&(saved.model==model)].iloc[0]
            for k in ['mae_bps','rmse_bps','mse_bps2','mean_bias_bps']:
                np.testing.assert_allclose(result[k],float(ref[k]),rtol=1e-9,atol=1e-7)
            pooled.append(dict(side=side,model=model,**result))
            for issue,g in frame.groupby('issue',sort=True):
                # Descriptive bins use forecasts only, from all timestamps in
                # the fold. They are not learned thresholds for future use.
                lo,hi=np.quantile(g[col],[1/3,2/3])
                labels=np.where(g[col]<=lo,'low',np.where(g[col]<=hi,'medium','high'))
                mask=g.actual_direction.eq(side)
                z=g.loc[mask].copy();z['forecast_group']=labels[mask.to_numpy()]
                folds.append(dict(issue=issue,side=side,model=model,**measure(z.actual_magnitude_bps.to_numpy(float),z[col].to_numpy(float),z[f'{side}_baseline_mean'].to_numpy(float))))
                for label in ['low','medium','high']:
                    h=z[z.forecast_group.eq(label)]
                    if h.empty:continue
                    calibration.append(dict(issue=issue,side=side,model=model,forecast_group=label,
                        lower_cut_bps=float(lo),upper_cut_bps=float(hi),rows=len(h),
                        mean_predicted_bps=float(h[col].mean()),mean_actual_bps=float(h.actual_magnitude_bps.mean()),
                        mean_bias_bps=float((h[col]-h.actual_magnitude_bps).mean())))
            for days in [7,14]:
                intervals.append(bootstrap(frame,side,model,days))
            print(f'Analyzed {side} {model}',flush=True)
    fold=pd.DataFrame(folds);pool=pd.DataFrame(pooled);cal=pd.DataFrame(calibration);ci=pd.DataFrame(intervals)
    for (side,model,group),g in cal.groupby(['side','model','forecast_group']):
        weights=g.rows.to_numpy(float)
        bin_rows.append(dict(side=side,model=model,forecast_group=group,rows=int(weights.sum()),
            mean_predicted_bps=float(np.average(g.mean_predicted_bps,weights=weights)),
            mean_actual_bps=float(np.average(g.mean_actual_bps,weights=weights)),
            mean_bias_bps=float(np.average(g.mean_bias_bps,weights=weights))))
    for name,data in [('fold_scores',fold),('pooled_scores',pool),('calibration_by_fold',cal),('calibration_pooled',pd.DataFrame(bin_rows)),('paired_block_uncertainty',ci)]:
        data.to_csv(OUT/f'{name}.csv',index=False)
    lines=['Conditional mean forecast review','Positive gain means model error is LOWER than the training-mean baseline.','']
    for r in pool.to_dict('records'):
        f=fold[(fold.side==r['side'])&(fold.model==r['model'])]
        c=ci[(ci.side==r['side'])&(ci.model==r['model'])&(ci.block_days==14)].iloc[0]
        lines.append(f"{r['side']} {r['model']}: MSE gain {r['mse_gain_bps2']:.2f}; wins {int((f.mse_gain_bps2>0).sum())}/4 folds; 14-day 95% interval [{c.mse_gain_lower95:.2f}, {c.mse_gain_upper95:.2f}]")
    lines+=['','Interpretation:',
        '- A gain interval spanning zero does not establish an improvement.',
        '- Calibration tables compare predicted and actual means for low/medium/high forecasts within each fold. Empty groups are omitted when forecasts tie.',
        '- Monotonic actual group means suggest ranking signal; close predicted/actual means suggest group-level calibration. Neither establishes individual conditional means.',
        '- Block intervals are approximate paired percentile intervals conditional on saved fits. They do not include refitting, selection uncertainty or multiple-comparison correction.',
        '- Seven- and fourteen-day calendar blocks exceed the 48-hour outcome horizon; longer dependence can remain. Fold-end partial blocks are retained.',
        '- Historical test data has prior exposure. Fixed feature-set winners picked after seeing test results are exploratory.',
        '- No training, calibration fitting, direction model, or trading simulation was performed.']
    (OUT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    contract=dict(status='COMPLETE',repeats=REPEATS,seed=SEED,block_days=[7,14],
        method='Paired nonoverlapping calendar-block bootstrap, stratified by outer fold, row-weighted pooled losses',
        input_sha256=hashlib.sha256((SRC/'predictions.csv').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scores_reproduced=True,training_performed=False,
        bins='Per-fold forecast tertiles from all test timestamps; descriptive only; outcomes never determine thresholds')
    (OUT/'COMPLETE.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(f'Outputs saved to {OUT}',flush=True)

if __name__=='__main__':
    main()
