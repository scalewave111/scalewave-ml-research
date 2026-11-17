"""Read-only review of saved MAE-model diagnostics and forecast ranking.
Save in Week 2/Code. Requires pandas, numpy, scipy. No model imports or fitting.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source',type=Path,help='Original Week 1 diagnostics folder')
    parser.add_argument('--out',type=Path,help='New output directory')
    args=parser.parse_args()
    source=args.source or args.workspace.parent/'Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1'
    out=args.out or args.workspace/'Code/experiments/weak_fold_review_v1'
    require(not out.exists(),f'Preserving existing output: {out}; use --out with a new folder.')
    diagnostic_path=source/'fit_diagnostics.csv'
    prediction_path=source/'predictions.csv'
    d=pd.read_csv(diagnostic_path)
    p=pd.read_csv(prediction_path,index_col='decision_time')
    p.index=pd.to_datetime(p.index,utc=True)
    for f in [d,p]:
        f['issue']=pd.to_datetime(f['issue'],utc=True)
    require(p.index.is_unique and p.index.is_monotonic_increasing,'Invalid prediction timestamps')
    require(len(p)==18481,'Unexpected test row count')
    require(p.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected direction counts')
    starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
    require(set(p.issue)==set(starts),'Unexpected folds')
    choices={'UP':('MLP','mlp_up_magnitude_bps','2024-01-01'),
             'DOWN':('Hybrid RF + median regression','hybrid_down_magnitude_bps','2024-07-01')}
    stages=['train_in_sample','validation_selection','validation_calibration_raw','test_raw']
    rows=[]; groups=[]; ranks=[]
    print('Reading saved diagnostics only. No fitting or training.',flush=True)
    for side,(name,column,weak_date) in choices.items():
        for i,issue in enumerate(starts):
            all_fold=p[p.issue.eq(issue)]
            require((all_fold.index>=issue).all(),'Test timestamps precede issue')
            if i+1<len(starts):
                require((all_fold.index<starts[i+1]).all(),'Test timestamps outside fold')
            f=all_fold[all_fold.actual_direction.eq(side)]
            actual=f.actual_magnitude_bps.to_numpy(float)
            pred=f[column].to_numpy(float)
            require(len(f)>0 and np.isfinite(actual).all() and np.isfinite(pred).all(),'Invalid test data')
            weak=issue==pd.Timestamp(weak_date,tz='UTC')
            for stage in stages:
                q=d[(d.issue==issue)&(d.side==side)&(d.model==name)&(d.stage==stage)]
                require(len(q)==1,f'Missing/duplicate diagnostic: {issue} {side} {stage}')
                rec=q.iloc[0].to_dict()
                require(np.isfinite([rec['mae_bps'],rec['mean_actual_bps'],rec['mean_predicted_bps'],rec['mean_bias_bps']]).all(),'Nonfinite diagnostics')
                if stage=='test_raw':
                    np.testing.assert_allclose(rec['mae_bps'],np.abs(pred-actual).mean(),rtol=0,atol=1e-7)
                    np.testing.assert_allclose(rec['mean_actual_bps'],actual.mean(),rtol=0,atol=1e-7)
                    np.testing.assert_allclose(rec['mean_predicted_bps'],pred.mean(),rtol=0,atol=1e-7)
                    require(int(rec['rows'])==len(f),'Diagnostic row count mismatch')
                rows.append(dict(**rec,highlighted_weak_fold=weak))
            ic=pd.Series(pred).corr(pd.Series(actual),method='spearman') if np.ptp(pred)>0 and np.ptp(actual)>0 else np.nan
            met=dict(issue=issue,side=side,model=name,rows=len(f),mae_bps=float(np.abs(pred-actual).mean()),
                     rank_ic=ic,actual_mean_bps=float(actual.mean()),actual_median_bps=float(np.median(actual)),
                     actual_p90_bps=float(np.quantile(actual,.9)),prediction_std_bps=float(np.std(pred)),
                     highlighted_weak_fold=weak)
            for base in ['mean','median']:
                b=f[f'{side.lower()}_train_{base}_baseline_bps'].to_numpy(float)
                require(np.isfinite(b).all() and np.ptp(b)==0,'Invalid baseline')
                met[f'{base}_baseline_bps']=float(b[0])
                met[f'mae_gain_vs_{base}_bps']=float(np.abs(b-actual).mean()-np.abs(pred-actual).mean())
            ranks.append(met)
            # Forecast-only cutpoints from every test timestamp; evaluate the
            # matching direction afterward. Ties stay together; groups may be empty.
            cuts=all_fold[column].quantile([1/3,2/3]).to_numpy()
            labels=np.where(pred<=cuts[0],'low',np.where(pred<=cuts[1],'medium','high'))
            for label in ['low','medium','high']:
                m=labels==label
                if not m.any():
                    continue
                a=actual[m]; pr=pred[m]
                groups.append(dict(issue=issue,side=side,model=name,forecast_group=label,rows=int(m.sum()),
                    mean_prediction_bps=float(pr.mean()),median_prediction_bps=float(np.median(pr)),
                    mean_actual_bps=float(a.mean()),median_actual_bps=float(np.median(a)),
                    mae_bps=float(np.abs(a-pr).mean()),highlighted_weak_fold=weak))
    stage_table=pd.DataFrame(rows)
    test_table=pd.DataFrame(ranks)
    group_table=pd.DataFrame(groups)
    show=['issue','side','stage','rows','mae_bps','mean_actual_bps','mean_predicted_bps','mean_bias_bps']
    summary=('READ-ONLY MAE / RANK IC DIAGNOSTIC REVIEW\n\n'
        'UP raw MLP and DOWN raw hybrid were chosen after historical inspection: exploratory.\n'
        'Training MAE is in-sample; selection-validation MAE participated in tuning.\n'
        'Validation calibration RAW refers to the last validation month before scaling.\n'
        'Validation predictions use inner fits; test predictions use refitted outer models.\n'
        'Stage populations and sample sizes differ. Gaps alone do not establish overfitting.\n'
        'Only test forecasts are saved here: training/validation Rank IC cannot be recovered.\n'
        'Group actual medians are descriptive and relevant to MAE; no calibration is fitted.\n'
        'No confidence intervals, new model selection, or causal diagnosis.\n\n'
        'HIGHLIGHTED STAGES\n'+stage_table[stage_table.highlighted_weak_fold][show].to_string(index=False)
        +'\n\nALL TEST FOLDS\n'+test_table.to_string(index=False)
        +'\n\nHIGHLIGHTED FORECAST GROUPS\n'+group_table[group_table.highlighted_weak_fold].to_string(index=False))
    out.mkdir(parents=True)
    for filename,table in [('stage_comparison.csv',stage_table),('test_fold_metrics.csv',test_table),('forecast_groups.csv',group_table)]:
        table.to_csv(out/filename,index=False)
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETED',new_fits=0,
        test_diagnostics_match_predictions=True,inputs=[str(diagnostic_path),str(prediction_path)],
        limitation='Reads saved forecasts and diagnostics; does not audit original training or feature causality.'),indent=2),encoding='utf-8')
    print(summary); print(f'\nOutputs saved to: {out}')


if __name__=='__main__':
    main()
