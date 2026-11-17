"""Read-only UP MLP feature diagnostics; no fitting, tuning, or feature selection.
Save beside fit_mlp_regularization_mae.py in Week 2/Code.
Needs completed mlp_regularization_mae_v1 and original Week 1 prepared data.
Neutralization replaces a standardized feature with zero (its training median).
This measures saved-model sensitivity, NOT performance after retraining without it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits
import fit_mlp_regularization_mae as recipe

NAMES=['liq_activity_15m','liq_activity_5m','liq_activity_1h','vol_ratio_4h_24h',
       'realized_vol_168h','liq_venue_breadth_1h','funding_persistence_5m','basis_level_1h']
GROUPS={name:[i] for i,name in enumerate(NAMES)}
GROUPS.update(liq_activity_all=[0,1,2], liquidation_all=[0,1,2,5], volatility_all=[3,4])


def check(ok,msg):
    if not ok: raise ValueError(msg)


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)
    a=a[valid];b=b[valid]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0: return None
    value=pd.Series(a).corr(pd.Series(b),method='spearman')
    return float(value) if np.isfinite(value) else None


def mean_prediction(models,z):
    return np.mean([recipe.predict(m,z) for m in models],axis=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--source',type=Path)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();workspace=args.workspace.resolve()
    source=args.source or workspace/'Code/experiments/mlp_regularization_mae_v1'
    out=args.out or workspace/'Code/experiments/mlp_feature_stability_v1'
    check(not out.exists(),f'Preserving existing output {out}; use --out with a new folder')
    check((source/'COMPLETE.json').exists(),'Run the regularization experiment first')
    check(json.loads((source/'COMPLETE.json').read_text())['status']=='COMPLETED','Incomplete source run')
    x,y,dev,ref=joblib.load(workspace.parent/'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid input index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected evaluation universe')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected horizon')
    x=x[recipe.COLS].replace([np.inf,-np.inf],np.nan)
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    saved=pd.read_csv(source/'predictions.csv',index_col='decision_time')
    saved.index=pd.to_datetime(saved.index,utc=True)
    check(saved.index.equals(dev.sort_values()),'Saved timestamps mismatch')
    np.testing.assert_allclose(saved.actual_magnitude_bps,target.loc[saved.index],rtol=0,atol=1e-8)
    check(np.array_equal(saved.actual_direction,side.loc[saved.index]),'Direction mismatch')
    issues=pd.to_datetime(saved.issue,utc=True)
    check(set(issues.unique())==set(recipe.STARTS),'Unexpected folds')
    shifts=[];audits=[];frames=[];correlations=[]
    print('Saved UP MLP feature diagnostics only. No training or selection.',flush=True)
    for issue in recipe.STARTS:
        # Match original inference population, order, and 4096-row batching.
        all_keys=saved.index[issues.eq(issue)]
        up_mask=saved.loc[all_keys,'actual_direction'].eq('UP').to_numpy()
        keys=all_keys[up_mask]
        train=y.index[(y.label_end<issue)&side.eq('UP')]
        path=source/f'{issue:%Y-%m}_UP_original.joblib'
        art=joblib.load(path)
        check(pd.Timestamp(art['issue'])==issue and art['outer_train_keys'].equals(train),'Artifact/training population mismatch')
        check([a['seed'] for a in art['outer']]==recipe.SEEDS,'Unexpected seeds')
        check(all(a['setting']=='original' for a in art['outer']),'Wrong saved model setting')
        prep=art['outer_prep'];all_z=recipe.design(x,all_keys,prep)
        z=all_z[torch.from_numpy(up_mask)]
        models=[recipe.restore(a) for a in art['outer']]
        full_all=mean_prediction(models,all_z)
        np.testing.assert_allclose(full_all,saved.loc[all_keys,'UP_original'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(full_all,saved.loc[all_keys,'UP_existing_mlp'],rtol=0,atol=.001)
        full=full_all[up_mask]
        audits.append(dict(issue=str(issue),max_replay_error_bps=float(np.max(np.abs(full_all-saved.loc[all_keys,'UP_original']))),
                           artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        frame=pd.DataFrame({'issue':str(issue),'actual_bps':target.loc[keys],'original':full},index=keys)
        for group,positions in GROUPS.items():
            neutral=all_z.clone();neutral[:,positions]=0.
            frame[group]=mean_prediction(models,neutral)[up_mask]
        frames.append(frame)
        # Compute marginal correlations on observed raw feature values, excluding missing rows.
        # Training is in-sample; these correlations are descriptive, not performance estimates.
        for j,col in enumerate(recipe.COLS):
            a=x.loc[train,col];b=x.loc[keys,col];scale=float(prep['scale'][col]);med=float(prep['median'][col])
            train_ic=corr(a,target.loc[train]);test_ic=corr(b,target.loc[keys])
            shifts.append(dict(issue=str(issue),feature=NAMES[j],column=col,train_rows=len(a),test_rows=len(b),
                train_missing_fraction=float(a.isna().mean()),test_missing_fraction=float(b.isna().mean()),
                train_median=float(a.median()),test_median=float(b.median()),training_scale=scale,
                median_shift_in_training_scale=float((b.median()-med)/scale),
                test_below_train_p01_fraction=float((b.dropna()<a.quantile(.01)).mean()),
                test_above_train_p99_fraction=float((b.dropna()>a.quantile(.99)).mean()),
                test_at_clip_boundary_fraction=float((np.abs(z[:,j].numpy())>=8).mean()),
                train_feature_target_ic=train_ic,test_feature_target_ic=test_ic,
                correlation_sign_flip=(train_ic*test_ic<0) if train_ic is not None and test_ic is not None else None))
        for population,idx in [('train',train),('test',keys)]:
            for i in range(len(recipe.COLS)):
                for j in range(i+1,len(recipe.COLS)):
                    correlations.append(dict(issue=str(issue),population=population,feature_a=NAMES[i],feature_b=NAMES[j],
                                             rank_correlation=corr(x.loc[idx,recipe.COLS[i]],x.loc[idx,recipe.COLS[j]])))
        print(f'Reviewed {issue.date()}: {len(keys)} UP test rows',flush=True)
    pred=pd.concat(frames).sort_index()
    expected=dev.sort_values()[side.loc[dev.sort_values()].eq('UP')]
    check(pred.index.equals(expected) and len(pred)==10191,'Final UP timestamps mismatch')
    rows=[]
    for issue,f in [('POOLED',pred)]+list(pred.groupby('issue')):
        original=recipe.metrics(f.actual_bps,f.original)
        for group in GROUPS:
            altered=recipe.metrics(f.actual_bps,f[group])
            rows.append(dict(issue=issue,neutralized_input=group,rows=len(f),
                original_mae_bps=original['mae_bps'],neutralized_mae_bps=altered['mae_bps'],
                mae_gain_from_neutralizing_bps=original['mae_bps']-altered['mae_bps'],
                original_rank_ic=original['rank_ic'],neutralized_rank_ic=altered['rank_ic'],
                mean_abs_forecast_change_bps=float((f[group]-f.original).abs().mean())))
    sensitivity=pd.DataFrame(rows);shift=pd.DataFrame(shifts)
    consistency=[]
    for group,g in sensitivity[sensitivity.issue.ne('POOLED')].groupby('neutralized_input',sort=False):
        gains=g.mae_gain_from_neutralizing_bps
        consistency.append(dict(neutralized_input=group,folds=len(g),
            mae_improved_folds=int((gains>1e-8).sum()),mae_worsened_folds=int((gains < -1e-8).sum()),
            min_mae_gain_bps=float(gains.min()),max_mae_gain_bps=float(gains.max())))
    summary=('READ-ONLY UP MLP FEATURE STABILITY\n'
        'Positive MAE gain means replacing that input with its training median reduced saved-model error.\n'
        'Negative gain means the original input helped this saved model. No retraining was performed.\n'
        'Neutralization is a sensitivity check, NOT proof a feature should be removed.\n'
        'Correlated inputs may substitute for one another; grouped checks are included.\n'
        'Neutralization can create unusual feature combinations. Marginal correlation is not conditional importance.\n'
        'Feature shifts compare matching UP training/test populations; realized direction is known only afterward.\n'
        'Historical tests are exposed. No feature selection, causal diagnosis, confidence intervals, or significance tests.\n\n'
        +sensitivity[sensitivity.issue.eq('POOLED')].to_string(index=False)
        +'\n\nFOLD CONSISTENCY\n'+pd.DataFrame(consistency).to_string(index=False)
        +'\n\nFEATURE SHIFTS AND MARGINAL CORRELATIONS\n'
        +shift[['issue','feature','median_shift_in_training_scale','train_feature_target_ic','test_feature_target_ic','correlation_sign_flip']].to_string(index=False))
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    sensitivity.to_csv(out/'sensitivity.csv',index=False)
    shift.to_csv(out/'feature_shifts.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'feature_correlations.csv',index=False)
    pd.DataFrame(consistency).to_csv(out/'fold_consistency.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    (out/'replay_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETED',up_rows=len(pred),replay_checks=4)),encoding='utf-8')
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1): main()
