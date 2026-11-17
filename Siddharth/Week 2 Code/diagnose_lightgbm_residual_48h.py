"""Read-only model inference diagnostics; no fitting or hyperparameter selection."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tune_lightgbm_residual_48h as run
b=run.b
SOURCE=b.EXP/'lightgbm_residual_48h_v1/20261113T050448624486Z'


def main():
    run.lightgbm()
    done=json.loads((SOURCE/'COMPLETE.json').read_text())
    assert done['status']=='COMPLETE' and b.sha(SOURCE/'predictions.csv')==done['prediction_sha256']
    contract=json.loads((SOURCE/'CONTRACT.json').read_text())
    for path,sha in contract['hashes'].items():
        assert b.sha(Path(path))==sha, path
    x,features,y,signed,ref=run.load();target=signed.abs()
    f=pd.read_csv(SOURCE/'predictions.csv',index_col='decision_time')
    f.index=pd.to_datetime(f.index,utc=True);f['issue']=pd.to_datetime(f.issue,utc=True)
    assert f.index.equals(ref.index)
    np.testing.assert_allclose(f.actual_signed_return_bps,signed.loc[f.index],atol=1e-7,rtol=0)
    out=b.EXP/'lightgbm_diagnostics_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    diagnostics=[];importance=[];missing=[]
    audits=json.loads((SOURCE/'FIT_AUDIT.json').read_text())
    for audit in audits:
        path=SOURCE/audit['artifact'];assert b.sha(path)==audit['sha256']
        a=joblib.load(path);issue=pd.Timestamp(audit['issue']);side=audit['side'];mode=audit['mode']
        mask=signed>0 if side=='UP' else signed<0
        tr=b.train_keys(y,mask,issue,None);allte=f.index[f.issue.eq(issue)];te=allte[mask.loc[allte]]
        np.testing.assert_allclose(run.predict(a,x.loc[allte],features.loc[allte]),f.loc[allte,f'{side}_{mode}'],atol=1e-7,rtol=0)
        trainpred=run.predict(a,x.loc[tr],features.loc[tr]);testpred=f.loc[te,f'{side}_{mode}'].to_numpy()
        ridge_train=np.maximum(0,run.ridge_predict(a['ridge'],x.loc[tr]));ridge_test=f.loc[te,f'{side}_ridge'].to_numpy()
        trainm=b.metric(target.loc[tr],trainpred);testm=b.metric(target.loc[te],testpred)
        rt=b.metric(target.loc[tr],ridge_train);re=b.metric(target.loc[te],ridge_test)
        delta=testpred-ridge_test;err=target.loc[te].to_numpy()-ridge_test
        threshold=float(target.loc[tr].quantile(.9));tail=target.loc[te].to_numpy()>=threshold
        excess=(testpred-target.loc[te].to_numpy())**2-(ridge_test-target.loc[te].to_numpy())**2
        trees=a['model'].booster_.dump_model()['tree_info']
        diagnostics.append(dict(issue=issue,side=side,mode=mode,best_iteration=a['best_iteration'],
            fitted_trees=len(trees),average_leaves=float(np.mean([t['num_leaves'] for t in trees])),
            train_rmse=trainm['rmse_bps'],ridge_train_rmse=rt['rmse_bps'],train_rmse_gain=rt['rmse_bps']-trainm['rmse_bps'],
            outer_rmse=testm['rmse_bps'],ridge_outer_rmse=re['rmse_bps'],outer_rmse_gain=re['rmse_bps']-testm['rmse_bps'],
            train_mean=float(target.loc[tr].mean()),outer_mean=float(target.loc[te].mean()),
            outer_prediction_mean=float(testpred.mean()),outer_bias=testm['bias_bps'],
            correction_rms=float(np.sqrt(np.mean(delta**2))),correction_alignment=float(np.mean(delta*err)),
            correction_energy=float(np.mean(delta**2)),excess_mse=float(excess.mean()),
            tail_cutoff=threshold,tail_rows=int(tail.sum()),tail_excess_contribution=float(np.sum(excess[tail])/len(te)),
            nontail_excess_contribution=float(np.sum(excess[~tail])/len(te))))
        imp=a['model'].booster_.feature_importance(importance_type='gain')
        names=list(features.columns)+['missing__'+c for c in features]
        for name,gain in zip(names,imp):
            importance.append(dict(issue=issue,side=side,mode=mode,feature=name,gain=float(gain)))
        for col in features:
            missing.append(dict(issue=issue,side=side,mode=mode,feature=col,
                training_missing=float(features.loc[tr,col].isna().mean()),outer_missing=float(features.loc[allte,col].isna().mean())))
        print('Replayed',path.name,flush=True)
    d=pd.DataFrame(diagnostics);d.to_csv(out/'model_diagnostics.csv',index=False)
    imp=pd.DataFrame(importance);imp.to_csv(out/'feature_gain.csv',index=False)
    pd.DataFrame(missing).to_csv(out/'missingness.csv',index=False)
    selections=json.loads((SOURCE/'SELECTIONS.json').read_text());selection_rows=[]
    for s in selections:
        for mode in ['direct','residual']:
            row=d[(d.issue==pd.Timestamp(s['issue']))&(d.side==s['side'])&(d['mode']==mode)].iloc[0]
            selection_rows.append(dict(issue=s['issue'],side=s['side'],mode=mode,selected=s['selected'],
                earlier_validation_rmse_gain=np.sqrt(s['ridge_validation_mse'])-np.sqrt(s['best'][mode][0]),
                outer_rmse_gain=row.outer_rmse_gain))
    selection=pd.DataFrame(selection_rows);selection.to_csv(out/'validation_transfer.csv',index=False)
    rows=[];counter=[]
    groups=[('ALL',f)]+[(f'Cycle {q}',f[(f.index.year==2024)&(f.index.quarter==q)]) for q in range(1,5)]
    groups+=[('Jan2025',f[f.index.year==2025])]
    for period,g in groups:
        if g.empty:continue
        truth=g.actual_signed_return_bps.to_numpy();ridge=g.ridge_signed_bps.to_numpy();err=truth-ridge
        for mode in ['direct','residual','selected']:
            du=g.p_up.to_numpy()*(g[f'UP_{mode}']-g.UP_ridge).to_numpy()
            dd=-g.p_down.to_numpy()*(g[f'DOWN_{mode}']-g.DOWN_ridge).to_numpy()
            delta=du+dd
            gain=2*np.mean(delta*err)-np.mean(delta**2)
            np.testing.assert_allclose(gain,np.mean((ridge-truth)**2)-np.mean((g[f'{mode}_signed_bps']-truth)**2),atol=1e-7)
            threshold=float(np.quantile(np.abs(truth),.9));tail=np.abs(truth)>=threshold
            excess=(ridge+delta-truth)**2-(ridge-truth)**2
            rows.append(dict(period=period,mode=mode,rmse=b.metric(truth,ridge+delta)['rmse_bps'],
                up_only_rmse=b.metric(truth,ridge+du)['rmse_bps'],down_only_rmse=b.metric(truth,ridge+dd)['rmse_bps'],
                ridge_rmse=b.metric(truth,ridge)['rmse_bps'],correction_energy=float(np.mean(delta**2)),
                twice_alignment=float(2*np.mean(delta*err)),mse_gain=float(gain),
                tail_excess_contribution=float(np.sum(excess[tail])/len(g)),
                nontail_excess_contribution=float(np.sum(excess[~tail])/len(g))))
            for weight in [0,.1,.25,.5,.75,1]:
                counter.append(dict(period=period,mode=mode,weight=weight,**b.metric(truth,ridge+weight*delta)))
    signed_diag=pd.DataFrame(rows);signed_diag.to_csv(out/'signed_correction_diagnostics.csv',index=False)
    pd.DataFrame(counter).to_csv(out/'hindsight_shrinkage_NOT_VALIDATED.csv',index=False)
    v=pd.read_csv(SOURCE/'validation_metrics.csv')
    summary='NO RETRAINING. Saved-model replay and source hashes verified.\n\nFINAL MODEL DIAGNOSTICS\n'+d.to_string(index=False)
    summary+='\n\nVALIDATION TRANSFER\n'+selection.to_string(index=False)
    summary+='\n\nSIGNED CORRECTIONS\n'+signed_diag.to_string(index=False)
    summary+='\n\nINNER BEST ITERATIONS\n'+v.groupby('mode').best_iteration.describe().to_string()
    summary+='\n\nFeature gain is model usage, not causal or out-of-sample importance. Train scores use final refits; not original early-stop validation scores.\nHindsight component swaps and shrinkage are diagnostics only, not deployable selected policies.\n'
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',source=str(SOURCE),no_training=True,models_replayed=len(audits)))
    print(summary);print('Saved to',out)


if __name__=='__main__':main()
