"""User-run joint MAE/RMSE refinement; no real fitting with --check-inputs."""
import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_magnitude_models_120h as source
import transformed_dual_120h as models
import fit_dual_objective_120h as previous_runner

b = models.b
REFERENCE=b.HERE/'experiments/dual_objective_120h_v1/20261114T132002835184Z'


def load(args):
    x,truth,dev,old,dictionary,files,anchors=previous_runner.load(args)
    control=source.read_verified(REFERENCE)
    keys=['decision_time','issue','target']
    old=old.drop(columns=['_merge'],errors='ignore').merge(
        control[keys+['actual','selected','candidate']].rename(columns={
            'actual':'dual_actual','selected':'previous_dual_selected','candidate':'previous_dual_candidate'}),
        on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==2*len(dev),'Prior dual forecast alignment')
    np.testing.assert_allclose(old.actual,old.dual_actual,rtol=0,atol=1e-10)
    files += [REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv']
    return x,truth,dev,old,dictionary,files,anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/transformed_dual_120h_v1')
    args = parser.parse_args()
    b.check(args.threads>0,'Invalid threads')
    x,truth,dev,old,dictionary,files,anchors = load(args)
    folds = {issue:source.windows(x.index,issue) for issue in b.ISSUES}
    print(f'Checked {len(x)} rows / {len(dev)} origins per target; all eight existing anchor models replayed.',flush=True)
    if args.check_inputs:return
    out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    dictionary.to_csv(out/'feature_dictionary.csv',index=False)
    x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv',index=False)
    paths=[Path(__file__),Path(models.__file__),Path(previous_runner.__file__),Path(source.__file__),Path(source.models.__file__),
        Path(source.models.neural.__file__),Path(models.market.__file__),Path(source.market_features.__file__),
        Path(source.source.__file__),Path(source.source.feature_module.__file__),
        Path(source.source.prior.__file__),Path(source.source.prior.canonical.__file__),Path(b.__file__)]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_hours=120,targets=b.TARGETS,
        configs=models.CONFIGS,mean_weights=models.MEAN_WEIGHTS,update_weights=models.UPDATE_WEIGHTS,
        losses=['epsilon0_absolute_error_or_HGB_absolute_error','squared_error'], input_transforms=['raw','signed_log','rank','quadratic'], target_transform='positive constant scaling only; no target logarithm',estimator_rows='all mature rows',
        anchors=[dict(target=t,issue=str(i),features=a['features']) for (t,i),a in anchors.items()],
        method='Training-only robust scaling, signed-log/rank/quadratic basis; Ridge mean and epsilon0 SGD median or shallow HGB pair. Earlier four purged quarters only; minimize worst relative MAE/RMSE versus median/mean and anchor. Replacement requires .1% gain in both baseline metrics, no pooled degradation versus anchor in either metric, 3/4 wins each and no >2% window loss. Otherwise retain saved anchor unchanged.',
        limitations=['Repeatedly exposed historical research; no guaranteed baseline win',
            'Anchor feature lists were chosen in previous adaptive searches; not fresh validation',
            'Single blended forecast is assessed against both criteria; candidate is diagnostic if gates fail',
            'Existing source availability/masking limitations inherited; overlapping 120h outcomes'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__),
        input_hashes={str(p):b.sha(p) for p in files},source_hashes={str(p):b.sha(p) for p in paths},
        guidance_hashes={str(p):b.sha(p) for p in source.GUIDANCE}))
    for p in paths:(out/p.name).write_bytes(p.read_bytes())
    frames,selections,ledger,feature_rows=[],[],[],[]
    with threadpool_limits(limits=args.threads):
        for target in b.TARGETS:
            y=truth[target].to_numpy(float)
            control=old[old.target.eq(target)].set_index('decision_time')
            for i,issue in enumerate(b.ISSUES):
                end=b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
                te=x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
                tr=np.flatnonzero(x.index+pd.Timedelta(hours=120)<issue)
                anchor=anchors[(target,issue)]
                folder=out/target/f'{issue:%Y-%m}';folder.mkdir(parents=True)
                print(f'{target} {issue:%Y-%m}: joint MAE/RMSE search',flush=True)
                plan=models.search(x,y,folds[issue],anchor['features'],folder)
                bundle=dict(anchor=anchor['model'],anchor_features=anchor['features'],plan=plan,target=target,issue=issue)
                cache={}
                for name in ['candidate','selected']:
                    spec=plan[name]
                    if spec['model']=='dual':
                        key=(spec['pack'],spec['config'])
                        if key not in cache:
                            cache[key]=models.fit_pair(x.iloc[tr],y[tr],spec['features'],spec['config'])
                        bundle[name]=cache[key]
                        b.dump(folder/f'{name}_final_quality.json',bundle[name]['quality'])
                    used = anchor['features'] if spec['model']=='anchor' else list(dict.fromkeys(anchor['features']+bundle[name]['features']))
                    feature_rows.extend(dict(target=target,issue=str(issue),forecast=name,feature=c) for c in used)
                forecasts={name:models.predict(bundle,x.iloc[te],name) for name in ['selected','candidate']}
                joblib.dump(bundle,folder/'model.joblib');saved=joblib.load(folder/'model.joblib')
                for name,p in forecasts.items():
                    np.testing.assert_allclose(p,models.predict(saved,x.iloc[te],name),rtol=0,atol=1e-10)
                b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
                refs=control.loc[x.index[te]]
                previous=refs.selected.to_numpy() if target.startswith('up_') else refs.previous_long.to_numpy()
                np.testing.assert_allclose(refs.actual,y[te],rtol=0,atol=1e-10)
                np.testing.assert_allclose(refs.training_median,np.median(y[tr]),rtol=0,atol=1e-8)
                if not plan['accepted']:np.testing.assert_allclose(forecasts['selected'],previous,rtol=0,atol=1e-7)
                frames.append(pd.DataFrame(dict(decision_time=x.index[te],target=target,issue=issue,
                    actual=y[te],**forecasts,previous_best=previous,previous_dual_selected=refs.previous_dual_selected.to_numpy(),previous_dual_candidate=refs.previous_dual_candidate.to_numpy(),training_median=np.median(y[tr]),training_mean=np.mean(y[tr]))))
                selections.append(dict(target=target,issue=str(issue),**plan))
                for label,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                    ledger.append(dict(target=target,issue=issue,window=label,training_rows=len(train),evaluation_rows=len(val),
                        training_label_end_max=x.index[train].max()+pd.Timedelta(hours=120),evaluation_start=x.index[val].min()))
    pred=pd.concat(frames,ignore_index=True)
    b.check(len(pred)==2*len(dev) and not pred.duplicated(['decision_time','target']).any(),'Output coverage')
    b.check(np.isfinite(pred[['selected','candidate']]).all().all(),'Nonfinite predictions')
    pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    scores=[];dual=[]
    for target,g in pred.groupby('target'):
        for period,f in [('ALL',g)]+list(g.groupby('period')):
            metrics={name:b.metric(f.actual.to_numpy(),f[name].to_numpy()) for name in
                     ['selected','candidate','previous_best','previous_dual_selected','previous_dual_candidate','training_median','training_mean']}
            for name,v in metrics.items():scores.append(dict(target=target,period=period,model=name,**v))
            for name in ['selected','candidate','previous_best','previous_dual_selected','previous_dual_candidate']:
                mae_win=metrics[name]['mae']<metrics['training_median']['mae']
                rmse_win=metrics[name]['rmse']<metrics['training_mean']['rmse']
                dual.append(dict(target=target,period=period,model=name,beats_median_mae=mae_win,
                    beats_mean_rmse=rmse_win,beats_both=mae_win and rmse_win))
    scores=pd.DataFrame(scores);scores.to_csv(out/'metrics.csv',index=False)
    dual=pd.DataFrame(dual);dual.to_csv(out/'dual_baseline_checks.csv',index=False)
    b.dump(out/'selections.json',selections);pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    pd.DataFrame(feature_rows).merge(dictionary,on='feature',how='left').to_csv(out/'selected_features.csv',index=False)
    summary='120h transformed-input joint MAE/RMSE refinement; exposed historical evaluation.\n'+scores[scores.period.eq('ALL')].to_string(index=False)+'\n'+dual[dual.period.eq('ALL')].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins_per_target=len(dev),replay_passed=True,
        predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary);print('Saved:',out)


if __name__=='__main__':main()
