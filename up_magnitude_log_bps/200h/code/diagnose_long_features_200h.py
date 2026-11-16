"""Inference-only diagnostics for the completed 200h long-feature experiment."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import discover_long_features_200h as source

base = source.base
DEFAULT_RUN = base.HERE/'experiments/long_features_200h_v1/20261113T123706785745Z'
MODELS = ['candidate','selected','previous','training_mean','training_median']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def no_fit(*args, **kwargs):
    raise RuntimeError('Training is forbidden in this diagnostic runner')


def describe(y, p, median):
    y,p=np.asarray(y,float),np.asarray(p,float)
    gain=np.abs(y-median)-np.abs(y-p)
    return dict(**base.metric(y,p), actual_mean=float(y.mean()), predicted_mean=float(p.mean()),
        actual_median=float(np.median(y)), predicted_median=float(np.median(p)),
        bias=float((p-y).mean()), mae_gain_vs_median=float(gain.mean()),
        fraction_helped=float((gain>1e-9).mean()),fraction_hurt=float((gain < -1e-9).mean()))


def bins(values, low, high):
    v=np.asarray(values,float)
    return np.where(~np.isfinite(v),'missing',np.where(v<=low,'low',np.where(v<=high,'medium','high')))


def correlation(a,b):
    frame=pd.DataFrame({'a':np.asarray(a,float),'b':np.asarray(b,float)}).dropna()
    if len(frame)<3 or frame.a.nunique()<2 or frame.b.nunique()<2:
        return None
    return float(frame.a.corr(frame.b,method='spearman'))


def spacing_count(index):
    # A descriptive greedy count, NOT effective sample size or a significance test.
    count=0
    last=None
    for t in index.sort_values():
        if last is None or t>last+pd.Timedelta(hours=200):
            count+=1
            last=t
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output-dir',type=Path,default=base.HERE/'experiments/long_feature_diagnostics_200h_v1')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    base.check(args.threads>0,'Threads must be positive')
    base.fit=base.search=no_fit
    run=args.run_dir.resolve()
    done,contract=read(run/'COMPLETE.json'),read(run/'CONTRACT.json')
    base.check(done['status']=='COMPLETE' and done['replay_passed'],'Incomplete run')
    base.check(contract['horizon_minutes']==12000,'Wrong horizon')
    base.check(base.sha(run/'predictions.csv')==done['predictions_sha256'],'Prediction hash mismatch')
    for path,digest in contract['input_hashes'].items():
        base.check(base.sha(Path(path))==digest,'Changed input/source: '+path)
    inputs=argparse.Namespace(**{k:Path(contract['arguments'][k]) for k in ['data_dir','y_dir','returns','reference_run']})
    with threadpool_limits(limits=args.threads):
        x,y,dev,old,definitions,excluded,files=source.load(inputs)
        pred=pd.read_csv(run/'predictions.csv')
        pred['decision_time']=pd.to_datetime(pred.decision_time,utc=True).dt.as_unit('ns')
        pred['issue']=pd.to_datetime(pred.issue,utc=True)
        base.check(len(pred)==2*len(dev) and len(dev)==done['development_origins'] and
                   not pred.duplicated(['target','decision_time']).any(),'Coverage mismatch')
        base.check(set(pred.target)==set(base.TARGETS),'Target mismatch')
        base.check(np.isfinite(pred[['actual']+MODELS]).all().all() and (pred[['actual']+MODELS]>=0).all().all(),'Invalid predictions')
        periods=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
        periods.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC')]='2025-01'
        base.check(pred.period.equals(periods),'Period labels changed')
        metrics,training,validation,transfer,relationships,feature_bins,groups,dependence=[],[],[],[],[],[],[],[]
        evidence=[run/n for n in ['COMPLETE.json','CONTRACT.json','predictions.csv','metrics.csv','selections.json']]
        cards=read(run/'selections.json')
        for target in base.TARGETS:
            f=pred[pred.target.eq(target)].set_index('decision_time').sort_index()
            prior=old[old.target.eq(target)].set_index('decision_time').sort_index()
            base.check(f.index.equals(dev),'Timestamp mismatch')
            np.testing.assert_allclose(f.actual,y.loc[dev,target],rtol=0,atol=1e-10)
            np.testing.assert_allclose(f.previous,prior.selected,rtol=0,atol=1e-8)
            base.check(f.issue.equals(prior.issue),'Issue mismatch')
            for period,part in [('ALL',f)]+list(f.groupby('period')):
                dependence.append(dict(target=target,period=period,rows=len(part),
                    greedy_origins_more_than_200h_apart=spacing_count(part.index),
                    median_origin_spacing_hours=float(part.index.to_series().diff().dt.total_seconds().median()/3600)))
                for name in MODELS:
                    metrics.append(dict(target=target,period=period,model=name,**describe(part.actual,part[name],part.training_median.to_numpy())))
            for issue,part in f.groupby('issue'):
                folder=run/f'{issue:%Y-%m}_{target}'
                path=folder/'model.joblib'
                base.check(base.sha(path)==read(folder/'MODEL_HASH.json')['sha256'],'Model hash mismatch')
                bundle=joblib.load(path)
                choice,policy=read(folder/'selection.json'),read(folder/'policy.json')
                card=[c for c in cards if pd.Timestamp(c['issue'])==issue and c['target']==target]
                base.check(len(card)==1 and card[0]['search']==choice and card[0]['policy']==policy,'Selection records disagree')
                base.check(bundle['features']==choice['features'] and bundle['policy']==policy and bundle['issue']==issue and bundle['target']==target,'Bundle mismatch')
                tr=np.flatnonzero(x.index+pd.Timedelta(hours=200)<issue)
                truth=y[target].to_numpy(float)
                mean,median=float(truth[tr].mean()),float(np.median(truth[tr]))
                np.testing.assert_allclose(part.training_mean,mean,rtol=0,atol=1e-8)
                np.testing.assert_allclose(part.training_median,median,rtol=0,atol=1e-8)
                np.testing.assert_allclose(bundle['constant'],mean if policy['fallback']=='mean' else median,rtol=0,atol=1e-8)
                folds=base.windows(x.index,issue)
                base.check(source.baseline_gate(choice,truth,folds)==policy,'Validation policy failed reproduction')
                for population,keys in [('training_in_sample',x.index[tr]),('later_evaluation',part.index)]:
                    raw=base.predict(bundle['model'],x.loc[keys,bundle['features']].to_numpy(float))
                    selected=raw if policy['accepted'] else np.full(len(keys),bundle['constant'])
                    if population=='later_evaluation':
                        np.testing.assert_allclose(raw,part.candidate,rtol=0,atol=1e-7)
                        np.testing.assert_allclose(selected,part.selected,rtol=0,atol=1e-7)
                    for name,p in [('candidate',raw),('selected',selected),('training_mean',np.full(len(keys),mean)),('training_median',np.full(len(keys),median))]:
                        training.append(dict(target=target,issue=str(issue),population=population,model=name,
                            **describe(y.loc[keys,target],p,median)))
                inner=np.array(choice['validation_window_mae'])
                deployed=inner if policy['accepted'] else np.array(policy[policy['fallback']+'_window_mae'])
                med=np.array(policy['median_window_mae'])
                for j,(itr,va) in enumerate(folds):
                    validation.append(dict(target=target,issue=str(issue),window=j+1,rows=len(va),
                        candidate_mae=inner[j],selected_mae=deployed[j],median_mae=med[j],mean_mae=policy['mean_window_mae'][j],
                        training_label_end_max=x.index[itr].max()+pd.Timedelta(hours=200),validation_start=x.index[va].min(),
                        validation_label_end_max=x.index[va].max()+pd.Timedelta(hours=200)))
                transfer.append(dict(target=target,issue=str(issue),accepted=policy['accepted'],model=choice['model'],
                    feature_count=len(choice['features']),inner_gain_vs_median=float(np.average(med-deployed,weights=[len(v) for _,v in folds])),
                    later_gain_vs_median=describe(part.actual,part.selected,median)['mae_gain_vs_median']))
                for col in bundle['features']:
                    train=x.iloc[tr][col]
                    later=x.loc[part.index,col]
                    q=train.quantile([1/3,2/3]).to_numpy()
                    train_ic=correlation(train,truth[tr]);later_ic=correlation(later,part.actual)
                    relationships.append(dict(target=target,issue=str(issue),feature=col,used_in_selected=policy['accepted'],
                        train_ic=train_ic,later_ic=later_ic,sign_reversed=bool(train_ic*later_ic<0) if train_ic is not None and later_ic is not None else None,
                        train_median=train.median(),later_median=later.median(),train_missing=train.isna().mean(),later_missing=later.isna().mean(),
                        training_tercile_low=q[0],training_tercile_high=q[1]))
                    for population,values,actual in [('training_in_sample',train,truth[tr]),('later_evaluation',later,part.actual.to_numpy())]:
                        labels=bins(values,*q)
                        for label in np.unique(labels):
                            a=np.asarray(actual)[labels==label]
                            feature_bins.append(dict(target=target,issue=str(issue),feature=col,population=population,group=label,
                                rows=len(a),actual_mean=float(a.mean()),actual_median=float(np.median(a))))
                # Past-volatility groups are usable at origin; actual-size groups are hindsight only.
                for dimension,values,threshold in [
                    ('past_200h_volatility',x.loc[part.index,'LONGPRICE__rv__200h'],x.iloc[tr]['LONGPRICE__rv__200h'].quantile([1/3,2/3]).to_numpy()),
                    ('future_actual_size_HINDSIGHT',part.actual,np.quantile(truth[tr],[.5,.9]))]:
                    labels=bins(values,*threshold)
                    for label in np.unique(labels):
                        sub=part.iloc[np.flatnonzero(labels==label)]
                        for name in MODELS:
                            groups.append(dict(target=target,issue=str(issue),dimension=dimension,group=label,
                                training_cut_low=threshold[0],training_cut_high=threshold[1],model=name,
                                **describe(sub.actual,sub[name],median)))
                evidence += [path,folder/'MODEL_HASH.json',folder/'selection.json',folder/'policy.json']
        result=pd.DataFrame(metrics)
        keys=['target','period','model']
        saved=pd.read_csv(run/'metrics.csv').set_index(keys).sort_index()
        actual=result.set_index(keys).sort_index()
        base.check(actual.index.equals(saved.index),'Metric keys differ')
        np.testing.assert_allclose(actual[['rows','mae','rmse','ic']],saved[['rows','mae','rmse','ic']],rtol=1e-9,atol=1e-7,equal_nan=True)
        out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        out.mkdir(parents=True,exist_ok=False)
        for name,data in [('period_metrics',result),('training_vs_evaluation',training),('validation_windows',validation),
                          ('validation_transfer',transfer),('feature_relationships',relationships),('feature_terciles',feature_bins),
                          ('group_metrics',groups),('overlap_counts',dependence)]:
            pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
        summary='\n'.join(['200h long-feature diagnostics; no training; errors in log bps.',
            'Source/prediction/model hashes checked; new model inference replayed; all saved metrics reproduced.',
            '\nEarlier-to-later validation transfer:',pd.DataFrame(transfer).to_string(index=False),
            '\nTraining versus later:',pd.DataFrame(training).query("model == 'selected'").to_string(index=False),
            '\nSelected-feature relationships:',pd.DataFrame(relationships).to_string(index=False),
            '\nTraining scores are in-sample. Inner candidate scores are stored records, not replayed: inner models were not saved.',
            'Feature correlations/bins are descriptive, not causal importance. Reversal can be noise; selection used training history.',
            'Actual-size groups use future outcomes for diagnosis only. Tercile thresholds use mature training data.',
            'Spacing counts are not effective sample size; overlap does not by itself prove the cause of failure.',
            'Historically exposed adaptive evaluation. No automatic retraining, feature selection or model promotion.'])
        (out/'SUMMARY.txt').write_text(summary+'\n',encoding='utf-8')
        (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        base.dump(out/'AUDIT.json',dict(training=False,run=str(run),metrics_reproduced=True,model_replay=True,
            input_source_hashes={str(q):base.sha(q) for q in set(files+evidence+[Path(__file__)])},
            guidance_hashes={str(q):base.sha(q) for q in [base.HERE.parent/'PROJECT_CONTEXT.md']+
                [Path('C:/Users/Siddh/Desktop/ML/ml_framework')/n for n in ['AGENTS.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]},
            versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=base.sklearn.__version__)))
        base.dump(out/'COMPLETE.json',dict(status='COMPLETE',training=False,replay_passed=True,metrics_reproduced=True,
            development_origins=len(dev),summary_sha256=base.sha(out/'SUMMARY.txt')))
        print(summary)
        print('Saved:',out)


if __name__=='__main__':
    main()
