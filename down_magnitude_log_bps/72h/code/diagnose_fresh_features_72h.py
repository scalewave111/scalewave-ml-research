"""Inference-only review of fresh 72h features; never trains a model."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import discover_long_features_72h as source

b=source.b
REFERENCE=b.HERE/'experiments/long_features_72h_v1/20261114T074315743848Z'
MODELS=['new_features','previous','previous_neural','training_median','training_mean']


def no_fit(*args,**kwargs):
    raise RuntimeError('Diagnostics cannot fit models')


def describe(y,p,median):
    y,p=np.asarray(y),np.asarray(p)
    return dict(**b.metric(y,p),bias=float(np.mean(p-y)),
        actual_mean=float(y.mean()),predicted_mean=float(p.mean()),
        actual_median=float(np.median(y)),predicted_median=float(np.median(p)),
        predicted_std=float(p.std()),actual_std=float(y.std()),
        mae_gain_vs_median=float(np.mean(np.abs(y-median)-np.abs(y-p))))


def analyze(run,x,y,dev,old):
    done=json.loads((run/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and done['replay_passed'],'Incomplete reference')
    b.check(b.sha(run/'predictions.csv')==done['predictions_sha256'],'Prediction hash')
    pred=pd.read_csv(run/'predictions.csv')
    pred.decision_time=pd.to_datetime(pred.decision_time,utc=True).dt.as_unit('ns')
    pred.issue=pd.to_datetime(pred.issue,utc=True)
    b.check(len(pred)==2*len(dev) and done['development_origins']==len(dev) and
        not pred.duplicated(['target','decision_time']).any(),'Coverage')
    b.check(set(pred.target)==set(b.TARGETS),'Targets')
    b.check(np.isfinite(pred[['actual']+MODELS]).all().all(),'Nonfinite predictions')
    expected_period=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    expected_period.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC')]='2025-01'
    b.check(pred.period.equals(expected_period),'Period labels')
    metrics,fit_rows,features,transfer=[],[],[],[]
    cards=json.loads((run/'selections.json').read_text())
    for target in b.TARGETS:
        f=pred[pred.target.eq(target)].set_index('decision_time').sort_index()
        control=old[old.target.eq(target)].set_index('decision_time').sort_index()
        b.check(f.index.equals(dev),'Timestamp alignment')
        b.check(f.issue.equals(control.issue),'Issuance alignment')
        np.testing.assert_allclose(f.actual,y.loc[dev,target],rtol=0,atol=1e-10)
        np.testing.assert_allclose(f.previous,control.selected,rtol=0,atol=1e-8)
        np.testing.assert_allclose(f.previous_neural,control.previous_neural,rtol=0,atol=1e-8)
        for period,part in [('ALL',f)]+list(f.groupby('period')):
            for name in MODELS:
                metrics.append(dict(target=target,period=period,model=name,
                    **describe(part.actual,part[name],part.training_median.to_numpy())))
        for issue,part in f.groupby('issue'):
            tr=np.flatnonzero(x.index+pd.Timedelta(hours=72)<issue)
            truth=y[target].to_numpy(float)
            median=float(np.median(truth[tr]))
            np.testing.assert_allclose(part.training_median,median,atol=1e-10,rtol=0)
            np.testing.assert_allclose(part.training_mean,np.mean(truth[tr]),atol=1e-10,rtol=0)
            folder=run/f'{issue:%Y-%m}_{target}'
            digest=json.loads((folder/'MODEL_HASH.json').read_text())['sha256']
            b.check(b.sha(folder/'model.joblib')==digest,'Model hash')
            bundle=joblib.load(folder/'model.joblib')
            matches=[c for c in cards if c['target']==target and pd.Timestamp(c['issue'])==issue]
            b.check(len(matches)==1,'Selection coverage')
            card=matches[0];cols=bundle['features']
            b.check(cols==card['features'] and bundle['target']==target and pd.Timestamp(bundle['issue'])==issue,'Bundle metadata')
            model=bundle['model']
            replay=b.predict(model,x.loc[part.index,cols].to_numpy(float))
            np.testing.assert_allclose(replay,part.new_features,atol=1e-8,rtol=0)
            train_pred=b.predict(model,x.iloc[tr][cols].to_numpy(float))
            fit_rows.append(dict(target=target,issue=issue,split='training_in_sample',**describe(truth[tr],train_pred,median)))
            for period,later in part.groupby('period'):
                fit_rows.append(dict(target=target,issue=issue,period=period,split='later',**describe(later.actual,later.new_features,median)))
                transfer.append(dict(target=target,issue=issue,period=period,
                    stored_inner_gain=card['baseline_mae']-card['validation_mae'],
                    later_gain=float(np.mean(np.abs(later.actual-median)-np.abs(later.actual-later.new_features)))))
                for col in cols:
                    for split,values,actual in [('training',x.iloc[tr][col],truth[tr]),('later',x.loc[later.index,col],later.actual.to_numpy())]:
                        finite=values.dropna()
                        ic=pd.Series(values.to_numpy()).corr(pd.Series(actual),method='spearman') if finite.nunique()>1 else np.nan
                        features.append(dict(target=target,issue=issue,period=period,feature=col,split=split,
                            missing_fraction=float(values.isna().mean()),median=float(finite.median()) if len(finite) else np.nan,
                            p10=float(finite.quantile(.1)) if len(finite) else np.nan,
                            p90=float(finite.quantile(.9)) if len(finite) else np.nan,ic=ic))
    result=pd.DataFrame(metrics)
    saved=pd.read_csv(run/'metrics.csv')
    keys=['target','period','model']
    check=result.merge(saved,on=keys,suffixes=('','_saved'),how='outer',validate='one_to_one',indicator=True)
    b.check(check['_merge'].eq('both').all(),'Metric coverage')
    for name in ['rows','mae','rmse','ic']:
        np.testing.assert_allclose(check[name],check[name+'_saved'],atol=1e-8,rtol=0,equal_nan=True)
    return dict(period_diagnostics=result,training_vs_later=pd.DataFrame(fit_rows),
        feature_shifts=pd.DataFrame(features),validation_transfer=pd.DataFrame(transfer))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/fresh_diagnostics_72h_v1')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    b.check(args.threads>0,'Threads must be positive')
    contract=json.loads((REFERENCE/'CONTRACT.json').read_text())
    b.check(contract['horizon_minutes']==4320,'Wrong horizon')
    for name in ['input_hashes','source_hashes']:
        for path,digest in contract[name].items():
            b.check(b.sha(Path(path))==digest,'Changed reference dependency: '+path)
    with patch.object(b,'fit',no_fit),patch.object(b,'search',no_fit),threadpool_limits(limits=args.threads):
        x,y,dev,old,_,files=source.load(args)
        tables=analyze(REFERENCE,x,y,dev,old)
    out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():
        table.to_csv(out/(name+'.csv'),index=False)
    summary='72h fresh-feature diagnostics; NO TRAINING. Errors in log bps.\n'+tables['period_diagnostics'].query("model=='new_features'").to_string(index=False)
    summary+='\nBias = prediction minus actual. Training metrics are in-sample. Inner errors are stored, not replayed. Feature shifts are descriptive, not causal. Exposed historical evaluation.\n'
    (out/'SUMMARY.txt').write_text(summary)
    evidence=files+[REFERENCE/n for n in ['COMPLETE.json','CONTRACT.json','predictions.csv','metrics.csv','selections.json']]
    evidence+=list(REFERENCE.glob('*/MODEL_HASH.json'))+list(REFERENCE.glob('*/model.joblib'))
    b.dump(out/'AUDIT.json',dict(reference=REFERENCE,no_training=True,replay_passed=True,metrics_reproduced=True,
        input_hashes={str(p):b.sha(p) for p in evidence},source_hashes={str(p):b.sha(p) for p in [Path(__file__),Path(source.__file__),Path(b.__file__)]}))
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',no_training=True,replay_passed=True,metrics_reproduced=True,
        development_origins=len(dev),summary_sha256=b.sha(out/'SUMMARY.txt')))
    print(summary);print('Saved:',out)


if __name__=='__main__':
    main()
