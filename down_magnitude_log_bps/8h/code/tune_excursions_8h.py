"""User-run 8h excursion tuning with earlier-only selection and frozen controls."""
import argparse
import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import improve_excursion_features_8h as price

b = price.base
REFERENCE = b.HERE/'experiments/excursion_pipeline_8h_v1/20261113T104458041903Z/price_followup/20261113T110455348677Z'
SETTINGS = [dict(max_iter=120, max_depth=2, max_leaf_nodes=4, min_samples_leaf=128, l2_regularization=10.),
            dict(max_iter=160, max_depth=3, max_leaf_nodes=7, min_samples_leaf=64, l2_regularization=5.),
            dict(max_iter=160, max_depth=4, max_leaf_nodes=12, min_samples_leaf=128, l2_regularization=10.)]


def fit(x, y, spec):
    model = HistGradientBoostingRegressor(loss=spec['loss'], learning_rate=.04,
        early_stopping=False, random_state=48, max_bins=127, **SETTINGS[spec['setting']])
    model.fit(x[spec['features']].to_numpy(float), y)
    return dict(model=model, spec=spec)


def predict(bundle, x):
    p = np.maximum(0., bundle['model'].predict(x[bundle['spec']['features']].to_numpy(float)))
    b.check(np.isfinite(p).all(), 'Nonfinite prediction')
    return p


def errors(y, p):
    e = y-p
    return dict(mae=float(np.abs(e).mean()), mse=float(np.square(e).mean()))


def select(records, control, weights, metric, guard=False):
    score = lambda r, m: float(np.average([v[m] for v in r], weights=weights))
    eligible = []
    for r in records:
        v = r['windows']
        wins = sum(a[metric] < z[metric] for a, z in zip(v, control))
        if score(v, metric) >= score(control, metric)*.999 or wins < 2:
            continue
        if guard and score(v, 'mse') > score(control, 'mse'):
            continue
        eligible.append(r)
    return min(eligible, key=lambda r: (score(r['windows'], metric), r['id'])) if eligible else None


def load(args):
    reference = args.reference_run
    contract = json.loads((reference/'CONTRACT.json').read_text())
    b.check(contract['horizon_minutes'] == 480, 'Reference horizon differs')
    initial = Path(contract['arguments']['reference_run'])
    initial_args = argparse.Namespace(data_dir=b.DATA,
        y_dir=b.CANONICAL/'y/horizon_minutes=480/symbol=BTCUSDT',
        reference_run=initial, returns=Path(contract['arguments']['returns']))
    x, y, dev, _, _, groups, files = price.inputs(initial_args)
    for module in [Path(b.__file__), Path(price.__file__), Path(price.canonical.__file__)]:
        b.check(b.sha(module) == b.sha(reference/module.name), 'Reference helper changed: '+module.name)
    done = json.loads((reference/'COMPLETE.json').read_text())
    b.check(done['status'] == 'COMPLETE' and done['replay_passed'] and done['original_replay_passed'], 'Incomplete reference')
    b.check(b.sha(reference/'predictions.csv') == done['predictions_sha256'], 'Reference hash mismatch')
    old = pd.read_csv(reference/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    b.check(len(old) == 2*len(dev) and not old.duplicated(['decision_time','target']).any(), 'Reference coverage differs')
    cards = {}
    for target in b.TARGETS:
        g = old[old.target.eq(target)].set_index('decision_time').sort_index()
        b.check(g.index.equals(dev), 'Reference timestamps differ')
        np.testing.assert_allclose(g.actual, y.loc[dev,target], atol=1e-10, rtol=0)
        for i, issue in enumerate(b.ISSUES):
            end = b.ISSUES[i+1] if i+1 < len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=1)
            part = g[g.issue.eq(issue)]
            b.check(part.index.equals(dev[(dev>=issue)&(dev<end)]), 'Reference issuance differs')
            folder = reference/f'{issue:%Y-%m}_{target}'
            plan = json.loads((folder/'selection.json').read_text())
            card = plan['variants'][plan['selected_variant']]
            model_path = folder/card['artifact']
            b.check(b.sha(model_path) == card['artifact_sha256'], 'Reference model hash mismatch')
            bundle = joblib.load(model_path)
            b.check(bundle['features'] == card['features'], 'Reference feature order differs')
            np.testing.assert_allclose(b.predict(bundle['model'], x.loc[part.index,card['features']].to_numpy(float)),
                                       part.selected, atol=1e-7, rtol=0)
            cards[(issue,target)] = card
            files += [folder/'selection.json', model_path]
    return x,y,dev,old,cards,groups,files+[reference/'predictions.csv',reference/'CONTRACT.json',reference/'COMPLETE.json']


def self_test():
    rng = np.random.default_rng(48)
    x = pd.DataFrame({'signal':rng.normal(size=2400), 'missing':rng.normal(size=2400)})
    x.loc[::11,'missing'] = np.nan
    y = 40+30*x.signal.to_numpy()**2+rng.normal(size=len(x))
    with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=2):
        for loss in ['absolute_error','squared_error']:
            spec = dict(features=list(x), loss=loss, setting=0)
            model = fit(x.iloc[:1800], y[:1800], spec)
            pred = predict(model,x.iloc[1800:])
            assert errors(y[1800:],pred)['mae'] < errors(y[1800:],np.full(600,np.median(y[:1800])))['mae']
            path = Path(tmp)/'m.joblib'; joblib.dump(model,path)
            np.testing.assert_array_equal(pred,predict(joblib.load(path),x.iloc[1800:]))
            altered = x.copy(); altered.iloc[1800:,:] = 1e9
            again = fit(altered.iloc[:1800],y[:1800],spec)
            np.testing.assert_array_equal(pred,predict(again,x.iloc[1800:]))
    control = [dict(mae=10,mse=200)]*3
    good = dict(id='good',windows=[dict(mae=9,mse=190)]*3)
    tail = dict(id='tail',windows=[dict(mae=8,mse=210)]*3)
    assert select([good,tail],control,[1]*3,'mae',True)['id']=='good'
    assert select([tail],control,[1]*3,'mae',True) is None
    assert select([good],control,[1]*3,'mse')['id']=='good'
    assert select([],control,[1]*3,'mae',True) is None
    index = pd.date_range('2022-04-22','2025-01-22',freq='30min',tz='UTC')
    for issue in b.ISSUES:
        for tr,va in b.windows(index,issue):
            assert (index[tr]+pd.Timedelta(hours=8)<index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=8)<issue).all()
    print('PASS: both losses, nonlinear signal, missing values, future isolation, replay, selection guards/fallback and 8h purges.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,default=REFERENCE)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/excursion_tuning_8h_v1')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    b.check(args.threads>0,'Threads must be positive')
    with threadpool_limits(limits=args.threads):
        x,y,dev,old,cards,groups,files = load(args)
    print(f'Inputs and reference replay verified: {len(x)} rows, {len(dev)} development origins.',flush=True)
    if args.check_inputs:
        return
    out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    print('Output:',out,flush=True)
    files += [Path(__file__),Path(b.__file__),Path(price.__file__),Path(price.canonical.__file__),b.HERE.parent/'PROJECT_CONTEXT.md']
    framework = Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    files += [framework/p for p in ['AGENTS.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=480,settings=SETTINGS,
        losses=['absolute_error','squared_error'],blend_weights=[.5,1.],
        selection='Three earlier quarterly 8h-purged windows. >=0.1% pooled improvement and >=2/3 wins. MAE selector also requires pooled MSE no worse than current; separate MSE selector has no MAE guard. Otherwise unchanged current.',
        limitations=['Exposed historical evaluation; adaptive reuse of earlier validation and previously selected anchors; no independent confirmation.',
                     'Inherited X availability limits; native NaN preprocessing differs from original.',
                     'No guaranteed gain. No automatic resume. Constant baselines reported; fallback is current model.'],
        versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__),
        hashes={str(p):b.sha(p) for p in files}))
    for module in [Path(__file__),Path(b.__file__),Path(price.__file__),Path(price.canonical.__file__)]:
        (out/module.name).write_bytes(module.read_bytes())
    frames,ledger = [],[]
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            folds = b.windows(x.index,issue)
            tr = np.flatnonzero(x.index+pd.Timedelta(hours=8)<issue)
            for target in b.TARGETS:
                folder = out/f'{issue:%Y-%m}_{target}'; folder.mkdir()
                print(f'{issue:%Y-%m} {target}: tuning 18 candidates...',flush=True)
                card = cards[(issue,target)]; truth = y[target].to_numpy()
                ref = old[old.issue.eq(issue)&old.target.eq(target)].set_index('decision_time').sort_index()
                te = x.index.get_indexer(ref.index)
                control, control_pred = [],[]
                for train,va in folds:
                    model = b.fit(x.iloc[train][card['features']].to_numpy(float),truth[train],card['model'])
                    p = b.predict(model,x.iloc[va][card['features']].to_numpy(float))
                    control_pred.append(p); control.append(errors(truth[va],p))
                all_price = [c for cols in groups.values() for c in cols]
                packs = {'current':card['features'], 'current_all_price':list(dict.fromkeys(card['features']+all_price)), 'price_only':all_price}
                records,specs = [],{}
                for pack,features in packs.items():
                    for loss in ['absolute_error','squared_error']:
                        for setting in range(len(SETTINGS)):
                            ident = f'{pack}_{loss}_{setting}'
                            spec = dict(features=features,loss=loss,setting=setting)
                            specs[ident] = spec
                            predictions = [predict(fit(x.iloc[train],truth[train],spec),x.iloc[va]) for train,va in folds]
                            for weight in [.5,1.]:
                                scores = [errors(truth[va],weight*p+(1-weight)*cp) for (_,va),p,cp in zip(folds,predictions,control_pred)]
                                records.append(dict(id=ident+f'_blend{weight}',candidate=ident,weight=weight,windows=scores))
                            b.dump(folder/'validation_trials.json',records)
                weights = [len(va) for _,va in folds]
                choices = {'mae_selected':select(records,control,weights,'mae',True),
                           'rmse_selected':select(records,control,weights,'mse')}
                # Freeze selection before any outer model fitting or evaluation.
                b.dump(folder/'selection.json',dict(choices=choices,control_windows=control,specs=specs))
                f = pd.DataFrame(dict(decision_time=ref.index,issue=issue,target=target,actual=truth[te],
                    current=ref.selected.to_numpy(),training_mean=ref.training_mean.to_numpy(),training_median=ref.training_median.to_numpy()))
                fitted = {}
                for name,choice in choices.items():
                    if choice is None:
                        f[name] = f.current
                        continue
                    ident = choice['candidate']
                    if ident not in fitted:
                        model = fit(x.iloc[tr],truth[tr],specs[ident]); path = folder/(ident+'.joblib')
                        joblib.dump(model,path)
                        fitted[ident] = predict(model,x.iloc[te])
                        np.testing.assert_allclose(fitted[ident],predict(joblib.load(path),x.iloc[te]),rtol=0,atol=1e-10)
                        b.dump(folder/(ident+'_HASH.json'),dict(sha256=b.sha(path)))
                    f[name] = choice['weight']*fitted[ident]+(1-choice['weight'])*f.current
                f.to_csv(folder/'predictions.csv',index=False); frames.append(f)
                for label,train,va in [('outer',tr,te)]+[(str(i),a,z) for i,(a,z) in enumerate(folds)]:
                    ledger.append(dict(issue=issue,target=target,window=label,training_rows=len(train),
                        training_label_end_max=(x.index[train]+pd.Timedelta(hours=8)).max(),evaluation_start=x.index[va].min(),
                        evaluation_label_end_max=(x.index[va]+pd.Timedelta(hours=8)).max(),evaluation_rows=len(va)))
    pred = pd.concat(frames,ignore_index=True)
    b.check(len(pred)==2*len(dev) and not pred.duplicated(['decision_time','target']).any(),'Output coverage failed')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for target,g in pred.groupby('target'):
        for period,f in [('ALL',g)]+list(g.groupby('period')):
            for name in ['current','mae_selected','rmse_selected','training_mean','training_median']:
                rows.append(dict(target=target,period=period,model=name,**b.metric(f.actual.to_numpy(),f[name].to_numpy())))
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    summary=metrics[metrics.period.eq('ALL')].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary+'\nHistorical evaluation; scores in log bps.\n',encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),replay_passed=True,
        reference_replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary); print('Saved:',out)


if __name__=='__main__':
    main()
