"""Config-driven feature additions to a verified nested walk-forward hybrid.

Targets, label endpoints, folds, and benchmark/anchor artifacts are explicit inputs.
Different target horizons require matching anchor/benchmark artifacts, not relabeling
48-hour predictions. No model fitting occurs on import or during preflight.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_features(config):
    return list(dict.fromkeys(config['anchor_features']+list(config['additions'].values())))


def transform(frame, metadata=None):
    if metadata is None:
        metadata = {}
        for f in frame:
            median = float(frame[f].median())
            filled = frame[f].fillna(median)
            metadata[f] = {'median':median, 'center':float(filled.mean()),
                           'scale':float(filled.std(ddof=0)) or 1.0}
    assert list(metadata) == list(frame.columns)
    columns = []
    for f in frame:
        a = metadata[f]
        columns.extend([(frame[f].fillna(a['median'])-a['center'])/a['scale'], frame[f].isna().astype(float)])
    return np.column_stack(columns), metadata


def load(config, labels=False):
    xs, ys = [], []
    for part in config['data_parts']:
        xs.append(pd.read_parquet(ROOT/part['X'], columns=all_features(config)))
        columns = [config['label_end_column']]
        if labels:
            columns.append(config['target_column'])
        ys.append(pd.read_parquet(ROOT/part['y'], columns=columns))
    X, y = pd.concat(xs).sort_index(), pd.concat(ys).sort_index()
    assert X.index.equals(y.index) and X.index.is_unique
    assert X.index.tz is not None and str(X.index.tz) == 'UTC'
    assert (y[config['label_end_column']]-y.index == pd.Timedelta(minutes=config['horizon_minutes'])).all()
    if labels:
        assert np.isfinite(y[config['target_column']]).all()
        assert (y[config['target_column']] >= 0).all()
    return X.replace([np.inf,-np.inf],np.nan), y


def splits(config, y):
    for fold, (start,end) in enumerate(config['validation_windows'],1):
        start,end = pd.Timestamp(start),pd.Timestamp(end)
        inner_start = start-pd.DateOffset(months=config['inner_validation_months'])
        ot = y.index[y[config['label_end_column']] < start]
        ov = y.index[(y.index >= start)&(y.index < end)]
        it = y.index[y[config['label_end_column']] < inner_start]
        iv = ot[ot >= inner_start]
        assert all(len(v) for v in [ot,ov,it,iv])
        assert y.loc[it,config['label_end_column']].max() < iv.min()
        assert y.loc[ot,config['label_end_column']].max() < ov.min()
        assert y.loc[iv,config['label_end_column']].max() < ov.min()
        yield fold,ot,ov,it,iv


def anchors(config, fold):
    profile = ROOT/config['anchor_profile']
    inner = json.loads((profile/f'fold_{fold}_inner.json').read_text())
    outer = json.loads((ROOT/config['outer_pair_profile']/f'fold_{fold}_pair.json').read_text())
    return inner,outer


def pair_prediction(X, card):
    A,_ = transform(X,card['preprocessing'])
    return np.maximum(card['intercept']+A@card['coefficients'],0)


def preflight(config):
    X,y = load(config)
    assert config['horizon_minutes'] == config['anchor_horizon_minutes'] == config['benchmark_horizon_minutes']
    prior = json.loads((ROOT/config['source_hybrid_config']).read_text())
    assert config['forest'] == prior['forest']
    assert config['validation_windows'] == prior['validation_windows']
    assert config['forest_weights'] == prior['forest_weight_candidates']
    assert config['anchor_features'] == prior['features']
    assert json.loads((ROOT/config['anchor_profile']/'verification.json').read_text())['status']=='PASS'
    expected = pd.read_parquet(ROOT/config['benchmark_predictions'],columns=[]).index
    keys, ledger, admission, summaries = [],[],[],[]
    paths = [HERE/'config.json', HERE/'feature_experiment.py', ROOT/config['benchmark_predictions'],
             ROOT/config['source_hybrid_config'], ROOT/config['anchor_profile']/'verification.json']
    paths += [ROOT/p[k] for p in config['data_parts'] for k in ['X','y']]
    for fold,ot,ov,it,iv in splits(config,y):
        keys.extend(ov)
        inner,outer = anchors(config,fold)
        for role,index in [('inner_train',it),('inner_validation',iv),('outer_train',ot),('outer_validation',ov)]:
            frame = y.loc[index].copy()
            frame['fold'],frame['role'] = fold,role
            ledger.append(frame)
            summaries.append({'fold':fold,'role':role,'rows':len(index),'first':str(index.min()),
                'last':str(index.max()),'max_label_end':str(y.loc[index,config['label_end_column']].max())})
        for role,index,card in [('inner',it,inner),('outer',ot,outer)]:
            _,prep = transform(X.loc[index,config['anchor_features']])
            assert prep == card['preprocessing']
            for feature in X:
                values=X.loc[index,feature]
                count,unique=int(values.count()),int(values.nunique())
                admitted=count>=config['minimum_valid_training'] and unique>=2
                admission.append({'fold':fold,'role':role,'feature':feature,'valid':count,'unique':unique,'admitted':admitted})
                assert admitted,f'Inadmissible {feature} in {role} fold {fold}'
        paths += [ROOT/config['anchor_profile']/f'fold_{fold}_inner.json',
                  ROOT/config['outer_pair_profile']/f'fold_{fold}_pair.json',
                  ROOT/config['anchor_profile']/f'fold_{fold}_inner_predictions.parquet']
    assert pd.DatetimeIndex(keys).equals(expected) and len(set(keys))==len(keys)
    assert config['new_fit_budget'] == len(config['validation_windows'])*len(config['additions'])*3
    out=HERE/'preflight';out.mkdir(exist_ok=False)
    pd.concat(ledger).to_csv(out/'split_ledger.csv',index_label='decision_time')
    pd.DataFrame(summaries).to_csv(out/'split_summary.csv',index=False)
    pd.DataFrame(admission).to_csv(out/'admission.csv',index=False)
    receipt={'status':'PASS','horizon_minutes':config['horizon_minutes'],'validation_rows':len(keys),
             'target_values_loaded':False,'new_fits':0,
             'hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths}}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2))
    print(f'Preflight PASS: {len(keys)} identical rows; all three additions admitted inside every inner/outer training fold.',flush=True)


def run(config):
    receipt=json.loads((HERE/'preflight/verification.json').read_text())
    for name,sha in receipt['hashes'].items():
        assert digest(ROOT/name)==sha,f'Changed input: {name}'
    out=HERE/'run_001';out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__},indent=2))
    X,y=load(config,True);target=config['target_column']
    baseline=pd.read_parquet(ROOT/config['benchmark_predictions'])
    np.testing.assert_array_equal(baseline.actual,y.loc[baseline.index,target])
    benchmark=float(np.abs(baseline.actual-baseline[config['benchmark_column']]).mean())
    assert abs(benchmark-config['benchmark_mae'])<1e-10
    ledger=[];parts={name:[] for name in config['additions']}

    def fit(fold,candidate,name,fn):
        assert len(ledger)<config['new_fit_budget']
        row={'fit_number':len(ledger)+1,'fold':fold,'candidate':candidate,'name':name,
             'status':'started','time_utc':pd.Timestamp.now(tz='UTC').isoformat()}
        ledger.append(row);path=out/'fit_ledger.json';path.write_text(json.dumps(ledger,indent=2))
        print(f'Fit {len(ledger)}/{config["new_fit_budget"]}: fold {fold}, {candidate}, {name}',flush=True)
        try:
            answer=fn();row['status']='completed';return answer
        except Exception as exc:
            row.update(status='failed',error=repr(exc));raise
        finally:
            path.write_text(json.dumps(ledger,indent=2))

    for fold,ot,ov,it,iv in splits(config,y):
        inner,outer=anchors(config,fold)
        inner_pair=pair_prediction(X.loc[iv,config['anchor_features']],inner)
        outer_pair=pair_prediction(X.loc[ov,config['anchor_features']],outer)
        np.testing.assert_allclose(outer_pair,baseline.loc[ov,'refitted_pair'],rtol=0,atol=1e-10)
        cached_inner=pd.read_parquet(ROOT/config['anchor_profile']/f'fold_{fold}_inner_predictions.parquet')
        assert cached_inner.index.equals(iv)
        np.testing.assert_array_equal(cached_inner.actual,y.loc[iv,target])
        np.testing.assert_allclose(cached_inner.pair,inner_pair,rtol=0,atol=1e-10)
        for name,extra in config['additions'].items():
            features=config['anchor_features']+[extra]
            A,prep=transform(X.loc[it,features]);B,_=transform(X.loc[iv,features],prep)
            model=fit(fold,name,'inner_forest',lambda:RandomForestRegressor(**config['forest']).fit(A,y.loc[it,target]))
            ip=np.maximum(model.predict(B),0)
            joblib.dump(model,out/f'fold_{fold}_{name}_inner.joblib')
            scores=[{'weight':w,'MAE':float(np.abs(y.loc[iv,target]-((1-w)*inner_pair+w*ip)).mean())}
                    for w in config['forest_weights']]
            weight=fit(fold,name,'weight_selection',lambda:min(scores,key=lambda r:(r['MAE'],r['weight']))['weight'])
            (out/f'fold_{fold}_{name}_inner.json').write_text(json.dumps({'features':features,'preprocessing':prep,
                'scores':scores,'chosen_weight':weight,'training_rows':len(it)},indent=2))
            pd.DataFrame({'actual':y.loc[iv,target],'pair':inner_pair,'forest':ip},index=iv).to_parquet(
                out/f'fold_{fold}_{name}_inner_predictions.parquet')
            A,prep=transform(X.loc[ot,features]);B,_=transform(X.loc[ov,features],prep)
            model=fit(fold,name,'outer_forest',lambda:RandomForestRegressor(**config['forest']).fit(A,y.loc[ot,target]))
            fp=np.maximum(model.predict(B),0)
            joblib.dump(model,out/f'fold_{fold}_{name}_outer.joblib')
            (out/f'fold_{fold}_{name}_outer.json').write_text(json.dumps({'features':features,'preprocessing':prep,
                'weight':weight,'training_rows':len(ot)},indent=2))
            frame=pd.DataFrame({'fold':fold,'actual':y.loc[ov,target],'pair':outer_pair,'forest':fp,
                'prediction':(1-weight)*outer_pair+weight*fp,'weight':weight,'added_feature_missing':X.loc[ov,extra].isna()},index=ov)
            assert np.isfinite(frame[['actual','pair','forest','prediction','weight']]).all().all()
            frame.to_parquet(out/f'fold_{fold}_{name}_predictions.parquet');parts[name].append(frame)
            print(f'Completed fold {fold}: {name}; forest weight {weight:.2f}',flush=True)
    assert len(ledger)==config['new_fit_budget']
    predictions=baseline[['actual','refitted_pair']].copy()
    predictions['incumbent_hybrid']=baseline[config['benchmark_column']]
    weights={};missing=[]
    for name,frames in parts.items():
        frame=pd.concat(frames).sort_index();assert frame.index.equals(baseline.index)
        predictions[name]=frame.prediction
        weights[name]=frame.groupby('fold').weight.first().to_dict()
        for status,subset in frame.groupby('added_feature_missing'):
            missing.append({'candidate':name,'added_feature_missing':bool(status),'rows':len(subset),
                'candidate_MAE':float(np.abs(subset.actual-subset.prediction).mean()),
                'benchmark_MAE':float(np.abs(subset.actual-baseline.loc[subset.index,config['benchmark_column']]).mean())})
    predictions.to_parquet(out/'predictions.parquet')
    losses=predictions.drop(columns='actual').sub(predictions.actual,axis=0).abs()
    overall=losses.mean();monthly=losses.groupby(losses.index.strftime('%Y-%m')).mean()
    monthly.to_csv(out/'monthly_MAE.csv');pd.DataFrame(missing).to_csv(out/'missingness_diagnostics.csv',index=False)
    losses.groupby(baseline.fold).mean().to_csv(out/'fold_MAE.csv')
    intervals=[]
    for name in config['additions']:
        delta=losses[name]-losses.incumbent_hybrid
        for days in [7,14]:
            group=((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int)
            sums,sizes=delta.groupby(group).sum().to_numpy(),delta.groupby(group).size().to_numpy()
            draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)))
            samples=sums[draws].sum(1)/sizes[draws].sum(1);lo,hi=np.quantile(samples,[.025,.975])
            intervals.append({'candidate':name,'block_days':days,'mean_delta':float(delta.mean()),'low':lo,'high':hi})
    winner=min(['incumbent_hybrid']+list(config['additions']),key=lambda name:overall[name])
    report={'status':'COMPLETE_PENDING_REPLAY_VERIFICATION','horizon_minutes':config['horizon_minutes'],'rows':len(predictions),
        'pooled_MAE':overall.to_dict(),'lowest_development_MAE':winner,'selected_forest_weights':weights,
        'months_better_incumbent':{n:int((monthly[n]<monthly.incumbent_hybrid-1e-10).sum()) for n in config['additions']},
        'nominal_paired_intervals':intervals,'new_fits_including_weight_selection':len(ledger),
        'cumulative_fits_including_weight_selection':config['prior_fit_count']+len(ledger),
        'interpretation':'Adaptive development feature comparison. Fixed pair anchors; one added RF feature per candidate. Lower MAE may update the scoreboard only after verification, not establish independent confirmation. Fresh test not accessed.'}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pooled_MAE':report['pooled_MAE'],'best':winner},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','run'])
    args=parser.parse_args();config=json.loads((HERE/'config.json').read_text())
    {'preflight':preflight,'run':run}[args.command](config)
