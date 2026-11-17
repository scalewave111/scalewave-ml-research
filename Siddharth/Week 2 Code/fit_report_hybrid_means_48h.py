r"""Seven-market-input linear/RF hybrid adapted to conditional mean magnitudes.

Run: python .\Code\fit_report_hybrid_means_48h.py
No training on import. This is a documented mean adaptation, not PDF replay.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import tempfile
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as b

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parents[1] / 'Week 1'
DATA = WEEK1 / 'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data'
PREPARED = WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
REF = b.EXP / 'best_ridge_signed_48h_v1/20261112T090717301790Z'
PAIR = [f'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__{w}__value' for w in ['15m', '5m']]
COLS = PAIR + [f'F4Z__flow__flow_activity_percentile__{w}__value' for w in ['1h', '15m', '5m']] + [
    'F4Z__liquidation__venue_fragmentation_percentile__1h__value',
    'F4Z__liquidation__liq_temporal_concentration_percentile__1h__value']
WEIGHTS = [0., .1, .25, .5, .75, 1.]
SEEDS = [48, 148, 248]


def fit_component(x, target, keys, columns, forest, threads):
    a = x.loc[keys, columns]
    median = a.median().fillna(0.)
    filled = a.fillna(median)
    prep = dict(columns=list(columns), median=median, center=filled.mean(),
                scale=filled.std(ddof=0).where(lambda s: s > 1e-8, 1.).fillna(1.))
    z = b.transform(a, prep)
    yy = target.loc[keys].to_numpy(float)
    center, scale = float(yy.mean()), max(float(yy.std()), 1.)
    models = []
    for seed in (SEEDS if forest else [None]):
        model = (RandomForestRegressor(n_estimators=128, criterion='squared_error',
            max_depth=3, min_samples_leaf=128, max_features=1., bootstrap=True,
            max_samples=min(4096, len(keys)), n_jobs=threads, random_state=seed)
            if forest else LinearRegression())
        model.fit(z, (yy-center)/scale)
        models.append(model)
    return dict(models=models, prep=prep, target_center=center, target_scale=scale)


def predict_component(a, x):
    z = b.transform(x, a['prep'])
    p = np.mean([m.predict(z) for m in a['models']], axis=0)*a['target_scale']+a['target_center']
    b.check(np.isfinite(p).all(), 'Nonfinite predictions')
    return np.maximum(0., p)


def predict_bundle(bundle, x):
    """Label-free inference on a dataframe containing the seven named columns."""
    linear = predict_component(bundle['linear'], x)
    forest = predict_component(bundle['forest'], x)
    return (1-bundle['weight'])*linear + bundle['weight']*forest


def choose_weight(truth, linear, forest):
    rows = [dict(weight=w, **b.metric(truth, (1-w)*linear+w*forest)) for w in WEIGHTS]
    return min(rows, key=lambda r: (r['mse_bps2'], r['weight']))['weight'], rows


def direction(values):
    return np.where(values > 0, 'UP', np.where(values < 0, 'DOWN', 'FLAT'))


def load_data():
    _, y, dev, _ = joblib.load(PREPARED)
    y = y.copy(); y.index = pd.DatetimeIndex(y.index).as_unit('ns')
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    b.check(y.index.is_unique and y.index.is_monotonic_increasing and
            (y.label_end == y.index+pd.Timedelta(hours=48)).all(), 'Invalid labels/index')
    frames = []
    with duckdb.connect() as con:
        for filename in ['X_train.parquet', 'X_development.parquet']:
            f = con.execute('SELECT decision_time,'+','.join('"'+c+'"' for c in COLS)+
                ' FROM read_parquet(?)', [str(DATA/filename)]).fetchdf()
            f['decision_time'] = pd.to_datetime(f.decision_time, utc=True)
            frames.append(f.set_index('decision_time'))
    x = pd.concat(frames).sort_index(); x.index = x.index.as_unit('ns')
    b.check(x.index.is_unique and x.index.equals(y.index), 'Exact market/label timestamp alignment failed')
    x = x.replace([np.inf, -np.inf], np.nan)
    path = REF/'signed_return_predictions.csv'
    done = json.loads((REF/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(path)==done['prediction_sha256'], 'Reference integrity failed')
    ref = pd.read_csv(path, index_col='decision_time')
    ref.index = pd.to_datetime(ref.index, utc=True).as_unit('ns')
    ref['issue'] = pd.to_datetime(ref.issue, utc=True); ref = ref.sort_index()
    b.check(ref.index.is_unique and len(ref)==18481 and
        pd.DatetimeIndex(dev).as_unit('ns').sort_values().equals(ref.index), 'Evaluation population differs')
    b.check(ref.symbol.eq('BTCUSDT').all() and ref.horizon_hours.eq(48).all(), 'Reference scope differs')
    b.check(ref.p_up.between(0, 1).all() and np.allclose(ref.p_up+ref.p_down, 1., rtol=0, atol=1e-7), 'Invalid probabilities')
    ref['p_down'] = 1.-ref.p_up
    signed = 10000*np.expm1(y.signed_return_log_bps/10000)
    b.check(np.isfinite(signed).all() and signed.loc[ref.index].ne(0).all(), 'Invalid or flat evaluation labels')
    np.testing.assert_allclose(signed.loc[ref.index], ref.actual_signed_return_bps, rtol=0, atol=1e-7)
    b.check(set(ref.issue)==set(b.ISSUES), 'Unexpected evaluation periods')
    for issue in b.ISSUES:
        means = {}
        for side, mask in [('UP',signed>0), ('DOWN',signed<0)]:
            keys = b.train_keys(y,mask,issue,None)
            means[side] = float(signed.loc[keys].abs().mean())
            for offset in [9,6,3]:
                b.train_keys(y,mask,issue-pd.DateOffset(months=offset),None)
        g = ref[ref.issue.eq(issue)]
        np.testing.assert_allclose(g.p_up*means['UP']-g.p_down*means['DOWN'],
                                   g.side_mean_baseline_bps,rtol=0,atol=1e-6)
    return x, y, signed, ref


def self_test():
    rng = np.random.default_rng(48)
    x = pd.DataFrame(rng.normal(size=(1200, 7)), columns=COLS)
    target = pd.Series(300+20*x[PAIR[0]]+rng.normal(size=len(x)))
    x.iloc[::13, 1] = np.nan
    keys = x.index[:1000]
    bundle = dict(linear=fit_component(x, target, keys, PAIR, False, 1),
                  forest=fit_component(x, target, keys, COLS, True, 1), weight=.5)
    p = predict_bundle(bundle, x.iloc[1000:])
    changed = x.copy(); changed.iloc[1000:] = 1e6
    again = fit_component(changed, target, keys, PAIR, False, 1)
    np.testing.assert_allclose(predict_component(bundle['linear'], x), predict_component(again, x))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'model.joblib'; joblib.dump(bundle, path)
        np.testing.assert_allclose(p, predict_bundle(joblib.load(path), x.iloc[1000:]))
    truth = np.array([10., 20., 30.])
    assert choose_weight(truth, truth, truth+10)[0] == 0.
    assert choose_weight(truth, truth+10, truth)[0] == 1.
    assert np.isfinite(p).all() and (p >= 0).all()
    assert len(bundle['forest']['models']) == 3
    print('Synthetic preprocessing isolation, ensemble, blend selection and saved replay checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-data-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    b.check(args.threads > 0, 'Threads must be positive')
    if args.self_test:
        self_test(); return
    x, y, signed, ref = load_data(); target = signed.abs()
    if args.check_data_only:
        print(f'DATA CHECK PASSED: {len(x)} rows, {len(COLS)} features, {len(ref)} evaluation origins; no fitting.'); return
    out = b.EXP/'report_hybrid_means_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    inputs = [PREPARED, REF/'signed_return_predictions.csv', REF/'COMPLETE.json',
              DATA/'X_train.parquet', DATA/'X_development.parquet',
              HERE.parent/'Report/ScaleWave_Hybrid_Magnitude_Report.pdf']
    b.dump(out/'CONTRACT.json', dict(features=COLS, pair=PAIR, seeds=SEEDS, weights=WEIGHTS,
        model='Separate UP/DOWN OLS two-input means blended with three 128-tree squared-error Random Forests using seven inputs.',
        adaptation='PDF does not specify exact columns, seeds, depth or leaf size. Percentile flow/fragmentation/concentration columns mapped by meaning; depth3 leaf128 bootstrap4096 seeds48/148/248 are explicit new choices, not exact report reproduction.',
        target='Conditional positive ordinary-bps 48h endpoint magnitude; affine target normalization; no target trimming or log-target fitting.',
        selection='Each side: minimum row-weighted MSE across three earlier purged windows. Blend grid includes zero and one; ties favor lower forest weight.',
        history='Expanding training label_end strictly before cutoff; validation first2months at9/6/3months before issue, all validation labels mature before outer issue.',
        preprocessing='Training-only median imputation, mean/std scaling, feature clipping at8sigma, missing flags; nonnegative magnitude projection.',
        direction='Existing v45-style reference probabilities unchanged except complement normalization; not exact historical v45.',
        limitation='Historically exposed evaluation; inherited F4Z availability, no fresh source availability audit. No promised improvement. PDF log-bps MAE is a different target.',
        inputs={str(p):b.sha(p) for p in inputs},
        sources={str(p):b.sha(p) for p in [Path(__file__), Path(b.__file__)]},
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib','duckdb']}))
    for path in [Path(__file__), Path(b.__file__)]:
        (out/path.name).write_bytes(path.read_bytes())
    frames, validations, selections, audits = [], [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            b.check(len(te)>0 and (te>=issue).all(), 'Invalid outer period')
            f = ref.loc[te, ['issue','p_up','p_down','actual_signed_return_bps']].copy()
            f['symbol'], f['horizon_hours'] = 'BTCUSDT', 48
            f['forecast_end'] = te+pd.Timedelta(hours=48)
            f['ridge_signed_bps'] = ref.loc[te, 'signed_return_forecast_bps']
            f['zero_signed_bps'] = 0.
            for side in ['UP','DOWN']:
                print(f'{issue.date()} {side}: fitting validation components', flush=True)
                mask = signed>0 if side=='UP' else signed<0
                truth, linear, forest = [], [], []
                for offset in [9,6,3]:
                    cut = issue-pd.DateOffset(months=offset)
                    tr = b.train_keys(y, mask, cut, None)
                    va = y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&mask]
                    b.check(len(va)>0 and y.loc[va,'label_end'].max()<issue, 'Validation maturity failed')
                    a = fit_component(x,target,tr,PAIR,False,args.threads)
                    c = fit_component(x,target,tr,COLS,True,args.threads)
                    lp, fp = predict_component(a,x.loc[va]), predict_component(c,x.loc[va])
                    truth.extend(target.loc[va]); linear.extend(lp); forest.extend(fp)
                    _, rows = choose_weight(target.loc[va].to_numpy(),lp,fp)
                    validations.extend(dict(issue=issue,side=side,offset=offset,
                        training_rows=len(tr),max_training_label_end=y.loc[tr,'label_end'].max(),**r) for r in rows)
                weight, scores = choose_weight(np.array(truth),np.array(linear),np.array(forest))
                selections.append(dict(issue=issue, side=side, weight=weight, pooled_validation=scores))
                tr = b.train_keys(y,mask,issue,None)
                bundle = dict(linear=fit_component(x,target,tr,PAIR,False,args.threads),
                    forest=fit_component(x,target,tr,COLS,True,args.threads),weight=weight,
                    issue=issue,side=side,training_rows=len(tr),max_training_label_end=y.loc[tr,'label_end'].max())
                f[f'{side}_linear_bps'] = predict_component(bundle['linear'],x.loc[te])
                f[f'{side}_forest_bps'] = predict_component(bundle['forest'],x.loc[te])
                f[f'{side}_hybrid_bps'] = predict_bundle(bundle,x.loc[te])
                f[f'{side}_side_means_bps'] = float(target.loc[tr].mean())
                path = out/f'{issue:%Y-%m}_{side}.joblib'; joblib.dump(bundle,path,compress=3)
                np.testing.assert_allclose(f[f'{side}_hybrid_bps'],predict_bundle(joblib.load(path),x.loc[te]),rtol=0,atol=1e-7)
                audits.append(dict(issue=issue,side=side,weight=weight,path=path.name,sha256=b.sha(path),
                    training_rows=len(tr),max_training_label_end=y.loc[tr,'label_end'].max(),replay_passed=True))
                print(f'{issue.date()} {side}: selected forest weight {weight}',flush=True)
            for model in ['linear','forest','hybrid','side_means']:
                f[f'{model}_signed_bps'] = f.p_up*f[f'UP_{model}_bps']-f.p_down*f[f'DOWN_{model}_bps']
            np.testing.assert_allclose(f.side_means_signed_bps,ref.loc[te,'side_mean_baseline_bps'],rtol=0,atol=1e-6)
            f.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time'); frames.append(f)
            pd.DataFrame(validations).to_csv(out/'validation_metrics.csv',index=False)
            b.dump(out/'SELECTIONS.json',selections); b.dump(out/'FIT_AUDIT.json',audits)
    result = pd.concat(frames).sort_index()
    b.check(result.index.equals(ref.index),'Final timestamp set differs')
    metrics, magnitudes, comparisons = [], [], []
    models = ['linear','forest','hybrid','side_means','ridge','zero']
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for model in models:
            pred,actual = g[f'{model}_signed_bps'],g.actual_signed_return_bps
            metrics.append(dict(period=period,model=model,**b.metric(actual,pred),
                direction_accuracy_pct=100*float((direction(pred)==direction(actual)).mean())))
        for side in ['UP','DOWN']:
            gside = g[g.actual_signed_return_bps>0] if side=='UP' else g[g.actual_signed_return_bps<0]
            for model in ['linear','forest','hybrid','side_means']:
                magnitudes.append(dict(period=period,side=side,model=model,
                    **b.metric(gside.actual_signed_return_bps.abs(),gside[f'{side}_{model}_bps'])))
    for model in models:
        g = result[['actual_signed_return_bps']].copy(); g['model'] = model
        g['predicted_return_bps'] = result[f'{model}_signed_bps']
        g['predicted_return_pct'] = g.predicted_return_bps/100
        g['actual_return_pct'] = g.actual_signed_return_bps/100
        g['predicted_direction'] = direction(g.predicted_return_bps)
        g['actual_direction'] = direction(g.actual_signed_return_bps)
        g['direction_correct'] = g.predicted_direction==g.actual_direction
        g['error_bps'] = g.predicted_return_bps-g.actual_signed_return_bps
        comparisons.append(g)
    table = pd.DataFrame(metrics); side_table = pd.DataFrame(magnitudes)
    result.to_csv(out/'signed_return_predictions.csv',index_label='decision_time')
    pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    table.to_csv(out/'signed_metrics.csv',index=False)
    side_table.to_csv(out/'conditional_mean_metrics.csv',index=False)
    summary = ('Seven-input report hybrid adapted to conditional means. Exposed historical evaluation.\n'
        'SIGNED RETURNS (ordinary bps):\n'+table[table.period.eq('ALL')].to_string(index=False)+
        '\n\nCONDITIONAL MAGNITUDES (ordinary bps):\n'+side_table[side_table.period.eq('ALL')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),replay_passed=True,
        prediction_sha256=b.sha(out/'signed_return_predictions.csv')))
    print(summary+'\n\nSaved to '+str(out),flush=True)


if __name__ == '__main__':
    main()
