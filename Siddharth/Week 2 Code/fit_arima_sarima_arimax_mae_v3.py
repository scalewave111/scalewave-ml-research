r"""Version 3: conditional 48h BTC magnitude experiments; inputs read-only.

Save in Week 2/Code. Example:
  python .\Code\fit_arima_sarima_arimax_mae_v3.py --model arimax

Fixed recipes, NOT auto-ARIMA or MAE hyperparameter search:
  ARIMA (1,0,1); SARIMA (1,0,1)x(1,0,0,48); ARIMAX (1,0,1)+features.
30-minute regular calendar; 48 seasonal steps = one day.
Separate side series retain opposite-side and absent observations as missing.
No target imputation, interpolation, future-side input or time compression.
Parameters fit quarterly on trailing 180 days (configurable); state updates daily.
Only labels ending STRICTLY before UTC day start enter that day's state.
ARIMAX is regression with ARIMA errors, implemented using statsmodels SARIMAX.
Uses Gaussian likelihood parameter fitting; MAE and Spearman IC are evaluation metrics.
Historical exposed comparison, not direction prediction or trading simulation.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import warnings
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

COLS = [
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
    'PRICE__vol_ratio_4h_24h', 'PRICE__realized_vol_168h',
    'F4Z__liquidation__venue_breadth_percentile__1h__value',
    'F4Z__funding__funding_persistence_percentile__5m__value',
    'F4Z__basis__basis_level_rz__1h__value',
]
STARTS = pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'], utc=True)
STEP = pd.Timedelta(minutes=30)
HORIZON = pd.Timedelta(hours=48)
RECIPES = {
    'arima': ((1,0,1), (0,0,0,0)),
    'sarima': ((1,0,1), (1,0,0,48)),
    'arimax': ((1,0,1), (0,0,0,0)),
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def metrics(actual, predicted):
    a = np.asarray(actual, float); p = np.asarray(predicted, float)
    check(a.shape == p.shape and len(a) > 0, 'Scoring shape mismatch')
    check(np.isfinite(a).all() and np.isfinite(p).all(), 'Nonfinite scoring data')
    ic = float(spearmanr(a,p).statistic) if np.ptp(a)>0 and np.ptp(p)>0 else None
    return dict(rows=len(a), mae_bps=float(np.mean(np.abs(a-p))), rank_ic=ic)


def prepare_exog(x, observed_train):
    """Train-only median/IQR and flags; remove constants and exact dependencies."""
    a = x.loc[observed_train, COLS].replace([np.inf,-np.inf],np.nan)
    median = a.median().fillna(0.)
    filled = a.fillna(median)
    scale = filled.quantile(.75)-filled.quantile(.25)
    scale = scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    prep = dict(median=median, scale=scale, retained=list(range(16)))
    z = transform_exog(a, prep)
    names = COLS + [c+'__missing' for c in COLS]
    keep = []; design = np.ones((len(z),1))
    for j in range(z.shape[1]):
        candidate = np.column_stack([design,z[:,j]])
        if np.linalg.matrix_rank(candidate) > design.shape[1]:
            keep.append(j); design = candidate
    check(bool(keep), 'ARIMAX has no usable varying training features')
    prep['retained'] = keep
    prep['retained_names'] = [names[j] for j in keep]
    prep['dropped_names'] = [names[j] for j in range(16) if j not in keep]
    return prep


def transform_exog(x, prep):
    a = x[COLS].replace([np.inf,-np.inf], np.nan)
    z = ((a.fillna(prep['median'])-prep['median'])/prep['scale']).clip(-8,8)
    result = np.column_stack([z.to_numpy(float),a.isna().to_numpy(float)])
    result = np.ascontiguousarray(result[:,prep['retained']])
    check(np.isfinite(result).all(), 'Nonfinite imputed exogenous features')
    return result


def fit_model(values, exog, kind, maxiter):
    order, seasonal = RECIPES[kind]
    model = SARIMAX(np.asarray(values,float), exog=exog, order=order,
                    seasonal_order=seasonal, trend='c',
                    enforce_stationarity=True, enforce_invertibility=True)
    attempts = []; best = None
    # Retry solver initialization only on training likelihood; never use test scores.
    for method in ['lbfgs', 'powell']:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = model.fit(method=method, maxiter=maxiter, disp=False,
                                   cov_type='none', low_memory=True)
                good = bool(result.mle_retvals.get('converged',False))
                good = good and np.isfinite(result.params).all() and np.isfinite(result.llf)
                attempts.append(dict(method=method, converged=good,
                    warnings=[str(w.message) for w in caught]))
                if good:
                    best = result
                    break
            except Exception as exc:
                attempts.append(dict(method=method, converged=False, error=str(exc)))
    if best is None:
        raise RuntimeError('No converged fit. Training solver details: '+json.dumps(attempts))
    return best, attempts


def extend_state(state, values, exog=None):
    # low_memory fits do not expose results.predicted_state for .extend().
    # Carry the final prior mean/covariance retained by the Kalman filter instead.
    fr = state.filter_results
    # A training-varying feature may be constant within a short update block.
    # Parameters are already fitted: keep every column and its coefficient.
    # Disable only the constructor specification check on this filtering clone;
    # initial parameter fitting still uses the normal specification validation.
    values = np.asarray(values, float)
    check(values.ndim == 1 and not np.isinf(values).any(), 'Invalid update outcomes')
    if state.model.k_exog:
        exog = np.asarray(exog, float)
        check(exog.shape == (len(values), state.model.k_exog), 'Update exog shape changed')
        check(np.isfinite(exog).all(), 'Nonfinite update exog')
    else:
        check(exog is None, 'Unexpected update exog')
    model = state.model.clone(values, exog=exog, validate_specification=False)
    model.ssm.initialize_known(fr.predicted_state[:, -1].copy(),
                               fr.predicted_state_cov[:, :, -1].copy())
    return model.filter(state.params, cov_type='none', low_memory=True)


def forecast_fold(x, target, side, issue, test_keys, kind, train_days, maxiter,
                  artifact_path=None, progress=True):
    """Forecast each side without exposing its current/future realized membership.

    y indexed by outcome ORIGIN; y(origin) is known only after origin+48h.
    All opposite-side outcomes are NaN in that side's regular calendar series.
    For day D, filter only origins <= D-48h-30m. Freeze this state for the day.
    ARIMAX regression coefficients multiply each origin's own known X; the
    ARIMA error forecast is calculated with zero future exog. This avoids needing
    any X after a forecast origin, while preserving regression-with-ARIMA-errors.
    """
    check(kind in RECIPES, 'Unknown recipe')
    check(len(test_keys)>0 and test_keys.min()>=issue,'Invalid test keys')
    last_train = issue-HORIZON-STEP
    first_train = max(x.index.min().ceil('30min'),issue-pd.Timedelta(days=train_days))
    grid = pd.date_range(first_train,test_keys.max(),freq='30min')
    check(grid.isin(x.index).any(), 'No feature timestamps on calendar')
    yg = target.reindex(grid).where(side.reindex(grid).eq(side.name))/100.
    train_grid = grid[grid<=last_train]
    train_y = yg.loc[train_grid].to_numpy(float)
    observed = train_grid[np.isfinite(train_y)]
    check(len(observed)>=500, 'Need at least 500 observed direction-specific training outcomes')
    check((observed+HORIZON<issue).all(),'Training label overlap')
    prep = prepare_exog(x,observed) if kind=='arimax' else None
    # Input features at each origin only; training stats are frozen for this fold.
    exog = transform_exog(x.reindex(grid),prep) if prep is not None else None
    ntrain = len(train_grid)
    state, attempts = fit_model(train_y,exog[:ntrain] if exog is not None else None,kind,maxiter)
    k = state.model.k_exog
    # SARIMAX with trend='c' orders constant, then exog coefficients, then ARMA.
    check(state.model.k_trend==1, 'Unexpected trend layout')
    beta = np.asarray(state.params)[1:1+k] if k else None
    info = dict(model=kind, side=side.name, issue=str(issue),
        first_training_origin=str(first_train),last_training_origin=str(train_grid[-1]),
        max_training_label_end=str(observed.max()+HORIZON),
        train_calendar_rows=ntrain,observed_training_rows=len(observed),
        missing_training_rows=int(np.isnan(train_y).sum()),attempts=attempts,
        parameter_names=state.model.param_names,parameters=np.asarray(state.params).tolist(),
        order=RECIPES[kind][0],seasonal_order=RECIPES[kind][1],
        retained_exog=prep['retained_names'] if prep else [],
        dropped_exog=prep['dropped_names'] if prep else [])
    if artifact_path:
        # Save only bounded training arrays/parameters; no giant covariance histories.
        joblib.dump(dict(info=info,prep=prep,train_y=train_y,
            train_exog=exog[:ntrain] if exog is not None else None),artifact_path,compress=3)
    if k:
        # Regression-with-ARIMA-errors: after fitting, beta is frozen.
        # Subtract X*beta and filter the SAME ARIMA error model without exog.
        # This removes every constant-exog clone path, without changing the model.
        original_state = state.filter_results
        residual_params = np.r_[np.asarray(state.params)[:1], np.asarray(state.params)[1+k:]]
        residual_model = SARIMAX(train_y-exog[:ntrain]@beta,
            order=RECIPES[kind][0], seasonal_order=RECIPES[kind][1], trend='c',
            enforce_stationarity=True,enforce_invertibility=True)
        state = residual_model.filter(residual_params, cov_type='none', low_memory=True)
        np.testing.assert_allclose(state.filter_results.predicted_state[:,-1],
            original_state.predicted_state[:,-1],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(state.filter_results.predicted_state_cov[:,:,-1],
            original_state.predicted_state_cov[:,:,-1],rtol=1e-7,atol=1e-8)
        info['residual_state_equivalence'] = 'PASS'
    pos = ntrain-1; predictions = pd.Series(index=test_keys,dtype=float)
    clipped = 0; audits = []
    for count,(day,keys_series) in enumerate(pd.Series(test_keys,index=test_keys).groupby(test_keys.floor('D'))):
        keys = pd.DatetimeIndex(keys_series.to_numpy())
        cutoff = day-HORIZON-STEP
        new_pos = int(grid.searchsorted(cutoff,side='right')-1)
        check(new_pos>=pos,'Nonchronological state update')
        if new_pos>pos:
            sl = slice(pos+1,new_pos+1)
            update_y = yg.iloc[sl].to_numpy(float)
            if k:
                update_y = update_y - exog[sl] @ beta
            state = extend_state(state, update_y)
            pos=new_pos
        check(grid[pos]+HORIZON<day,'Outcome not available at state cutoff')
        ids = grid.get_indexer(keys)
        check((ids>pos).all(),'Invalid forecast step')
        steps=int(ids.max()-pos)
        # Zero regressors isolate the forecast of the ARIMA error plus intercept.
        kw = {}  # ARIMAX filtering now runs on y-X*beta, with no exog in clones.
        means=np.asarray(state.get_forecast(steps=steps,**kw).predicted_mean,float)
        p=means[ids-pos-1]
        if k:
            p=p+exog[ids]@beta
        check(np.isfinite(p).all(),'Nonfinite forecast')
        clipped+=int((p<0).sum()); predictions.loc[keys]=np.maximum(0,p)*100.
        audits.append(dict(day=str(day),latest_filtered_origin=str(grid[pos]),
                           latest_possible_label_end=str(grid[pos]+HORIZON),forecast_rows=len(keys)))
        if progress and (count%30==0):
            print(f'  {kind} {side.name}: forecast day {day.date()}',flush=True)
    check(predictions.notna().all(),'Incomplete forecast fold')
    info.update(clipped_negative_forecasts=clipped,daily_state_audit=audits)
    return predictions,info


def load_inputs(data_path, reference_path):
    x,y,dev,ref=joblib.load(data_path)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,
          'Prepared X/y indexes must match and be unique/chronological')
    check(x.index.tz is not None,'Prepared timestamps need a timezone')
    x=x.copy(); y=y.copy(); x.index=x.index.tz_convert('UTC');y.index=y.index.tz_convert('UTC')
    check((x.index==x.index.floor('30min')).all(),'Expected 30-minute forecast-origin grid')
    y['label_end']=pd.to_datetime(y.label_end,utc=True)
    check((y.label_end==y.index+HORIZON).all(),'Expected 48-hour labels')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed/10000)),index=y.index)
    check(np.isfinite(target).all(),'Nonfinite target')
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    old=pd.read_csv(reference_path,index_col='decision_time')
    old.index=pd.to_datetime(old.index,utc=True);old['issue']=pd.to_datetime(old.issue,utc=True)
    check(old.index.is_unique and old.index.is_monotonic_increasing,'Invalid reference index')
    check(old.index.equals(pd.DatetimeIndex(dev).tz_convert('UTC').sort_values()),'Evaluation timestamps differ')
    check(len(old)==18481 and old.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},
          'Unexpected historical evaluation population')
    check(set(old.issue.unique())==set(STARTS),'Unexpected fold issue dates')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference directions mismatch')
    check(set(COLS).issubset(x.columns),'Missing required input features')
    return x[COLS],y,target,side,old


def build_scores(pred, model_names):
    rows=[]
    for issue,f in [('POOLED',pred)]+list(pred.groupby('issue')):
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)]
            base=metrics(g.actual_magnitude_bps,g[direction+'_baseline_mean'])['mae_bps']
            med=metrics(g.actual_magnitude_bps,g[direction+'_baseline_median'])['mae_bps']
            for name in model_names:
                col=direction+'_'+name
                # Never score incomplete folds/models as if they were the full comparison.
                if col not in g or not np.isfinite(g[col].to_numpy(float)).all():
                    continue
                met=metrics(g.actual_magnitude_bps,g[col])
                rows.append(dict(issue=str(issue),side=direction,model=name,**met,
                    mae_gain_vs_mean_bps=base-met['mae_bps'],
                    mae_gain_vs_median_bps=med-met['mae_bps']))
    return pd.DataFrame(rows)


def optional_references(ws, old):
    extras = {}
    candidates = [
        ('existing_gru', ws/'Code/experiments/gru_conditional_mae_v1/predictions.csv', 'gru'),
        ('existing_three_window', ws/'Code/experiments/original_multiwindow_mae_v1/predictions.csv', 'three_windows'),
    ]
    for name, path, suffix in candidates:
        if not path.is_file():
            print(f'Optional benchmark not found: {path}', flush=True)
            continue
        f = pd.read_csv(path, index_col='decision_time')
        f.index = pd.to_datetime(f.index, utc=True)
        check(f.index.equals(old.index), f'Optional benchmark timestamps differ: {path}')
        check(np.array_equal(f.actual_direction, old.actual_direction), f'Benchmark sides differ: {path}')
        np.testing.assert_allclose(f.actual_magnitude_bps,old.actual_magnitude_bps,rtol=0,atol=1e-8)
        extras[name] = {d: f[d+'_'+suffix] for d in ['UP','DOWN']}
        print(f'Loaded benchmark: {name}', flush=True)
    return extras


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',choices=['arima','sarima','arimax','all'],default='arimax')
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--data',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--train-days',type=int,default=180)
    ap.add_argument('--maxiter',type=int,default=100)
    args=ap.parse_args()
    check(args.train_days>=60 and args.maxiter>=10,'Use train-days >= 60 and maxiter >= 10')
    print('RUNNER VERSION 3: ARIMAX residual-state filtering', flush=True)
    print(f'Script: {Path(__file__).resolve()}', flush=True)
    ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    data=args.data or w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    reference=args.reference or w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    out=args.out or ws/f'Code/experiments/{args.model}_classical_mae_v3'
    check(not out.exists(),f'Output exists: {out}. Use --out with a new directory.')
    check(data.is_file() and reference.is_file(),f'Input missing. Expected {data} and {reference}')
    x,y,target,side,old=load_inputs(data,reference)
    models=list(RECIPES) if args.model=='all' else [args.model]
    extras=optional_references(ws,old)
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    contract=dict(runner_version=3,update_fix='After fitting beta, filter y-X*beta with equivalent exog-free ARIMA states; add current X*beta for forecasts',models=models,recipes=RECIPES,features=COLS,train_days=args.train_days,
        target='48h positive ordinary-bps magnitude, separate realized-direction training',
        estimator='Gaussian maximum likelihood, not MAE training; fixed orders, no hyperparameter selection',
        grid='30-minute UTC; opposite side and absent targets remain NaN, no compressed event time',
        updates='Quarterly parameter fitting; daily state filtering using labels ending strictly before UTC day start',
        arimax='Eight train-scaled features plus missing flags; constants/dependencies removed using training only',
        exog_timing='Current-origin X only; forecast ARIMA errors with zero exog then add current X times coefficients',
        output='Negative magnitude forecasts clipped at zero; no fitted output calibration',
        comparison='Original expanding baselines and raw models plus matched trailing-window baselines',
        limitation='Different fitting objective/history and daily online state protocol; historical exposed comparison',
        python=platform.python_version(),versions={k:importlib.metadata.version(k) for k in
            ['numpy','pandas','scipy','statsmodels','joblib','threadpoolctl']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'CONTRACT.json',contract)
    frames=[];failures=[];audits=[]
    references=['baseline_mean','baseline_median','window_mean','window_median','existing_mlp_raw','existing_hybrid_raw']+list(extras)
    for issue in STARTS:
        te=old.index[old.issue.eq(issue)]
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            for b in ['mean','median']:
                frame[f'{direction}_baseline_{b}']=old.loc[te,f'{direction.lower()}_train_{b}_baseline_bps']
            window=(y.index>=issue-pd.Timedelta(days=args.train_days))&(y.label_end<issue)&side.eq(direction)
            check(window.sum()>=500,'Too few matched-window baseline samples')
            for b in ['mean','median']:
                frame[f'{direction}_window_{b}']=float(getattr(target.loc[window],b)())
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}_raw']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
            for name, values in extras.items():
                frame[f'{direction}_{name}']=values[direction].loc[te]
            for kind in models:
                tag=f'{issue:%Y-%m}_{direction}_{kind}'
                print(f'START {tag}',flush=True)
                try:
                    p,info=forecast_fold(x,target,side.rename(direction),issue,te,kind,args.train_days,
                        args.maxiter,out/f'{tag}_fit.joblib')
                    frame[f'{direction}_{kind}']=p
                    dump(out/f'{tag}_audit.json',info);audits.append(dict(fold=tag,status='DONE'))
                    print(f'DONE {tag}',flush=True)
                except Exception as exc:
                    frame[f'{direction}_{kind}']=np.nan
                    failures.append(dict(fold=tag,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                    dump(out/'FAILURES.json',failures)
                    print(f'FAILED {tag}: {exc}',flush=True)
                gc.collect()
        frames.append(frame)
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
        partial=pd.concat(frames).sort_index()
        partial.to_csv(out/'predictions.csv',index_label='decision_time')
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Final timestamps differ')
    scores=build_scores(pred,models+references);scores.to_csv(out/'scores.csv',index=False)
    summary=('CLASSICAL TIME-SERIES COMPARISON\nMAE lower is better; positive gain means improvement.\n'
             'Rank IC is Spearman correlation, not accuracy. Constant within-fold IC is undefined.\n'
             'Fixed model orders; Gaussian likelihood fitting, no test-based parameter selection.\n'
             '180-day training default; quarterly fits; daily updates with matured 48h targets only.\n'
             'Historical exposed results; no direction classifier or trading simulation.\n'
             'Only complete side/model evaluations appear in POOLED results.\n\n'
             +scores[scores.issue.eq('POOLED')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'RUN_STATUS.json',dict(status='PARTIAL_FAILURE' if failures else 'COMPLETED',
        completed_fits=len(audits),expected_fits=8*len(models),failures=failures))
    print(summary);print(f'Outputs: {out}')
    if failures:
        raise SystemExit('Some fits failed; see FAILURES.json. No fallback forecasts were substituted.')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
