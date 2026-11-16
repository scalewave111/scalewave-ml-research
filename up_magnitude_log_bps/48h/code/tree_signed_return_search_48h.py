r"""User-run tree-family/feature search for conditional means and signed returns.
python .\Code\tree_signed_return_search_48h.py
No guarantee of beating Ridge. Earlier-only selection; exposed historical evaluation.
"""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import warnings

HERE = Path(__file__).resolve().parent
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
if (HERE / 'tree_model_deps').exists():
    sys.path.insert(0, str(HERE / 'tree_model_deps'))
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as base
import discover_mean_magnitude_features_48h as price

FAMILIES = ['decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting',
            'hist_gradient_boosting', 'xgboost', 'lightgbm', 'catboost']
OPTIONAL = {'xgboost': 'XGBRegressor', 'lightgbm': 'LGBMRegressor', 'catboost': 'CatBoostRegressor'}
REF = base.EXP / 'best_ridge_signed_48h_v1/20261112T090717301790Z'


def make_model(family, config, threads):
    depth, leaf, trees = config['depth'], config['leaf'], config['trees']
    if family == 'decision_tree':
        return DecisionTreeRegressor(criterion='squared_error', max_depth=depth, min_samples_leaf=leaf, random_state=48)
    if family in ['random_forest', 'extra_trees']:
        cls = RandomForestRegressor if family == 'random_forest' else ExtraTreesRegressor
        return cls(n_estimators=trees, criterion='squared_error', max_depth=depth,
                   min_samples_leaf=leaf, max_features=.8, n_jobs=threads, random_state=48)
    if family == 'gradient_boosting':
        return GradientBoostingRegressor(loss='squared_error', n_estimators=trees, learning_rate=.03,
                   max_depth=min(depth, 4), min_samples_leaf=leaf, subsample=.8, random_state=48)
    if family == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(loss='squared_error', max_iter=trees, learning_rate=.03,
                   max_leaf_nodes=2**min(depth, 4)-1, min_samples_leaf=leaf, l2_regularization=10.,
                   early_stopping=False, random_state=48)
    cls = getattr(importlib.import_module(family), OPTIONAL[family])
    if family == 'xgboost':
        return cls(objective='reg:squarederror', n_estimators=trees, max_depth=depth,
                   learning_rate=.03, min_child_weight=leaf, reg_lambda=10., subsample=.8,
                   colsample_bytree=.8, tree_method='hist', n_jobs=threads, random_state=48)
    if family == 'lightgbm':
        return cls(objective='regression_l2', n_estimators=trees, max_depth=depth,
                   num_leaves=2**min(depth, 4)-1, learning_rate=.03, min_child_samples=leaf,
                   reg_lambda=10., colsample_bytree=.8, n_jobs=threads, verbosity=-1, random_state=48)
    return cls(loss_function='RMSE', iterations=trees, depth=depth, learning_rate=.03,
               l2_leaf_reg=10., thread_count=threads, random_seed=48, verbose=False, allow_writing_files=False)


def fit(x, target, keys, columns, family, config, threads):
    a = x.loc[keys, columns]
    med = a.median().fillna(0.)
    filled = a.fillna(med)
    prep = dict(columns=columns, median=med, center=filled.mean(),
                scale=filled.std(ddof=0).where(lambda s: s > 1e-8, 1.).fillna(1.))
    yy = target.loc[keys].to_numpy(float)
    center, scale = float(yy.mean()), max(float(yy.std()), 1.)
    model = make_model(family, config, threads)
    with warnings.catch_warnings(record=True) as captured:
        model.fit(base.transform(x.loc[keys], prep), (yy-center)/scale)
    return dict(model=model, prep=prep, target_center=center, target_scale=scale,
                family=family, config=config, training_rows=len(keys),
                fit_warnings=[str(w.message) for w in captured])


def direction(values):
    return np.where(values > 0, 'UP', np.where(values < 0, 'DOWN', 'FLAT'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials-per-family', type=int, default=4, help='4 or 8: all four feature sets, optionally two capacities per set')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--allow-missing', action='store_true', help='Explicitly skip unavailable optional packages')
    p.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    p.add_argument('--prepared', type=Path, default=price.WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    p.add_argument('--returns', type=Path, default=price.WEEK1 / 'Dataset/canonical_returns_5m.parquet')
    p.add_argument('--reference-run', type=Path, default=REF)
    args = p.parse_args()
    base.check(args.trials_per_family in [4, 8] and args.threads > 0, 'Use 4 or 8 trials and positive threads')
    families, unavailable = [], {}
    for family in dict.fromkeys(args.families):
        try:
            if family in OPTIONAL:
                module = importlib.import_module(family)
                if not hasattr(module, OPTIONAL[family]):
                    raise ImportError(f'{family} installation is unreadable or incomplete')
            families.append(family)
        except ImportError as e:
            unavailable[family] = str(e)
    if unavailable and not args.allow_missing:
        raise SystemExit('Missing tree packages: '+str(unavailable)+'\nInstall with: python -m pip install xgboost lightgbm catboost\nOr explicitly pass --allow-missing.')
    base.check(bool(families), 'No available families')
    refpath = args.reference_run / 'signed_return_predictions.csv'
    done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and base.sha(refpath) == done['prediction_sha256'], 'Reference integrity failed')
    ref = pd.read_csv(refpath, index_col='decision_time')
    ref.index = pd.to_datetime(ref.index, utc=True)
    ref['issue'] = pd.to_datetime(ref.issue, utc=True)
    ref = ref.sort_index()
    base.check(ref.symbol.eq('BTCUSDT').all() and ref.horizon_hours.eq(48).all(), 'Reference scope differs')
    base.check(np.isfinite(ref[['p_up','p_down']]).all().all() and ref.p_up.between(0,1).all()
               and np.allclose(ref.p_up+ref.p_down, 1., rtol=0, atol=1e-7), 'Invalid probabilities')
    x, y, dev, _ = joblib.load(args.prepared)
    base.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing, 'Invalid input indexes')
    base.check(len(ref) == 18481 and ref.index.is_unique and pd.DatetimeIndex(dev).sort_values().equals(ref.index), 'Population differs')
    y = y.copy(); y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    base.check((y.label_end == y.index+pd.Timedelta(hours=48)).all(), 'Invalid horizon')
    signed = pd.Series(10000*np.expm1(y.signed_return_log_bps/10000), index=y.index)
    target = signed.abs()
    base.check(np.isfinite(signed).all() and (signed.loc[ref.index] != 0).all(), 'Invalid outcomes')
    np.testing.assert_allclose(signed.loc[ref.index], ref.actual_signed_return_bps, rtol=0, atol=1e-7)
    print('Preparing causal price features and verifying canonical targets...', flush=True)
    r = price.canonical.load_returns(args.returns)
    price_audit = price.canonical.validate(r, y)
    price.causal_check(r, x.index)
    extra = price.price_features(r, x.index)
    x = x.join(extra).replace([np.inf,-np.inf], np.nan)
    packs = {}
    for side in ['UP','DOWN']:
        packs[side] = dict(anchor=price.ANCHORS[side],
            reduced=[base.COLS[i] for i in ([0,1,3] if side == 'UP' else [0,1,3,4,5,6,7])],
            eight=list(base.COLS), expanded=list(base.COLS)+list(extra.columns))
    recipes = []
    for family in families:
        for i in range(args.trials_per_family):
            recipes.append(dict(id=f'{family}_{i}', family=family, pack=list(packs['UP'])[i%4],
                depth=3 if i < 4 else 6, leaf=128 if i < 4 else 64, trees=128 if i < 4 else 192))
    out = base.EXP / 'tree_signed_return_search_48h_v1' / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    base.dump(out/'CONTRACT.json', dict(task='Tree conditional means with unchanged direction probabilities',
        recipes=recipes, feature_sets=packs, unavailable=unavailable, price_audit=price_audit,
        target='UP/DOWN conditional ordinary-bps endpoint magnitudes; squared-error loss; no target trimming',
        selection='Separate per side/family minimum row-weighted conditional-magnitude MSE across 3 earlier windows; overall selector also earlier-only',
        inference='Both side models predict every origin; p_up*UP_mean-p_down*DOWN_mean; no realized-side routing',
        history='Expanding, strict label_end<fit cutoff; validation cutoffs9/6/3months before outer issue, first2months origins',
        preprocessing='Train-only medians/standardization/feature clipping8sigma/missing flags; affine target normalization',
        direction='Frozen existing v45-style probabilities; not exact reportedv45; not retrained',
        limitation='Previously exposed historical evaluation, configurations influenced by prior research; no guaranteed improvement',
        inner_fits=len(recipes)*24, maximum_final_fits=len(families)*8,
        input_sha256={str(v):base.sha(v) for v in [args.prepared,refpath,args.returns]},
        source_sha256=base.sha(Path(__file__))))
    for source in [Path(__file__),Path(base.__file__),Path(price.__file__),Path(price.canonical.__file__)]:
        (out/source.name).write_bytes(source.read_bytes())
    frames, validations, selections, audits, side_metrics = [], [], [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            base.check(len(te)>0 and (te>=issue).all(), 'Invalid outer fold')
            f = ref.loc[te,['issue','p_up','p_down','actual_signed_return_bps']].copy()
            f['symbol'],f['horizon_hours']='BTCUSDT',48
            f['ridge_signed_bps']=ref.loc[te,'signed_return_forecast_bps']
            f['side_means_signed_bps']=ref.loc[te,'side_mean_baseline_bps']
            f['zero_signed_bps']=0.
            for side in ['UP','DOWN']:
                mask = signed>0 if side=='UP' else signed<0
                scores=[]
                for config in recipes:
                    columns=packs[side][config['pack']]
                    loss,count=0.,0
                    print(f'{issue.date()} {side}: {config["id"]}/{config["pack"]}',flush=True)
                    for offset in [9,6,3]:
                        cut=issue-pd.DateOffset(months=offset)
                        tr=base.train_keys(y,mask,cut,None)
                        va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))
                                   &(y.label_end<cut+pd.DateOffset(months=3))&mask]
                        base.check(len(va)>=100 and y.loc[va,'label_end'].max()<issue,'Validation purge failed')
                        a=fit(x,target,tr,columns,config['family'],config,args.threads)
                        m=base.metric(target.loc[va],base.predict(a,x.loc[va]))
                        loss+=m['mse_bps2']*len(va);count+=len(va)
                        validations.append(dict(issue=issue,side=side,candidate=config['id'],offset=offset,
                            max_training_label_end=y.loc[tr,'label_end'].max(),fit_warnings=a['fit_warnings'],**m))
                    scores.append(dict(config=config,mse=loss/count))
                    pd.DataFrame(validations).to_csv(out/'validation_metrics.csv',index=False)
                best={family:min([s for s in scores if s['config']['family']==family],key=lambda s:(s['mse'],s['config']['id'])) for family in families}
                overall=min(scores,key=lambda s:(s['mse'],s['config']['id']))
                selections.append(dict(issue=issue,side=side,per_family=best,overall=overall))
                base.dump(out/'SELECTIONS.json',selections)
                tr=base.train_keys(y,mask,issue,None)
                for family,selection in best.items():
                    config=selection['config']
                    a=fit(x,target,tr,packs[side][config['pack']],family,config,args.threads)
                    a.update(issue=issue,side=side,max_training_label_end=y.loc[tr,'label_end'].max())
                    pred=base.predict(a,x.loc[te]);f[f'{side}_{family}_mean_bps']=pred
                    path=out/f'{issue:%Y-%m}_{side}_{family}.joblib';joblib.dump(a,path,compress=3)
                    np.testing.assert_allclose(pred,base.predict(joblib.load(path),x.loc[te]),rtol=0,atol=1e-6)
                    audits.append(dict(issue=issue,side=side,family=family,path=path.name,sha256=base.sha(path),
                        max_training_label_end=a['max_training_label_end'],training_rows=len(tr)))
                    keys=te[mask.loc[te]]
                    side_metrics.append(dict(issue=issue,side=side,family=family,**base.metric(target.loc[keys],f.loc[keys,f'{side}_{family}_mean_bps'])))
                f[f'{side}_selected_mean_bps']=f[f'{side}_{overall["config"]["family"]}_mean_bps']
                print(f'DONE {issue.date()} {side}: {overall["config"]["id"]}, validation RMSE={np.sqrt(overall["mse"]):.3f}',flush=True)
            for family in families+['selected']:
                f[f'{family}_signed_bps']=f.p_up*f[f'UP_{family}_mean_bps']-f.p_down*f[f'DOWN_{family}_mean_bps']
            f.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(f)
            base.dump(out/'FIT_AUDIT.json',audits)
    result=pd.concat(frames).sort_index();base.check(result.index.equals(ref.index),'Final origins differ')
    models=families+['selected','ridge','side_means','zero'];metrics=[]; comparisons=[]
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for model in models:
            pred=g[f'{model}_signed_bps'];actual=g.actual_signed_return_bps
            metrics.append(dict(period=period,model=model,direction_accuracy_pct=100*float((direction(pred)==direction(actual)).mean()),
                predicted_up_pct=100*float((pred>0).mean()),**base.metric(actual,pred)))
    for model in models:
        g=result[['actual_signed_return_bps']].copy();g['model']=model
        g['predicted_return_pct']=result[f'{model}_signed_bps']/100;g['actual_return_pct']=g.actual_signed_return_bps/100
        g['predicted_direction']=direction(result[f'{model}_signed_bps']);g['actual_direction']=direction(g.actual_signed_return_bps)
        g['direction_correct']=g.predicted_direction==g.actual_direction
        g['error_bps']=result[f'{model}_signed_bps']-g.actual_signed_return_bps
        comparisons.append(g)
    table=pd.DataFrame(metrics)
    result.to_csv(out/'signed_return_predictions.csv',index_label='decision_time')
    pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    table.to_csv(out/'signed_metrics.csv',index=False)
    pd.DataFrame(side_metrics).to_csv(out/'conditional_mean_metrics_by_fold.csv',index=False)
    summary=('TREE MODELS: CONDITIONAL MEANS -> SIGNED 48H RETURN\n'
        'Selected policies use earlier conditional-mean validation MSE; report every family, no automatic outer winner promotion.\n'
        'Historical exposed evaluation; no performance guarantee. Direction accuracy is SIGN OF FINAL RETURN, not p_up threshold.\n'
        +table[table.period.eq('ALL')].sort_values('rmse_bps').to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    base.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),training_performed=True,
        direction_retrained=False,model_replay_passed=True,unavailable=unavailable,
        prediction_sha256=base.sha(out/'signed_return_predictions.csv'),contract_sha256=base.sha(out/'CONTRACT.json')))
    print(summary);print('Outputs saved:',out)


if __name__=='__main__':
    main()
