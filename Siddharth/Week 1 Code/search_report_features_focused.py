"""Registered, resumable feature-subset search on the user's 38 unique columns.

No model fitting occurs on import. Full recipes reuse the November 4 engines.
"""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(_name, '1')
import argparse, concurrent.futures, hashlib, itertools, json, subprocess, sys, time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import mlp_feature_discovery as adapter

m = adapter.m
q = adapter.q
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'experiments/report38_subset_search_v2'
MAPPING = ROOT / 'Report/ic_rf_sources/magnitude_feature_mapping.csv'
PAIR = q.b.PAIR
WEIGHTS = [0., .1, .25, .5]
SCREEN_BUDGET = 200
SCREEN_ROWS = 6144
SEEDS = [48, 149, 250]


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def key(cols):
    return hashlib.sha256('\n'.join(sorted(cols)).encode()).hexdigest()[:16]


def metrics(pred, truth):
    return q.b.metric(np.asarray(pred), np.asarray(truth))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    features = list(pd.read_csv(MAPPING).column)
    assert len(features) == len(set(features)) == 38
    contract = dict(
        target='BTCUSDT absolute 48-hour endpoint log return, log basis points',
        input_rows_in_user_tables=40, unique_requested_columns=features,
        optional_original_controls=PAIR,
        search_scope='Requested 38 columns, alone or with either/both original liquidation controls. No new engineered predictors.',
        admission='Per inner training window: at least100 finite values and at least2 unique values; unusable columns are recorded, never treated as predictive. No test-based admission.',
        benchmark='Original median-regression/RF baseline; also report yesterday volatility-ratio winner and MLP.',
        outer_windows=[str(t) for t in q.STARTS],
        split='Same expanding quarterly windows and 18481 development observations. Training label_end < issuance; validation label_end < outer issuance.',
        search='Per model and outer window: all 38 singletons and 38 pair-plus-singletons; all-feature and family packs; top-k; all pairs among best six single additions, with/without controls; then add/drop neighborhoods until 200 unique subsets or convergence.',
        screen=dict(max_unique_subsets=SCREEN_BUDGET, deterministic_evenly_spaced_training_rows=SCREEN_ROWS,
                    rf='24 trees, max_samples 2048; same depth3/absolute_error/leaf128; median regression unchanged',
                    mlp='Same linear-skip 8/4 SiLU architecture; seed48, max16epochs, patience8'),
        finalists='Four best distinct screening subsets plus best singleton and best pair-plus-singleton; evaluate on all inner training rows with full original recipes; lowest validation MAE wins, fewer features/hash break ties.',
        full_rf='Exact saved original median-regression and128-tree RF parameters/preprocessing; validation chooses RF weight in [0,.1,.25,.5].',
        full_mlp='Original frozen median-regression skip alpha.0001 + 8/4 SiLU correction; dropout.05, AdamW.001/decay.01, batch512, max48epochs, patience8, three seeds48/149/250; average seeds.',
        ablation='After selection is frozen, remove each selected feature in turn, refit full recipe, select weight/epochs on preceding validation only. Outer scores are descriptive contributions, never fed back to selection.',
        inference='Nonnegative magnitude predictions; no deployment or trades.',
        limitation='Previously examined history and multiple adaptive comparisons. Historical point-estimate gains are exploratory, not untouched confirmation.',
        source_sha256=q.sha(__file__))
    cp = OUT/'CONTRACT.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, 'Source or contract changed; use a new version directory'
    else:
        dump(cp, contract)
        (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if (OUT/'prepared.joblib').exists():
        return
    x, y, dev, mapped = q.b.old.load()
    assert set(features).issubset(x.columns)
    x = x[PAIR + features].replace([np.inf, -np.inf], np.nan)
    ref = pd.read_csv(q.b.BASE/'report_baseline_predictions.csv', index_col=0, parse_dates=True)
    assert ref.index.equals(dev) and len(dev) == 18481
    np.testing.assert_allclose(ref.truth, y.loc[dev, 'magnitude_log_bps'], atol=1e-8, rtol=0)
    assert (y.label_end == y.index + pd.Timedelta(hours=48)).all()
    np.testing.assert_allclose(y.magnitude_log_bps, y.signed_return_log_bps.abs(), atol=1e-8, rtol=0)
    for issue in q.STARTS:
        q.base_window(x, y, dev, ref, issue)
    joblib.dump((x, y, dev, ref, features), OUT/'prepared.joblib', compress=3)
    sources = [q.b.old.PACKAGE/'data'/n for n in ['X_train.parquet','X_development.parquet','y_train.parquet','y_development.parquet']]
    sources += [MAPPING, q.b.BASE/'report_baseline_predictions.csv']
    framework = Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    instructions = [framework.parent/'AGENTS.md', framework.parent/'PROJECT_CONTEXT.md', HERE/'PROJECT_CONTEXT.md']
    instructions += [framework/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    dump(OUT/'MANIFEST.json', dict(inputs={str(p):q.sha(p) for p in sources},
        instruction_hashes={str(p):q.sha(p) for p in instructions},
        dependencies={str(p):q.sha(p) for p in [Path(m.__file__),Path(adapter.__file__),Path(q.__file__),Path(q.b.__file__)]},
        versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)))
    dump(OUT/'DATA_AUDIT.json', dict(status='PASS',rows=len(x),test_rows=len(dev),requested_columns=38,
        baseline_replayed=True,strict_48h_labels=True,missing_fraction=x.isna().mean().to_dict()))


def fit_pack(method, x, y, keys, cutoff, cols, folder, vk=None, chosen=None, screen=False):
    """Fit only permitted training rows. Validation selects blend or epochs."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder/'bundle.joblib'
    if saved.exists():
        a = joblib.load(saved)
        assert a['cols'] == cols and a['keys'].equals(keys) and a['cutoff'] == cutoff
        assert a['screen'] == screen and a['method'] == method
        return a
    assert y.loc[keys,'label_end'].max() < cutoff
    a = dict(method=method,cols=cols,keys=keys,cutoff=cutoff,screen=screen)
    if method == 'rf':
        prep = q.b.prep(x.loc[keys], cols)
        tx = q.b.design(x.loc[keys], prep)
        target = y.loc[keys,'magnitude_log_bps'].to_numpy()
        models = {}
        for kind in ['pair','rf']:
            template = joblib.load(q.b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib')['model']
            model = clone(template)
            if kind == 'rf':
                model.set_params(n_jobs=1)
                if screen: model.set_params(n_estimators=24,max_samples=2048)
            model.fit(tx, target)
            models[kind] = dict(model=model,prep=prep)
        if vk is not None:
            p = q.b.predict(models['pair'],x.loc[vk]); r = q.b.predict(models['rf'],x.loc[vk])
            truth = y.loc[vk,'magnitude_log_bps'].to_numpy()
            weight = min(WEIGHTS,key=lambda w:(np.mean(abs((1-w)*p+w*r-truth)),w))
        else: weight = chosen['weight']
        a.update(models=models,weight=weight)
    else:
        m.COLS = cols; m.OUT = folder; m.MAX_EPOCHS = 16 if screen else 48
        (folder/'linear').mkdir(exist_ok=True)
        linear = m.linear_fit(x,y,keys,cutoff)
        neural = []
        seeds = [48] if screen else SEEDS
        for i, seed in enumerate(seeds):
            art = m.fit(x,y,linear,seed,folder/f'seed{seed}.pt',vk=vk,
                        epochs=None if vk is not None else chosen['epochs'][i])
            neural.append(art)
        a.update(linear=linear,neural=neural,epochs=[v['epoch'] for v in neural])
    joblib.dump(a,saved,compress=3)
    return a


def predict(a,x,keys):
    if a['method']=='rf':
        p=q.b.predict(a['models']['pair'],x.loc[keys]);r=q.b.predict(a['models']['rf'],x.loc[keys])
        return (1-a['weight'])*p+a['weight']*r
    m.COLS=a['cols']
    return np.mean([m.predict(n,x,keys) for n in a['neural']],axis=0)


def worker(method, ordinal):
    torch.set_num_threads(1)
    x,y,dev,ref,features=joblib.load(OUT/'prepared.joblib')
    issue=q.STARTS[ordinal];inner,ik,vk,ot,te=q.windows(y,dev,issue)
    folder=OUT/method/f'{issue:%Y%m%d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists(): return
    admission=[dict(feature=c,finite=int(x.loc[ik,c].notna().sum()),unique=int(x.loc[ik,c].nunique())) for c in features]
    for row in admission:row['admitted']=row['finite']>=100 and row['unique']>=2
    dump(folder/'ADMISSION.json',admission)
    features=[r['feature'] for r in admission if r['admitted']]
    small=ik[np.unique(np.linspace(0,len(ik)-1,min(SCREEN_ROWS,len(ik)),dtype=int))]
    truth=y.loc[vk,'magnitude_log_bps'].to_numpy()
    trials={}
    rows_path=folder/'screen_trials.json'
    if rows_path.exists(): trials={r['id']:r for r in json.loads(rows_path.read_text())}
    def evaluate(cols,reason):
        cols=sorted(set(cols));ident=key(cols)
        if ident in trials:return trials[ident]
        if not cols or not set(cols)&set(features) or len(trials)>=SCREEN_BUDGET:return None
        dump(folder/'STATUS.json',dict(stage='screen',completed=len(trials),limit=SCREEN_BUDGET,current=cols,reason=reason))
        a=fit_pack(method,x,y,small,inner,cols,folder/'screen'/ident,vk=vk,screen=True)
        row=dict(id=ident,cols=cols,reason=reason,**metrics(predict(a,x,vk),truth))
        trials[ident]=row;dump(rows_path,list(trials.values()))
        print(method,ordinal,'SCREEN',len(trials),reason,round(row['mae'],5),flush=True)
        return row
    def ranked():return sorted(trials.values(),key=lambda r:(r['mae'],len(r['cols']),r['id']))
    for c in features:
        evaluate([c],'singleton');evaluate(PAIR+[c],'pair_plus_singleton')
    evaluate(features,'all_requested');evaluate(PAIR+features,'all_with_controls')
    for family in sorted({c.split('__')[1] for c in features}):
        group=[c for c in features if c.split('__')[1]==family]
        evaluate(group,'family');evaluate(PAIR+group,'family_with_controls')
    singles=sorted(features,key=lambda c:min(trials[key([c])]['mae'],trials[key(PAIR+[c])]['mae']))
    for count in [2,4,8,12,20]:
        evaluate(singles[:count],'top_k');evaluate(PAIR+singles[:count],'top_k_controls')
    for pair in itertools.combinations(singles[:6],2):
        evaluate(list(pair),'top6_pair');evaluate(PAIR+list(pair),'top6_pair_controls')
    # Add/drop neighborhoods start from several good packs; optional controls can be removed.
    visited=set()
    while len(trials)<SCREEN_BUDGET:
        origins=[r for r in ranked() if r['id'] not in visited][:3]
        if not origins:break
        before=len(trials)
        for origin in origins:
            visited.add(origin['id']);cols=origin['cols']
            for c in cols:evaluate([v for v in cols if v!=c],'backward_remove')
            for c in singles[:12]+PAIR:
                if c not in cols:evaluate(cols+[c],'forward_add')
        if len(trials)==before:break
    final_rows=[]
    choices=ranked()[:4]
    for reason in ['singleton','pair_plus_singleton']:
        choices.append(min((r for r in trials.values() if r['reason']==reason),key=lambda r:(r['mae'],r['id'])))
    choices=list({r['id']:r for r in choices}.values())
    dump(folder/'SHORTLIST.json',choices)
    for row in choices:
        ident=row['id'];cols=row['cols']
        dump(folder/'STATUS.json',dict(stage='full_finalist',completed=len(final_rows),total=len(choices),cols=cols))
        a=fit_pack(method,x,y,ik,inner,cols,folder/'full'/ident/'inner',vk=vk)
        final_rows.append(dict(id=ident,cols=cols,**metrics(predict(a,x,vk),truth)))
    selected=min(final_rows,key=lambda r:(r['mae'],len(r['cols']),r['id']))
    # Freeze all decisions before predicting the outer period; no test scores are computed here.
    dump(folder/'FULL_VALIDATION.json',final_rows)
    dump(folder/'SELECTION.json',dict(issue=str(issue),method=method,selected=selected,finalists=final_rows,
        screening_subsets=len(trials),selected_before_outer_predictions=True))
    chosen=fit_pack(method,x,y,ik,inner,selected['cols'],folder/'full'/selected['id']/'inner',vk=vk)
    dump(folder/'STATUS.json',dict(stage='outer_refit',cols=selected['cols']))
    outer=fit_pack(method,x,y,ot,issue,selected['cols'],folder/'full'/selected['id']/'outer',chosen=chosen)
    pred=pd.DataFrame(dict(selected=predict(outer,x,te),truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te)
    valid=pd.DataFrame(dict(selected=predict(chosen,x,vk),truth=truth,issue=str(issue)),index=vk)
    ablations=[]
    ablation_features=list(selected['cols'])
    permutation_rows=[]
    if len(ablation_features)>8:
        base_mae=metrics(predict(chosen,x,vk),truth)['mae']
        block=(vk.asi8-vk.asi8.min())//pd.Timedelta(days=7).value
        blocks=[np.flatnonzero(block==b) for b in np.unique(block)]
        for c in ablation_features:
            effects=[]
            for repeat in range(3):
                order=np.random.default_rng(20261105+repeat).permutation(len(blocks))
                indices=np.concatenate([blocks[b] for b in order])
                changed=x.loc[vk,selected['cols']].copy()
                changed[c]=changed[c].to_numpy()[indices]
                effects.append(metrics(predict(chosen,changed,vk),truth)['mae']-base_mae)
            permutation_rows.append(dict(feature=c,validation_mae_increase=float(np.mean(effects)),repeats=effects))
        ranked=sorted(permutation_rows,key=lambda r:(-r['validation_mae_increase'],r['feature']))
        ablation_features=[r['feature'] for r in ranked[:4]+ranked[-2:]]
        dump(folder/'VALIDATION_PERMUTATION.json',permutation_rows)
    dump(folder/'DIAGNOSTIC_SCOPE.json',dict(selected_features=selected['cols'],fully_refitted_removals=ablation_features,
        policy='All removals when <=8 inputs; validation block permutation of all inputs and top4/bottom2 full removals otherwise',
        validation_only_diagnostic_choice=True,primary_selection_unchanged=True))
    if len(selected['cols'])>1:
        for removed in ablation_features:
            cols=[c for c in selected['cols'] if c!=removed];ident=key(cols)
            dump(folder/'STATUS.json',dict(stage='ablation',completed=len(ablations),total=len(ablation_features),removed=removed))
            ai=fit_pack(method,x,y,ik,inner,cols,folder/'full'/ident/'inner',vk=vk)
            ao=fit_pack(method,x,y,ot,issue,cols,folder/'full'/ident/'outer',chosen=ai)
            name='without_'+ident
            pred[name]=predict(ao,x,te);valid[name]=predict(ai,x,vk)
            ablations.append(dict(removed=removed,id=ident,prediction_column=name,cols=cols,
                validation_mae=metrics(valid[name],truth)['mae']))
    pred.to_csv(folder/'predictions.csv',index_label='decision_time')
    valid.to_csv(folder/'validation.csv',index_label='decision_time')
    dump(folder/'ABLATIONS.json',ablations)
    # Reopen saved checkpoints, verify preprocessing and regenerate every reported prediction.
    for cols,name in [(selected['cols'],'selected')]+[(a['cols'],a['prediction_column']) for a in ablations]:
        ident=key(cols)
        for phase,keys,cut,pk,frame in [('inner',ik,inner,vk,valid),('outer',ot,issue,te,pred)]:
            art=joblib.load(folder/'full'/ident/phase/'bundle.joblib')
            assert art['keys'].equals(keys) and art['cutoff']==cut and art['cols']==cols
            assert y.loc[keys,'label_end'].max()<cut
            if method=='rf':
                expected=q.b.prep(x.loc[keys],cols)
                for model in art['models'].values():
                    for field in ['median','center','scale']:pd.testing.assert_series_equal(expected[field],model['prep'][field])
            else:
                expected=q.prep(x,keys,cols)
                for field in ['median','scale']:pd.testing.assert_series_equal(expected[field],art['linear']['prep'][field])
            np.testing.assert_allclose(predict(art,x,pk),frame[name],atol=1e-9,rtol=0)
    dump(folder/'DONE.json',dict(status='EXECUTED_AND_AUDITED',method=method,issue=str(issue),
        test_rows=len(te),screening_subsets=len(trials),full_finalists=len(choices),ablations=len(ablations),worker_source_sha256=q.sha(__file__),focused_diagnostics=True,
        strict_purges=True,training_only_preprocessing=True,checkpoint_replay=True))
    dump(folder/'STATUS.json',dict(stage='complete'))


def summarize():
    x,y,dev,ref,features=joblib.load(OUT/'prepared.joblib')
    p=pd.DataFrame(dict(truth=ref.truth,baseline=ref.report_hybrid),index=dev)
    old=pd.read_csv(HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True)
    old_mlp=pd.read_csv(HERE/'experiments/mlp_fixed_liquidation_1h_v1/predictions.csv',index_col=0,parse_dates=True)
    assert old.index.equals(dev) and old_mlp.index.equals(dev)
    p['yesterday_best_rf']=old.vol_ratio
    p['yesterday_mlp']=old_mlp.fixed_four_feature_mlp
    selections=[];ablation_rows=[];all_trials=[];window_rows=[]
    for method in ['rf','mlp']:
        pieces=[]
        for ordinal,issue in enumerate(q.STARTS):
            folder=OUT/method/f'{issue:%Y%m%d}'
            assert json.loads((folder/'DONE.json').read_text())['status']=='EXECUTED_AND_AUDITED'
            frame=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
            selected=json.loads((folder/'SELECTION.json').read_text())
            selections.append(selected)
            for r in json.loads((folder/'screen_trials.json').read_text()):all_trials.append(dict(method=method,issue=str(issue),**r))
            pieces.append(frame.selected)
            for a in json.loads((folder/'ABLATIONS.json').read_text()):
                full=metrics(frame.selected,frame.truth)['mae'];reduced=metrics(frame[a['prediction_column']],frame.truth)['mae']
                ablation_rows.append(dict(method=method,issue=str(issue),feature=a['removed'],rows=len(frame),
                    selected_mae=full,without_feature_mae=reduced,mae_increase_when_removed=reduced-full,
                    validation_mae_increase_when_removed=a['validation_mae']-selected['selected']['mae']))
        joined=pd.concat(pieces);assert joined.index.equals(dev)
        p[method]=joined
    names=[c for c in p if c!='truth']
    scores=pd.DataFrame([dict(model=n,**metrics(p[n],p.truth)) for n in names])
    for issue in q.STARTS:
        _,_,_,_,te=q.windows(y,dev,issue)
        for n in names:window_rows.append(dict(issue=str(issue),model=n,**metrics(p.loc[te,n],p.loc[te,'truth'])))
    uncertainty=[]
    for n in ['rf','mlp']:
        for reference in ['baseline','yesterday_best_rf']:
            d=abs(p[n]-p.truth)-abs(p[reference]-p.truth)
            block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=d.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count'])
            ids=np.random.default_rng(20261105).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1)
            lo,hi=np.quantile(boot,[.025,.975])
            uncertainty.append(dict(model=n,reference=reference,mae_difference=float(d.mean()),lower=float(lo),upper=float(hi)))
    p.to_csv(OUT/'predictions.csv',index_label='decision_time')
    scores.to_csv(OUT/'scores.csv',index=False)
    pd.DataFrame(window_rows).to_csv(OUT/'window_scores.csv',index=False)
    ab=pd.DataFrame(ablation_rows);ab.to_csv(OUT/'feature_ablations.csv',index=False)
    pd.DataFrame(all_trials).to_csv(OUT/'screen_trials.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    dump(OUT/'SELECTIONS.json',selections)
    np.testing.assert_allclose(scores.set_index('model').loc['baseline','mae'],181.52599276479125,atol=1e-8)
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert all(q.sha(path)==value for path,value in manifest['inputs'].items())
    replay=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert replay.index.equals(dev)
    for n in names:np.testing.assert_allclose(replay[n],p[n],atol=1e-9,rtol=0)
    audit=dict(status='PASS',test_rows=len(dev),source_inputs_unchanged=True,baseline_reproduced=True,
        all_eight_workers_audited=True,no_test_score_used_for_selection=True,selection_frozen_before_outer_prediction=True,
        screening_subsets=len(all_trials),requested_unique_features=38)
    dump(OUT/'AUDIT.json',audit)
    dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,scores=scores.to_dict('records'),
        uncertainty=uncertainty,limitation='Previously exposed history; adaptive multiple comparisons; not independent confirmation.'))
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False)+'\n\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n',encoding='utf-8')
    print((OUT/'RESULTS.txt').read_text(),flush=True)


def run():
    prepare()
    jobs=[(method,i) for i in range(4) for method in ['rf','mlp']]
    def launch(job):
        method,i=job;log=OUT/f'{method}_{i}.log'
        with log.open('a',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--method',method,'--ordinal',str(i)],
                                  stdout=handle,stderr=subprocess.STDOUT,cwd=HERE)
        if result.returncode:raise RuntimeError(f'{method} window{i} failed; see {log}')
        return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for job in pool.map(launch,jobs):print('FINISHED',job,flush=True)
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker','summarize'])
    parser.add_argument('--method',choices=['rf','mlp']);parser.add_argument('--ordinal',type=int)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command=='worker':worker(args.method,args.ordinal)
        else:globals()[args.command]()
