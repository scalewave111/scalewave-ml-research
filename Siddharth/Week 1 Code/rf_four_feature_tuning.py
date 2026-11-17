"""Bounded four-feature RF tuning with three prior purged quarterly folds."""
from pathlib import Path
import json,time,shutil,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import rf_fixed_liquidation_1h as original
s=original.s;q=original.q
HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/rf_four_feature_tuning_v1'
COLS=original.COLS
WEIGHTS=[0.,.1,.25,.5,.75,1.]
CONFIGS={
 'original':{},
 'shallow':dict(max_depth=2,min_samples_leaf=256),
 'deeper_small_leaf':dict(max_depth=5,min_samples_leaf=64),
 'deeper_subsample':dict(max_depth=5,min_samples_leaf=128,max_features=.75),
 'small_leaf':dict(max_depth=3,min_samples_leaf=64),
 'feature_subsample':dict(max_depth=3,min_samples_leaf=128,max_features=.75),
 'more_trees':dict(n_estimators=256,max_depth=4,min_samples_leaf=128,max_features=.75),
 'smaller_bootstrap':dict(max_depth=3,min_samples_leaf=128,max_samples=2048)}

def choose(rows):
    control=min([r for r in rows if r['config']=='original' and r['weight'] in s.WEIGHTS],key=lambda r:(r['mae'],r['weight']))
    eligible=[r for r in rows if r['gain_vs_control']>=.1 and r['fold_wins']>=2]
    # Simpler trees and lower RF weight break numerical ties; no test information.
    selected=min(eligible,key=lambda r:(r['mae'],r['complexity'],r['weight'],r['config'])) if eligible else control
    return control,selected

def run():
    begin=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    contract=dict(features=COLS,configs=CONFIGS,weights=WEIGHTS,seed=48,
        target='BTCUSDT48h absolute endpoint log-return magnitude, log bps; primary MAE, report Spearman and Pearson IC.',
        folds='Three preceding nonoverlapping validation quarters per outer issuance, models trained expanding with label_end strictly before each fold start; validation labels strictly mature before each fold end. Same four outer windows and18481 test rows.',
        fixed='Median-regression model, training-only original median/mean/std preprocessing plus missingness flags, RF absolute-error criterion/bootstrap, seed48. Only declared RF hyperparameters and blend weight vary.',
        selection='Control is original RF parameters with original weight grid selected on the same pooled three-fold validation. Choose lowest pooled validation MAE among settings improving control by>=.1bps and winning in>=2/3 folds by>.01bps. Else control. Freeze before outer fits/scoring. No choice using test results.',
        reports='Tuned blend primary; selected RF alone diagnostic; original-parameter nested-validation control separates tuning from changed selection history; historical untuned four-feature and best three-feature/original baseline references.',
        reuse='Identical cutoff/config fits cached; original four-feature median/RF fits reused only after exact keys/features/preprocessing/parameter checks.',
        limitation='Feature and historical test already examined; exploratory tuning, not independent confirmation. No test-driven retries or search expansion.',
        source_sha=q.sha(__file__),helper_sha=q.sha(s.__file__),original_source_sha=q.sha(original.__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    cached,y,dev,ref=joblib.load(s.OUT/'prepared.joblib');raw,ry,rd,_=q.b.old.load()
    assert raw.index.equals(cached.index) and rd.equals(dev);pd.testing.assert_series_equal(ry.magnitude_log_bps,y.magnitude_log_bps)
    x=cached[COLS[:-1]].copy();x[COLS[-1]]=raw[COLS[-1]];del raw,ry,cached
    old=pd.read_csv(original.OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert old.index.equals(dev) and len(dev)==18481
    np.testing.assert_allclose(old.baseline,ref.report_hybrid,atol=1e-8,rtol=0);np.testing.assert_allclose(old.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    templates={k:joblib.load(original.OUT/'model_cache'/f'20240101_{k}.joblib')['model'] for k in ['pair','rf']}
    assert templates['rf'].get_params()['random_state']==48
    prototypes={name:clone(templates['rf']).set_params(**cfg) for name,cfg in CONFIGS.items()}
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework');docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    paths=[original.OUT/'predictions.csv',original.OUT/'AUDIT.json',s.OUT/'prepared.joblib']+list((original.OUT/'model_cache').glob('*.joblib'))+[q.b.old.PACKAGE/'data'/n for n in ['X_train.parquet','X_development.parquet','y_train.parquet','y_development.parquet']]
    hashes={str(p):q.sha(p) for p in paths}
    if not (OUT/'MANIFEST.json').exists():q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']},full_rf_parameters={n:p.get_params() for n,p in prototypes.items()}))
    ledger=[];artifacts={};prediction_cache={}
    def get(cut,name):
        key=(str(cut),name)
        if key in artifacts:return artifacts[key]
        keys=y.index[y.label_end<cut];kind='pair' if name=='pair' else 'rf';prototype=templates['pair'] if name=='pair' else prototypes[name]
        path=OUT/'models'/f'{cut:%Y%m%d}_{name}.joblib';reused=False;new=False
        if not path.exists() and name in ['pair','original']:
            source=original.OUT/'model_cache'/f'{cut:%Y%m%d}_{kind}.joblib'
            if source.exists():shutil.copyfile(source,path);reused=True
        if not path.exists():new=True
        q.dump(OUT/'STATUS.json',dict(stage='fit_or_reuse',cutoff=cut,config=name,completed_unique_artifacts=len(artifacts)))
        a=s.fit(x,y,keys,cut,COLS,kind,path,prototype)
        assert a['train_keys'].equals(keys) and a['issuance']==cut and y.loc[keys,'label_end'].max()<cut and a['prep']['columns']==COLS
        expected=s.prepare(x,keys,COLS)
        for field in ['median','center','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
        for k,value in prototype.get_params().items():
            if k!='n_jobs':assert a['model'].get_params()[k]==value
        artifacts[key]=a;ledger.append(dict(cutoff=str(cut),config=name,path=str(path),new_fit_this_invocation=new,reused_historical=reused,rows=len(keys),sha256=q.sha(path)))
        q.dump(OUT/'FIT_LEDGER.json',ledger)
        print('MODEL',len(artifacts),cut.date(),name,'new' if new else 'reused',flush=True)
        return a
    def fold(cut,name):
        key=(str(cut),name)
        if key not in prediction_cache:
            end=cut+pd.DateOffset(months=3);keys=y.index[(y.index>=cut)&(y.index<end)&(y.label_end<end)]
            assert len(keys)>0 and y.loc[keys,'label_end'].max()<end
            prediction_cache[key]=pd.Series(q.b.predict(get(cut,name),x.loc[keys]),index=keys)
        return prediction_cache[key]
    def trials_for(issue):
        cuts=[issue-pd.DateOffset(months=k) for k in [9,6,3]]
        pp=pd.concat([fold(c,'pair') for c in cuts]);truth=y.loc[pp.index,'magnitude_log_bps'].to_numpy()
        assert pp.index.is_unique and y.loc[pp.index,'label_end'].max()<issue
        rf={name:pd.concat([fold(c,name) for c in cuts]).to_numpy() for name in CONFIGS}
        membership=np.concatenate([np.repeat(j,len(fold(c,'pair'))) for j,c in enumerate(cuts)])
        rows=[]
        for name in CONFIGS:
            for weight in WEIGHTS:
                pred=(1-weight)*pp.to_numpy()+weight*rf[name]
                fm=[float(abs(pred[membership==j]-truth[membership==j]).mean()) for j in range(3)]
                rows.append(dict(issue=str(issue),config=name,weight=weight,complexity=prototypes[name].max_depth,fold_maes=fm,**q.b.metric(pred,truth)))
        control=min([r for r in rows if r['config']=='original' and r['weight'] in s.WEIGHTS],key=lambda r:(r['mae'],r['weight']))
        for r in rows:
            gains=np.asarray(control['fold_maes'])-np.asarray(r['fold_maes']);r.update(gain_vs_control=control['mae']-r['mae'],fold_wins=int(sum(gains>.01)),fold_gains=gains.tolist())
        ctrl,selected=choose(rows)
        return rows,ctrl,selected,pp,rf,truth
    pieces=[];vals=[];decisions=[];alltrials=[]
    for issue in q.STARTS:
        rows,control,selected,pp,rf,truth=trials_for(issue);alltrials.extend(rows)
        decision=dict(issue=str(issue),control=control,selected=selected);decisions.append(decision)
        q.dump(OUT/f'{issue:%Y%m%d}_selection.json',decision);q.dump(OUT/'TRIALS.json',alltrials)
        print('SELECTED',issue.date(),selected['config'],'RF weight',selected['weight'],'validation MAE',selected['mae'],flush=True)
        # Primary decisions frozen before generating any outer-window predictions.
        _,_,_,ot,te=q.windows(y,dev,issue)
        pair=q.b.predict(get(issue,'pair'),x.loc[te]);selected_rf=q.b.predict(get(issue,selected['config']),x.loc[te]);control_rf=q.b.predict(get(issue,'original'),x.loc[te])
        p=old.loc[te,['baseline','vol_ratio','rf_plus_liq1h','truth','issue']].rename(columns={'rf_plus_liq1h':'untuned_4features'}).copy()
        p['tuned_blend']=(1-selected['weight'])*pair+selected['weight']*selected_rf;p['selected_rf_alone']=selected_rf
        p['nested_original_control']=(1-control['weight'])*pair+control['weight']*control_rf;pieces.append(p)
        v=pd.DataFrame(dict(truth=truth,issue=str(issue),tuned_blend=(1-selected['weight'])*pp.to_numpy()+selected['weight']*rf[selected['config']],nested_original_control=(1-control['weight'])*pp.to_numpy()+control['weight']*rf['original'],selected_rf_alone=rf[selected['config']]),index=pp.index);vals.append(v)
    p=pd.concat(pieces);v=pd.concat(vals);assert p.index.equals(dev)
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time');q.dump(OUT/'SELECTIONS.json',decisions)
    pd.DataFrame(alltrials).to_csv(OUT/'all_validation_trials.csv',index=False)
    names=['baseline','vol_ratio','untuned_4features','nested_original_control','tuned_blend','selected_rf_alone']
    scores=pd.DataFrame([dict(split='test',model=n,**q.b.metric(p[n],p.truth)) for n in names]+[dict(split='nested_validation_selection_used',model=n,**q.b.metric(v[n],v.truth)) for n in ['nested_original_control','tuned_blend','selected_rf_alone']]);scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False)
    uncertainty=[]
    for reference in ['baseline','vol_ratio','untuned_4features','nested_original_control']:
        delta=abs(p.tuned_blend-p.truth)-abs(p[reference]-p.truth);block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
        b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
        boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    q.dump(OUT/'STATUS.json',dict(stage='audit'));saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    # Reload ALL artifacts; repeat fold predictions and selection, rather than trust in-memory training objects.
    for row in ledger:
        path=Path(row['path']);a=joblib.load(path);cut=pd.Timestamp(row['cutoff']);keys=y.index[y.label_end<cut]
        assert a['train_keys'].equals(keys) and a['issuance']==cut and y.loc[keys,'label_end'].max()<cut and a['prep']['columns']==COLS
        expected=s.prepare(x,keys,COLS)
        for field in ['median','center','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
        prototype=templates['pair'] if row['config']=='pair' else prototypes[row['config']]
        for k,value in prototype.get_params().items():
            if k!='n_jobs':assert a['model'].get_params()[k]==value
        assert q.sha(path)==row['sha256'];artifacts[str(cut),row['config']]=a
    prediction_cache.clear()
    for d in decisions:
        issue=pd.Timestamp(d['issue']);rows,control,selected,pp,rf,truth=trials_for(issue)
        assert (selected['config'],selected['weight'])==(d['selected']['config'],d['selected']['weight'])
        np.testing.assert_allclose(selected['mae'],d['selected']['mae'],atol=1e-8,rtol=0)
        _,_,_,_,te=q.windows(y,dev,issue)
        pair=q.b.predict(get(issue,'pair'),x.loc[te]);r=q.b.predict(get(issue,selected['config']),x.loc[te]);cr=q.b.predict(get(issue,'original'),x.loc[te])
        np.testing.assert_allclose((1-selected['weight'])*pair+selected['weight']*r,saved.loc[te,'tuned_blend'],atol=1e-8,rtol=0)
        np.testing.assert_allclose((1-control['weight'])*pair+control['weight']*cr,saved.loc[te,'nested_original_control'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(r,saved.loc[te,'selected_rf_alone'],atol=1e-8,rtol=0)
    assert all(q.sha(Path(path))==h for path,h in hashes.items())
    audit=dict(status='PASS',unique_artifacts=len(ledger),new_fits_this_invocation=sum(r['new_fit_this_invocation'] for r in ledger),configs=8,weights=6,inner_quarters=3,outer_windows=4,rows=len(p),fixed_four_features=True,strict_label_purges=True,training_only_preprocessing=True,parameter_audit=True,selection_replayed=True,checkpoint_replay=True,original_inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='Four-feature RF hyperparameter tuning: completed and audited\n\n'+scores.to_string(index=False)+'\n\nSelected settings:\n'+pd.DataFrame([dict(issue=d['issue'],config=d['selected']['config'],rf_weight=d['selected']['weight'],validation_mae=d['selected']['mae'],fold_wins=d['selected']['fold_wins']) for d in decisions]).to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nDescriptive paired weekly bootstrap:\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nSelection-used validation pools overlapping earlier quarters across outer windows; not comparable to old single-quarter validation score. Historical test exposed; no fresh confirmation.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8');q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,scores=scores.to_dict('records'),seconds=time.monotonic()-begin));q.dump(OUT/'STATUS.json',dict(stage='complete'));print(report,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
