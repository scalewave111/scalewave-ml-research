"""Fixed fourth-feature MLP test, with verified reuse of identical completed fits."""
from pathlib import Path
import json,time,shutil,importlib.metadata
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits
import mlp_feature_discovery as f
m=f.m;q=f.q
HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/mlp_fixed_liquidation_1h_v1'
ADDED='F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value'
COLS=f.CORE+[ADDED]

def run():
    begin=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'linear').mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    contract=dict(features=COLS,primary='fixed_four_feature_mlp',seeds=m.SEEDS,
        method='Exactly prior linear-initialized residual MLP: frozen training-only median-regression skip alpha=.0001 on target/100; 8/4 SiLU nonlinear branch, zero head, dropout .05, AdamW lr .001 decay .01 nonlinear weights only, clip gradient1, batch512; clip nonnegative output.',
        preprocessing='Training median/IQR with std fallback, [-8,8] clipping, values only; original expanding rows and strict label-end purges.',
        selection='Feature fixed for all four windows; no inclusion gate, no fallback, no feature selection. Prior-quarter MAE selects epoch separately per seed: max48, patience8, minimum improvement .01bps, including epoch0. Refit on outer expanding history and average three seeds.',
        reuse='Copy exact audited final-window checkpoints and Jul/Oct2024 linear initializations after checking feature order, training rows, cutoff and preprocessing; same engine/source hashes. Reuse is not new evidence.',
        limitation='Fourth feature chosen following its prior final-window success. All historical evaluation exposed; fixed-feature retest is exploratory, not independent confirmation. No test-driven retries.',
        planned_new_fits='18 neural fits and3 distinct linear fits; reuse6 neural and2 linear artifacts.',source_sha=q.sha(__file__),engine_sha=q.sha(m.__file__),adapter_sha=q.sha(f.__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    x,y,dev=f.load();x=x[COLS].copy();m.COLS=COLS;m.OUT=OUT
    prior=pd.read_csv(f.BASE/'predictions.csv',index_col=0,parse_dates=True);priorv=pd.read_csv(f.BASE/'validation_predictions.csv',index_col=0,parse_dates=True)
    assert prior.index.equals(dev) and len(dev)==18481
    old_contract=json.loads((f.OUT/'CONTRACT.json').read_text());assert old_contract['engine_sha']==q.sha(m.__file__) and old_contract['source_sha']==q.sha(f.__file__)
    assert json.loads((f.OUT/'AUDIT.json').read_text())['status']=='PASS'
    source=f.OUT/'20241001';reused={}
    for phase,cut in [('inner',pd.Timestamp('2024-07-01',tz='UTC')),('outer',pd.Timestamp('2024-10-01',tz='UTC'))]:
        keys=y.index[y.label_end<cut];lp=source/'linear'/f'{cut:%Y%m%d}.joblib';a=joblib.load(lp)
        assert a['train_keys'].equals(keys) and a['cutoff']==cut and a['prep']['columns']==COLS
        expected=q.prep(x,keys,COLS)
        for field in ['median','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
        dest=OUT/'linear'/lp.name
        if not dest.exists():shutil.copyfile(lp,dest)
        assert q.sha(lp)==q.sha(dest);reused[str(lp)]=q.sha(lp)
        folder=OUT/'20241001';folder.mkdir(exist_ok=True)
        for seed in m.SEEDS:
            sp=source/f'{seed}_{phase}.pt';a=torch.load(sp,weights_only=False)
            assert a['prep']['columns']==COLS and a['train_keys'].equals(keys) and a['cutoff']==cut and a['seed']==seed
            dest=folder/sp.name
            if not dest.exists():shutil.copyfile(sp,dest)
            assert q.sha(sp)==q.sha(dest);reused[str(sp)]=q.sha(sp)
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework');docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    sources=[f.BASE/'predictions.csv',f.BASE/'validation_predictions.csv',f.SRC/'prepared.joblib',f.DICTIONARY]+[q.b.old.PACKAGE/'data'/n for n in ['X_train.parquet','X_development.parquet','y_train.parquet','y_development.parquet']]
    hashes={str(p):q.sha(p) for p in sources}
    if not (OUT/'MANIFEST.json').exists():q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,reused_artifacts=reused,instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','joblib']}))
    pieces=[];validations=[];training=[];decisions=[];seedrows=[];count=0
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue);assert y.loc[vk,'label_end'].max()<issue
        folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        q.dump(OUT/'STATUS.json',dict(stage='inner_linear',issue=issue));bi=m.linear_fit(x,y,ik,inner)
        inners=[]
        for seed in m.SEEDS:
            q.dump(OUT/'STATUS.json',dict(stage='inner_mlp',issue=issue,seed=seed,completed_checkpoints=count))
            a=m.fit(x,y,bi,seed,folder/f'{seed}_inner.pt',vk=vk);inners.append(a);count+=1
            print('CHECKPOINT',count,'/24',issue.date(),'inner',seed,'epoch',a['epoch'],flush=True)
        d=dict(issue=str(issue),feature=ADDED,epochs=[a['epoch'] for a in inners],reused_neural=bool(issue==q.STARTS[-1]));q.dump(folder/'selection.json',d);decisions.append(d)
        vv=priorv.loc[vk,['baseline','vol_ratio','initialized_mlp','truth','issue']].rename(columns={'initialized_mlp':'mlp_3features'}).copy()
        vv['linear_4features']=m.linear_predict(bi,x,vk);vv['fixed_four_feature_mlp']=np.mean([m.predict(a,x,vk) for a in inners],axis=0)
        q.dump(OUT/'STATUS.json',dict(stage='outer_linear',issue=issue));bo=m.linear_fit(x,y,ot,issue)
        pp=prior.loc[te,['baseline','vol_ratio','initialized_mlp','truth','issue']].rename(columns={'initialized_mlp':'mlp_3features'}).copy();pp['linear_4features']=m.linear_predict(bo,x,te)
        tt=pd.DataFrame(dict(truth=y.loc[ot,'magnitude_log_bps'],linear_4features=m.linear_predict(bo,x,ot),issue=str(issue)),index=ot)
        preds=[];tr=[]
        for a in inners:
            seed=a['seed'];q.dump(OUT/'STATUS.json',dict(stage='outer_mlp',issue=issue,seed=seed,completed_checkpoints=count))
            b=m.fit(x,y,bo,seed,folder/f'{seed}_outer.pt',epochs=a['epoch']);count+=1
            pred=m.predict(b,x,te);preds.append(pred);tr.append(m.predict(b,x,ot));seedrows.append(dict(issue=str(issue),seed=seed,**q.b.metric(pred,pp.truth)))
            print('CHECKPOINT',count,'/24',issue.date(),'outer',seed,'epoch',b['epoch'],flush=True)
        pp['fixed_four_feature_mlp']=np.mean(preds,axis=0);tt['fixed_four_feature_mlp']=np.mean(tr,axis=0)
        pieces.append(pp);validations.append(vv);training.append(tt)
    p=pd.concat(pieces);v=pd.concat(validations);t=pd.concat(training);assert p.index.equals(dev)
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time');q.dump(OUT/'SELECTIONS.json',decisions)
    names=['baseline','vol_ratio','mlp_3features','linear_4features','fixed_four_feature_mlp']
    scores=pd.DataFrame([dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]+[dict(split='training_in_sample',model=n,**q.b.metric(t[n],t.truth)) for n in ['linear_4features','fixed_four_feature_mlp']]);scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'seed_window_scores.csv',index=False)
    extreme=pd.DataFrame([dict(model=n,**q.b.metric(p.loc[p.truth>500,n],p.loc[p.truth>500,'truth'])) for n in names]);extreme.to_csv(OUT/'extreme_scores.csv',index=False)
    first3=p.loc[p.index<q.STARTS[-1]];pd.DataFrame([dict(model=n,**q.b.metric(first3[n],first3.truth)) for n in names]).to_csv(OUT/'first_three_windows_scores.csv',index=False)
    uncertainty=[]
    for reference in ['baseline','vol_ratio','mlp_3features','linear_4features']:
        delta=abs(p.fixed_four_feature_mlp-p.truth)-abs(p[reference]-p.truth);block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
        b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
        boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    q.dump(OUT/'STATUS.json',dict(stage='audit'));saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    for d in decisions:
        issue=pd.Timestamp(d['issue']);inner,ik,vk,ot,te=q.windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d')
        for phase,keys,cut,pk,df in [('inner',ik,inner,vk,v),('outer',ot,issue,te,saved)]:
            base=joblib.load(OUT/'linear'/f'{cut:%Y%m%d}.joblib');assert base['train_keys'].equals(keys) and base['cutoff']==cut and y.loc[keys,'label_end'].max()<cut
            assert base['prep']['columns']==COLS;expected=q.prep(x,keys,COLS)
            for field in ['median','scale']:pd.testing.assert_series_equal(base['prep'][field],expected[field])
            lp=m.linear_predict(base,x,pk);np.testing.assert_allclose(lp,df.loc[pk,'linear_4features'],atol=1e-8,rtol=0);preds=[]
            for j,seed in enumerate(m.SEEDS):
                a=torch.load(folder/f'{seed}_{phase}.pt',weights_only=False);assert a['train_keys'].equals(keys) and a['cutoff']==cut and a['epoch']==d['epochs'][j] and a['seed']==seed and a['prep']['columns']==COLS
                for field in ['median','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
                initial=f.Net(base['coef'],base['intercept'])
                for field in ['weight','bias']:torch.testing.assert_close(a['state']['linear.'+field],initial.linear.state_dict()[field],atol=0,rtol=0)
                if phase=='inner':
                    assert a['validation_keys'].equals(vk);best=a['history'][0]['validation_mae'];ep=0
                    for row in a['history'][1:]:
                        if row['validation_mae']<best-.01:best=row['validation_mae'];ep=row['epoch']
                    assert a['epoch']==ep
                else:assert a['validation_keys'] is None
                pred=m.predict(a,x,pk)
                if a['epoch']==0:np.testing.assert_allclose(pred,lp,atol=0,rtol=0)
                preds.append(pred)
            np.testing.assert_allclose(np.mean(preds,axis=0),df.loc[pk,'fixed_four_feature_mlp'],atol=1e-8,rtol=0)
    reused_pred=pd.read_csv(f.OUT/'predictions.csv',index_col=0,parse_dates=True);finalkeys=dev[dev>=q.STARTS[-1]]
    np.testing.assert_allclose(saved.loc[finalkeys,'fixed_four_feature_mlp'],reused_pred.loc[finalkeys,'challenger_mlp'],atol=1e-8,rtol=0)
    assert count==24 and all(q.sha(Path(path))==h for path,h in {**hashes,**reused}.items())
    audit=dict(status='PASS',neural_checkpoints=24,new_neural_fits=18,reused_neural_fits=6,new_linear_fits=3,reused_linear_fits=2,rows=len(p),fixed_four_features=True,strict_label_purges=True,training_only_preprocessing=True,linear_skip_preserved=True,epoch_and_prediction_replay=True,reused_window_reproduced=True,original_inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='Fixed fourth feature: 1h liquidation historical active rate\n\n'+scores.to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nDescriptive paired weekly block bootstrap:\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nFinal window exactly reuses prior experiment; this feature was chosen after its success there. Entire historical evaluation already exposed; not independent confirmation. Training scores pool overlapping expanding windows.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8');q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,scores=scores.to_dict('records'),seconds=time.monotonic()-begin));q.dump(OUT/'STATUS.json',dict(stage='complete'));print(report,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):run()
