"""MAE-focused LSTM optimization with unchanged outer walk-forward windows.
Run: python lstm_optimize.py run
Replay: python lstm_optimize.py audit
All model choices use three quarters strictly earlier than each outer issuance.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
import argparse, copy, hashlib, json, os, time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/lstm_optimization_v2'
EPOCHS=40
SEEDS=(48,149,250)
WORKERS=2
DATA=None
CACHE=OrderedDict()
LIMITER=None

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')

def cfg(pack='flow',hidden=4,lr=.0001,steps=24,stride=4):
    return dict(pack=pack,hidden=hidden,lr=lr,steps=steps,stride=stride,decay=.01)

def cid(c):
    return f"{c['pack']}_h{c['hidden']}_lr{c['lr']:g}_n{c['steps']}_s{c['stride']}"

def base_grid():
    return [cfg(hidden=h,lr=lr) for h in (4,8) for lr in (.0001,.0003,.001)]+[cfg(hidden=16,lr=.002)]

def job(phase,issue,c,seed=48,epochs=EPOCHS):
    return dict(phase=phase,issue=str(issue),config=c,seed=int(seed),epochs=int(epochs))

def artifact_path(j):
    folder=OUT/j['phase']/pd.Timestamp(j['issue']).strftime('%Y%m%d')
    prefix=cid(j['config'])+f"__seed{j['seed']}_e"
    existing=[]
    for p in folder.glob(prefix+'*.pt'):
        n=int(p.stem.rsplit('_e',1)[1])
        if n>=j['epochs']: existing.append((n,p))
    if existing: return min(existing,key=lambda t:t[0])[1]
    return folder/(prefix+str(j['epochs'])+'.pt')

def initialize():
    global DATA,LIMITER
    if DATA is not None: return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LIMITER=threadpool_limits(limits=2)
    DATA=joblib.load(OUT/'prepared_data.joblib')

def split(issue,phase):
    x,y,dev,ref,packs,checks=DATA
    tk=y.index[y.label_end<issue]
    end=issue+pd.DateOffset(months=3)
    vk=y.index[(y.index>=issue)&(y.index<end)&(y.label_end<end)] if phase=='inner' else None
    return tk,vk

def tensors(j):
    x,y,dev,ref,packs,checks=DATA
    issue=pd.Timestamp(j['issue']);c=j['config']
    key=(j['phase'],str(issue),c['pack'],c['steps'],c['stride'])
    if key in CACHE:
        CACHE.move_to_end(key)
        return CACHE[key]
    tk,vk=split(issue,j['phase'])
    assert y.loc[tk,'label_end'].max()<issue
    p=q.prep(x,tk,packs[c['pack']])
    tx=q.seq(x,tk,p,c)
    target=y.loc[tk,'magnitude_log_bps'].to_numpy(np.float32)
    ty=torch.from_numpy(target/100)
    vx=q.seq(x,vk,p,c) if vk is not None else None
    vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
    value=(tk,vk,p,tx,ty,vx,vy,float(np.median(target)))
    CACHE[key]=value
    while len(CACHE)>2: CACHE.popitem(last=False)
    return value

def fit_job(j):
    initialize()
    path=artifact_path(j)
    if path.exists():
        a=torch.load(path,map_location='cpu',weights_only=False)
        assert a['config']==j['config'] and a['seed']==j['seed'] and a['epochs']>=j['epochs']
        assert a['issuance']==pd.Timestamp(j['issue']) and a['phase']==j['phase']
        return dict(path=str(path),reused=True,seconds=a['seconds'])
    path.parent.mkdir(parents=True,exist_ok=True)
    tk,vk,p,tx,ty,vx,vy,median=tensors(j)
    c=j['config'];torch.manual_seed(j['seed']);rng=np.random.default_rng(j['seed'])
    model=q.Net(tx.shape[-1],c['hidden'],median)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['decay'])
    states=[];predictions=[];history=[];start=time.monotonic()
    for epoch in range(1,j['epochs']+1):
        model.train();order=rng.permutation(len(tk));loss_sum=0.
        for k in range(0,len(tk),q.BATCH):
            idx=order[k:k+q.BATCH]
            opt.zero_grad(set_to_none=True)
            loss=(model(tx[idx])-ty[idx]).abs().mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();loss_sum+=loss.item()*len(idx)*100
        if vx is not None:
            pred=q.infer(model,vx)
            predictions.append(pred)
            val=float(np.mean(abs(pred-vy)))
            states.append(copy.deepcopy(model.state_dict()))
        else:
            val=None
        history.append(dict(epoch=epoch,train_mae=loss_sum/len(tk),validation_mae=val))
        if epoch==1 or epoch%10==0:
            print('EPOCH',j['phase'],pd.Timestamp(j['issue']).date(),cid(c),
                  'seed',j['seed'],epoch,'seconds',round(time.monotonic()-start,1),flush=True)
    if vx is None: states=[copy.deepcopy(model.state_dict())]
    y=DATA[1]
    a=dict(config=c,seed=j['seed'],epochs=j['epochs'],phase=j['phase'],
        issuance=pd.Timestamp(j['issue']),train_keys=tk,valid_keys=vk,prep=p,
        max_label_end=y.loc[tk,'label_end'].max(),inputs=tx.shape[-1],median=median,
        states=states,validation_predictions=np.stack(predictions) if predictions else None,
        history=history,seconds=time.monotonic()-start)
    temp=path.with_suffix('.partial')
    torch.save(a,temp);temp.replace(path)
    dump(path.with_suffix('.json'),{k:v for k,v in a.items() if k not in
        ['states','validation_predictions','train_keys','valid_keys','prep']})
    print('FIT COMPLETE',path.name,'origin',a['issuance'].date(),
          'seconds',round(a['seconds'],1),flush=True)
    return dict(path=str(path),reused=False,seconds=a['seconds'])

def ensure(pool,jobs):
    distinct={str(artifact_path(j)):j for j in jobs}
    pending=[j for path,j in distinct.items() if not Path(path).exists()]
    print('BATCH',len(pending),'new fits;',len(distinct)-len(pending),'cached',flush=True)
    futures=[pool.submit(fit_job,j) for j in pending]
    for f in as_completed(futures): f.result()
    for j in distinct.values(): assert artifact_path(j).exists()

def read_artifact(j):
    a=torch.load(artifact_path(j),weights_only=False,map_location='cpu')
    assert a['config']==j['config'] and a['seed']==j['seed']
    assert a['issuance']==pd.Timestamp(j['issue']) and a['epochs']>=j['epochs']
    return a

def evaluation(c,origins,seeds=(48,),fixed_epoch=None):
    y=DATA[1];loss=None;count=0;folds=[];per_seed={s:[] for s in seeds}
    for origin in origins:
        arr=[];keys=None
        for seed in seeds:
            a=read_artifact(job('inner',origin,c,seed,fixed_epoch or EPOCHS))
            if keys is None: keys=a['valid_keys']
            else: assert keys.equals(a['valid_keys'])
            pred=a['validation_predictions'].astype(np.float64)
            pred=pred[fixed_epoch-1:fixed_epoch] if fixed_epoch else pred[:EPOCHS]
            arr.append(pred)
            per_seed[seed].append((pred,keys))
        ensemble=np.mean(arr,axis=0);truth=y.loc[keys,'magnitude_log_bps'].to_numpy()
        errors=abs(ensemble-truth).sum(axis=1)
        loss=errors if loss is None else loss+errors
        count+=len(keys);folds.append((origin,ensemble,keys))
    index=0 if fixed_epoch else int(np.argmin(loss))
    epoch=fixed_epoch or index+1
    fold_metrics=[dict(origin=str(o),rows=len(k),mae=float(abs(p[index]-y.loc[k,'magnitude_log_bps'].to_numpy()).mean())) for o,p,k in folds]
    seed_metrics={}
    for seed,items in per_seed.items():
        seed_metrics[str(seed)]=sum(float(abs(p[index]-y.loc[k,'magnitude_log_bps'].to_numpy()).sum()) for p,k in items)/count
    return dict(id=cid(c),config=c,epoch=epoch,mae=float(loss[index]/count),
                folds=fold_metrics,seed_mae=seed_metrics,rows=count,seeds=list(seeds))

def rank(rec):
    c=rec['config']
    return (rec['mae'],c['hidden'],len(DATA[4][c['pack']]),c['steps'],rec['id'])

def selected_validation(rec,origins):
    result=[]
    for origin in origins:
        arrays=[];keys=None
        for seed in rec['seeds']:
            a=read_artifact(job('inner',origin,rec['config'],seed,rec['epoch']))
            keys=a['valid_keys']
            arrays.append(a['validation_predictions'][rec['epoch']-1].astype(np.float64))
        result.append((origin,keys,np.mean(arrays,axis=0)))
    return result

def calibration(rec,origins):
    y=DATA[1];folds=selected_validation(rec,origins)
    residuals=np.concatenate([y.loc[k,'magnitude_log_bps'].to_numpy()-p for _,k,p in folds[:2]])
    offset=float(np.median(residuals))
    _,keys,pred=folds[2];truth=y.loc[keys,'magnitude_log_bps'].to_numpy()
    before=float(abs(pred-truth).mean());after=float(abs(np.maximum(0,pred+offset)-truth).mean())
    accepted=after<before-.1
    return dict(offset=offset if accepted else 0.,proposed_offset=offset,accepted=accepted,
        fit_origins=[str(o) for o,_,_ in folds[:2]],check_origin=str(folds[2][0]),
        check_mae_raw=before,check_mae_corrected=after,
        rule='Fit median residual on first two folds; retain only if third-fold MAE improves by >0.1 bps; freeze that offset for outer predictions')

def model_prediction(a,x,keys,epoch=None):
    model=q.Net(a['inputs'],a['config']['hidden'],a['median'])
    state=a['states'][epoch-1] if a['phase']=='inner' else a['states'][0]
    model.load_state_dict(state)
    return q.infer(model,q.seq(x,keys,a['prep'],a['config'])).astype(np.float64)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(target='absolute 48h endpoint log return in bps',objective='MAE only; no IC selection',
        outer_starts=list(q.STARTS),outer_rule='same expanding training rows and test rows as v1/report baseline',
        purge='label_end strictly before each issuance; validation labels complete strictly before next quarter',
        validation='three immediately preceding quarterly origins for each outer issue; pooled row-weighted MAE',
        stage_A=base_grid(),stage_B='matched pair/flow/market features at stage-A winning hyperparameters',
        stage_C='matched histories: 1x30m,16x30m,24x2h,42x4h at stage-B winning feature set',
        epochs=EPOCHS,epoch_selection='shared epoch minimizing pooled MAE across all three folds',
        finalists='two best unique staged candidates; repeat at fixed chosen epoch using all three seeds; choose by pooled ensemble MAE',
        seeds=list(SEEDS),calibration='fit median residual on first two folds, check third; >0.1 bps gain to retain; freeze offset',
        loss='MAE/100',dropout=.1,weight_decay=.01,batch=q.BATCH,gradient_clip=1.,
        outer='three-seed equal mean; report raw and validation-gated calibrated LSTM separately; no baseline fallback',
        selection_access='only earlier validation outcomes; never outer outcomes for current window choices',
        stopping='one staged search; no test-driven retries within this run',
        exposure='Historical outer period was already examined; exploratory follow-up, not fresh confirmation',
        source_sha256=q.sha(__file__),helper_sha256=q.sha(HERE/'lstm_walk_forward.py'),workers=WORKERS)
    if (OUT/'CONTRACT.json').exists():
        old=json.loads((OUT/'CONTRACT.json').read_text())
        assert old['source_sha256']==contract['source_sha256'] and old['helper_sha256']==contract['helper_sha256']
    else:
        dump(OUT/'CONTRACT.json',contract)
        (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'helper_source_snapshot.py').write_bytes((HERE/'lstm_walk_forward.py').read_bytes())
    if not (OUT/'prepared_data.joblib').exists():
        data=q.load()
        joblib.dump(data,OUT/'prepared_data.joblib',compress=3)
        manifest=json.loads((q.OUT/'MANIFEST.json').read_text())
        for path,digest in manifest['input_hashes'].items(): assert q.sha(path)==digest
        manifest.update(prepared_sha256=q.sha(OUT/'prepared_data.joblib'),
            source_sha256=contract['source_sha256'],helper_sha256=contract['helper_sha256'],
            context_sha256=q.sha(HERE/'PROJECT_CONTEXT.md'),logical_cpus=os.cpu_count())
        dump(OUT/'MANIFEST.json',manifest)
    print('DATA READY; run contract frozen;',os.cpu_count(),'logical CPUs;',WORKERS,'workers',flush=True)

def run():
    if (OUT/'COMPLETE.json').exists(): raise RuntimeError('Completed run exists; use audit')
    prepare();initialize()
    x,y,dev,ref,packs,checks=DATA
    decisions=[];trials=[];predictions=[];validations=[]
    # One real, reusable fit provides runtime measurement before parallel batches.
    first=q.STARTS[0]-pd.DateOffset(months=9)
    pilot=fit_job(job('inner',first,base_grid()[0]))
    print('FIRST FIT TIMING',pilot,flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=initialize) as pool:
        for issue in q.STARTS:
            origins=[issue-pd.DateOffset(months=n) for n in (9,6,3)]
            folder=OUT/'selections'/issue.strftime('%Y%m%d');folder.mkdir(parents=True,exist_ok=True)
            candidates={}
            stage_winners={}
            grid=base_grid()
            for stage in ('A_size_rate','B_features','C_history'):
                ensure(pool,[job('inner',o,c) for c in grid for o in origins])
                records=[]
                for c in grid:
                    rec=evaluation(c,origins);candidates[cid(c)]=rec;records.append(rec)
                    trials.append(dict(outer_issue=str(issue),stage=stage,**rec))
                winner=min(records,key=rank);stage_winners[stage]=winner
                print('STAGE',issue.date(),stage,winner['id'],'epoch',winner['epoch'],'validation MAE',winner['mae'],flush=True)
                c=winner['config']
                if stage=='A_size_rate': grid=[dict(c,pack=p) for p in ('pair','flow','market')]
                elif stage=='B_features': grid=[dict(c,steps=n,stride=s) for n,s in ((1,1),(16,1),(24,4),(42,8))]
                dump(folder/'stage_trials.json',[t for t in trials if t['outer_issue']==str(issue)])
            finalists=sorted(candidates.values(),key=rank)[:2]
            ensure(pool,[job('inner',o,r['config'],seed,r['epoch']) for r in finalists for o in origins for seed in SEEDS[1:]])
            robust=[evaluation(r['config'],origins,SEEDS,r['epoch']) for r in finalists]
            selected=min(robust,key=rank);cal=calibration(selected,origins)
            _,ik,vk,ot,te=q.windows(y,dev,issue)
            q.base_window(x,y,dev,ref,issue)
            selection=dict(issue=str(issue),validation_origins=list(origins),stage_winners=stage_winners,
                finalists=robust,selected=selected,calibration=cal,train_rows=len(ot),test_rows=len(te),
                max_training_label_end=str(y.loc[ot,'label_end'].max()))
            dump(folder/'selection.json',selection)
            ensure(pool,[job('outer',issue,selected['config'],seed,selected['epoch']) for seed in SEEDS])
            seed_predictions={}
            for seed in SEEDS:
                a=read_artifact(job('outer',issue,selected['config'],seed,selected['epoch']))
                assert a['train_keys'].equals(ot)
                seed_predictions[f'seed_{seed}']=model_prediction(a,x,te)
            pred=np.mean(list(seed_predictions.values()),axis=0)
            p=pd.DataFrame(dict(optimized_lstm=pred,calibrated_lstm=np.maximum(0,pred+cal['offset']),
                               baseline=ref.loc[te,'report_hybrid'],issue=str(issue),**seed_predictions),index=te)
            p.to_csv(folder/'predictions_unscored.csv',index_label='decision_time');predictions.append(p)
            for origin,keys,vpred in selected_validation(selected,origins):
                validations.append(pd.DataFrame(dict(outer_issue=str(issue),validation_origin=str(origin),
                    prediction=vpred,truth=y.loc[keys,'magnitude_log_bps'],latest=origin==origins[-1]),index=keys))
            decisions.append(selection);dump(OUT/'SELECTIONS.json',decisions);dump(OUT/'TRIALS.json',trials)
            print('OUTER WINDOW FROZEN',issue.date(),'selected',selected['id'],'epoch',selected['epoch'],
                  'calibration',cal['offset'],'test rows',len(te),flush=True)
    p=pd.concat(predictions)
    assert p.index.equals(dev) and p.index.is_unique
    p['truth']=y.loc[dev,'magnitude_log_bps']
    previous=pd.read_csv(q.OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert previous.index.equals(dev)
    p['previous_lstm']=previous.lstm
    p.to_csv(OUT/'predictions.csv',index_label='decision_time')
    pd.concat(validations).to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
    names=['baseline','previous_lstm','optimized_lstm','calibrated_lstm']+[f'seed_{s}' for s in SEEDS]
    scores=[dict(split=split,model=name,**q.b.metric(f[name],f.truth))
        for split,f in [('test',p),('last3months',p.loc['2024-11-01':])] for name in names]
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    windows=[dict(issue=issue,model=name,**q.b.metric(f[name],f.truth)) for issue,f in p.groupby('issue') for name in names]
    pd.DataFrame(windows).to_csv(OUT/'window_scores.csv',index=False)
    dump(OUT/'TRAINING_FINISHED.json',dict(status='TRAINED_AWAITING_AUDIT',scores=scores))
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    audit()

def audit():
    initialize()
    x,y,dev,ref,packs,checks=DATA
    contract=json.loads((OUT/'CONTRACT.json').read_text())
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert q.sha(OUT/'training_source_snapshot.py')==contract['source_sha256']
    assert q.sha(HERE/'lstm_walk_forward.py')==contract['helper_sha256']
    assert q.sha(OUT/'prepared_data.joblib')==manifest['prepared_sha256']
    for path,digest in manifest['input_hashes'].items(): assert q.sha(path)==digest
    p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert p.index.equals(dev) and p.index.is_unique and len(p)==18481
    np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(p.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
    count=0;preps={};max_error=0.
    for path in list((OUT/'inner').glob('*/*.pt'))+list((OUT/'outer').glob('*/*.pt')):
        a=torch.load(path,weights_only=False,map_location='cpu');issue=a['issuance']
        tk,vk=split(issue,a['phase'])
        assert a['train_keys'].equals(tk) and a['max_label_end']<issue
        key=(str(issue),a['config']['pack'])
        if key not in preps: preps[key]=q.prep(x,tk,packs[a['config']['pack']])
        for field in ('median','scale'): pd.testing.assert_series_equal(a['prep'][field],preps[key][field])
        if a['phase']=='inner':
            assert a['valid_keys'].equals(vk) and y.loc[vk,'label_end'].max()<issue+pd.DateOffset(months=3)
            assert len(a['states'])==a['epochs'] and np.isfinite(a['validation_predictions']).all()
        count+=1
    decisions=json.loads((OUT/'SELECTIONS.json').read_text())
    for s in decisions:
        issue=pd.Timestamp(s['issue']);origins=[pd.Timestamp(o) for o in s['validation_origins']]
        rec=s['selected'];recomputed=evaluation(rec['config'],origins,SEEDS,rec['epoch'])
        assert abs(recomputed['mae']-rec['mae'])<1e-10
        computed_cal=calibration(rec,origins)
        assert computed_cal==s['calibration']
        _,_,_,ot,te=q.windows(y,dev,issue)
        q.base_window(x,y,dev,ref,issue)
        preds=[]
        for seed in SEEDS:
            a=read_artifact(job('outer',issue,rec['config'],seed,rec['epoch']))
            assert a['train_keys'].equals(ot)
            replay=model_prediction(a,x,te);preds.append(replay)
            np.testing.assert_allclose(replay,p.loc[te,f'seed_{seed}'],rtol=0,atol=1e-8)
            sample=te[::max(1,len(te)//12)]
            np.testing.assert_array_equal(q.seq(x,sample,a['prep'],a['config']).numpy(),
                q.seq(x.loc[:sample.max()],sample,a['prep'],a['config']).numpy())
            for origin in origins:
                inner=read_artifact(job('inner',origin,rec['config'],seed,rec['epoch']))
                replay_val=model_prediction(inner,x,inner['valid_keys'],rec['epoch'])
                np.testing.assert_array_equal(replay_val,inner['validation_predictions'][rec['epoch']-1].astype(np.float64))
        raw=np.mean(preds,axis=0);corrected=np.maximum(0,raw+computed_cal['offset'])
        for name,values in [('optimized_lstm',raw),('calibrated_lstm',corrected)]:
            max_error=max(max_error,float(np.max(abs(values-p.loc[te,name]))))
            np.testing.assert_allclose(values,p.loc[te,name],rtol=0,atol=1e-8)
    scores=pd.read_csv(OUT/'scores.csv')
    test=scores[scores.split=='test'].set_index('model')
    for name in test.index:
        assert abs(q.b.metric(p[name],p.truth)['mae']-test.loc[name,'mae'])<1e-8
    result=dict(status='PASS',fit_artifacts=count,test_rows=len(p),baseline_reproduced=True,
        input_hashes_unchanged=True,training_only_preprocessing=True,three_purged_prior_folds=True,
        selected_validation_replayed=True,outer_predictions_replayed=True,causal_prefix=True,
        calibration_replayed=True,max_replay_error=max_error)
    dump(OUT/'AUDIT.json',result)
    dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=result,test_scores=test.to_dict('index'),
        source_sha256=contract['source_sha256'],audit_source_sha256=q.sha(__file__),
        optimized_beats_baseline=bool(test.loc['optimized_lstm','mae']<test.loc['baseline','mae']),
        calibrated_beats_baseline=bool(test.loc['calibrated_lstm','mae']<test.loc['baseline','mae']),
        limitation=contract['exposure']))
    print('AUDIT PASS',count,'fit artifacts;',len(p),'test rows',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','prepare','audit']);args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    else: run()

