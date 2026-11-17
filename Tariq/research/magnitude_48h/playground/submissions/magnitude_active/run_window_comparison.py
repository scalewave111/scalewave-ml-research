"""Local controlled RF history comparison; no training on import or --check."""
import sys, threading, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from feature_experiment import splits, transform
from run_model_comparison import preflight, inputs, unchanged, read, write, dump, digest, metric, HERE, ROOT, CURRENT

OUT = HERE / 'window_trial_run'
MONTHS = [6, 12, 18]

def window(index, start, months):
    return index[index >= start-pd.DateOffset(months=months)]

def prune(keep):
    assert OUT.resolve().parent == HERE.resolve() and not OUT.is_symlink()
    for p in OUT.iterdir():
        assert p.is_file() and not p.is_symlink() and p.resolve().parent == OUT.resolve()
        if p.suffix in ['.joblib', '.partial', '.parquet'] and p.name not in keep:
            p.unlink()

def replay(result, X, base):
    parts=[]
    for card in result['folds']:
        idx=base.index[base.fold == card['fold']]
        p=base.loc[idx,'pair'].to_numpy().copy()
        if card['weight']:
            B,_=transform(X.loc[idx],card['preprocessing'])
            models=[joblib.load(OUT/name)['model'] for name in card['files']]
            fp=np.mean([np.maximum(m.predict(B),0) for m in models],axis=0)
            p=(1-card['weight'])*p+card['weight']*fp
        parts.append(pd.Series(p,index=idx))
    return pd.concat(parts)

def main():
    c,reg,X,y,base=preflight()
    cards={f:read(CURRENT/f'fold_{f}_selected.json')['selection'] for f in range(1,5)}
    hashes=inputs(c)
    for p in [Path(__file__)]+[CURRENT/f'fold_{f}_selected.json' for f in cards]:
        hashes[str(p.relative_to(ROOT))]=digest(p)
    plan=[]
    for months in MONTHS:
        for fold,ot,ov,it,iv in splits(c,y):
            start=pd.Timestamp(c['validation_windows'][fold-1][0])
            idx=window(ot,start,months)
            assert y.loc[idx,c['label_end_column']].max()<start
            assert len(idx)>1000
            assert all(X.loc[idx,f].count()>=1000 and X.loc[idx,f].nunique()>1 for f in X)
            sel=cards[fold]
            plan.append({'months':months,'fold':fold,'rows':len(idx),'start':str(idx.min()),
                'max_label_end':str(y.loc[idx,c['label_end_column']].max()),
                'requested_max_samples':sel['max_samples'],'effective_max_samples':min(sel['max_samples'],len(idx)),
                'weight':sel['weight']})
    print('WINDOW SETUP PASSED: 6/12/18 calendar months; fixed pair and blend weights; 27 RF fits maximum.',flush=True)
    if '--check' in sys.argv:
        for row in plan: print(row,flush=True)
        print('No training started.');return
    OUT.mkdir(exist_ok=True)
    contract=OUT/'contract.json'
    if contract.exists(): assert read(contract)['hashes']==hashes,'Inputs changed; review required.'
    else:
        assert not list(OUT.iterdir()),'Unexpected files; review required.'
        write(contract,{'hashes':hashes,'plan':plan,'benchmark':reg['current'],
            'protocol':'Only RF history changes; current pair, per-fold depth/leaf and blend weights fixed. 128 trees, seeds 48/17/91, eight workers. Bootstrap draws capped at available rows. Train-only preprocessing and strict label-end purge. No inner reselection. Zero-weight outer RF fits skipped because predictions are identical.',
            'interpretation':'Adaptive development comparison; not fresh confirmation. Winner retained for review, incumbent unchanged.'})
    statepath=OUT/'state.json'
    state=read(statepath) if statepath.exists() else {'results':{},'leader':None,'completed_fits':[]}
    for months in MONTHS:
        key=str(months)
        if key in state['results']:continue
        parts=[];foldcards=[];files=[]
        for fold,ot,ov,it,iv in splits(c,y):
            sel=cards[fold];w=sel['weight']
            start=pd.Timestamp(c['validation_windows'][fold-1][0]);idx=window(ot,start,months)
            card={'fold':fold,'weight':w,'files':[],'rows':len(idx)}
            p=base.loc[ov,'pair'].to_numpy().copy()
            if w:
                A,prep=transform(X.loc[idx]);B,_=transform(X.loc[ov],prep)
                params=dict(c['forest'],max_depth=sel['depth'],min_samples_leaf=sel['leaf'],
                            max_samples=min(sel['max_samples'],len(idx)),n_jobs=8)
                estimates=[]
                for seed in c['seeds']:
                    name=f'm{months}_fold{fold}_seed{seed}.joblib';path=OUT/name
                    params['random_state']=seed
                    if path.exists():
                        payload=joblib.load(path)
                        assert payload['params']==params and payload['preprocessing']==prep
                    else:
                        print(f'FIT window {months} months, period {fold}/4, seed {seed}',flush=True)
                        began=time.monotonic()
                        model=RandomForestRegressor(**params).fit(A,y.loc[idx,c['target_column']])
                        payload={'model':model,'params':dict(params),'preprocessing':prep,'seconds':time.monotonic()-began}
                        dump(path,payload);payload=joblib.load(path)
                    model=payload['model'];assert len(model.estimators_)==128
                    p1=np.maximum(model.predict(B),0)
                    p2=np.maximum(np.mean([t.predict(B) for t in model.estimators_],axis=0),0)
                    np.testing.assert_allclose(p1,p2,atol=1e-9,rtol=0)
                    estimates.append(p1);files.append(name);card['files'].append(name)
                    if name not in state['completed_fits']:state['completed_fits'].append(name)
                    write(statepath,state)
                p=(1-w)*p+w*np.mean(estimates,axis=0)
                card.update(preprocessing=prep,parameters=params)
            foldcards.append(card);parts.append(pd.Series(p,index=ov))
            print(f'{months} months: period {fold}/4 complete',flush=True)
        pred=pd.concat(parts);assert pred.index.equals(base.index) and np.isfinite(pred).all()
        result={'months':months,**metric(base.actual,pred),'folds':foldcards,
            'fold_scores':{str(f):metric(base.loc[base.fold==f,'actual'],pred.loc[base.fold==f]) for f in cards},
            'status':'VERIFIED','retained_files':files}
        np.testing.assert_allclose(replay(result,X,base),pred,atol=1e-9,rtol=0)
        unchanged(hashes)
        leader=state['leader'];best=reg['current']['MAE'] if leader is None else state['results'][leader]['MAE']
        if result['MAE']<best:
            name=f'm{months}_predictions.parquet'
            pd.DataFrame({'actual':base.actual,'prediction':pred,'fold':base.fold}).to_parquet(OUT/name)
            saved=pd.read_parquet(OUT/name);np.testing.assert_array_equal(saved.prediction,pred)
            result['retained_files'].append(name);state['leader']=key
        state['results'][key]=result
        # Commit compact scores before deleting superseded generated artifacts.
        for k,r in state['results'].items():
            if k!=state['leader']:r['retained_files']=[]
        write(statepath,state)
        keep=[] if state['leader'] is None else state['results'][state['leader']]['retained_files']
        prune(keep)
        print(f"{months} months VERIFIED: MAE {result['MAE']:.9f}; losing artifacts pruned.",flush=True)
    unchanged(hashes)
    if state['leader'] is not None:
        leader=state['results'][state['leader']];p=replay(leader,X,base)
        assert abs(metric(base.actual,p)['MAE']-leader['MAE'])<1e-9
    prune([] if state['leader'] is None else state['results'][state['leader']]['retained_files'])
    lines=['WINDOW COMPARISON COMPLETE - VERIFIED',f"Incumbent MAE: {reg['current']['MAE']:.9f}"]
    for k,r in state['results'].items():
        lines.append(f"{k} months: MAE {r['MAE']:.9f}; Spearman {r['Spearman_IC']:.9f}; improvement {reg['current']['MAE']-r['MAE']:+.9f}")
    lines += [f"Winning window retained: {state['leader'] or 'None'}",f"Completed fits: {len(state['completed_fits'])}",
        'Current model unchanged. Fixed pair and blend weights; bootstrap sizes capped for short histories.',
        'Losing binaries/predictions removed; compact scores retained. Fresh data untouched.']
    p=OUT/'WINDOW_RESULTS.txt'
    if p.exists():p.read_text()
    p.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':
    import msvcrt
    with (HERE/'window_trial.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Window trial already running.')
        stop=threading.Event()
        def heartbeat():
            while not stop.wait(60):print('Still running; completed fits saved.',flush=True)
        threading.Thread(target=heartbeat,daemon=True).start()
        try:main()
        except KeyboardInterrupt:print('Stopped. Same command resumes completed fits.');sys.exit(130)
        finally:stop.set()
