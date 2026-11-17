"""Feature selection first, then LSTM architecture tests. Fifteen-minute compute cap."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy,json,math,time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/lstm_feature_architecture_v3'
MAX_EPOCHS=16
ARCHS=[(1,4,0),(1,8,0),(1,16,0),(2,8,0),(3,8,0),(1,8,8)]
DATA=None
LIMITER=None

class BudgetExpired(RuntimeError):pass

def check(deadline):
    if time.time()>=deadline:raise BudgetExpired('15-minute computation budget reached')

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')

def initialize():
    global DATA,LIMITER
    if DATA is not None:return
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LIMITER=threadpool_limits(limits=2)
    DATA=joblib.load(OUT/'prepared.joblib')

class Network(nn.Module):
    def __init__(self,inputs,layers,hidden,dense,median):
        super().__init__()
        self.rnn=nn.LSTM(inputs,hidden,num_layers=layers,batch_first=True,
                         dropout=.15 if layers>1 else 0.)
        if dense:
            self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(hidden+inputs,dense),
                                    nn.GELU(),nn.Dropout(.2),nn.Linear(dense,1))
        else:self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(hidden+inputs,1))
        nn.init.normal_(self.head[-1].weight,std=.01)
        nn.init.constant_(self.head[-1].bias,float(np.log(np.expm1(median/100))))
    def forward(self,x):
        h,_=self.rnn(x)
        return nn.functional.softplus(self.head(torch.cat([h[:,-1],x[:,-1]],1))).squeeze(1)

def infer(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[i:i+512]).numpy().astype(np.float64)*100
                               for i in range(0,len(x),512)])

def lr_at(epoch):
    if epoch<=2:return .0003*epoch/2
    return .00003+.5*(.0003-.00003)*(1+math.cos(math.pi*(epoch-2)/(MAX_EPOCHS-2)))

def spec(issue,columns,arch,phase='inner',epochs=None):
    layers,hidden,dense=arch
    tag=f'K{len(columns)}_L{layers}_H{hidden}_D{dense}'
    return dict(issue=str(issue),columns=columns,layers=layers,hidden=hidden,dense=dense,
                phase=phase,epochs=epochs,tag=tag)

def path_for(s):
    return OUT/'models'/pd.Timestamp(s['issue']).strftime('%Y%m%d')/(s['phase']+'_'+s['tag']+'.pt')

def fit(s,deadline):
    initialize();check(deadline)
    path=path_for(s)
    if path.exists():return str(path)
    x,y,dev,ref,packs,checks=DATA
    issue=pd.Timestamp(s['issue'])
    origin=issue-pd.DateOffset(months=1) if s['phase']=='inner' else issue
    tk=y.index[y.label_end<origin]
    vk=y.index[(y.index>=origin)&(y.index<issue)&(y.label_end<issue)] if s['phase']=='inner' else None
    assert y.loc[tk,'label_end'].max()<origin
    prep=q.prep(x,tk,s['columns']);sequence=dict(steps=24,stride=4)
    tx=q.seq(x,tk,prep,sequence)
    target=y.loc[tk,'magnitude_log_bps'].to_numpy(np.float32)
    ty=torch.from_numpy(target/100);median=float(np.median(target))
    vx=q.seq(x,vk,prep,sequence) if vk is not None else None
    vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
    torch.manual_seed(48);rng=np.random.default_rng(48)
    model=Network(tx.shape[-1],s['layers'],s['hidden'],s['dense'],median)
    weights=[p for p in model.parameters() if p.ndim>=2]
    biases=[p for p in model.parameters() if p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':weights,'weight_decay':.01},
                                {'params':biases,'weight_decay':0.}],lr=.0003)
    best=np.inf;stale=0;history=[];best_state=None;best_epoch=0
    started=time.monotonic()
    for epoch in range(1,(s['epochs'] or MAX_EPOCHS)+1):
        check(deadline)
        for group in optimizer.param_groups:group['lr']=lr_at(epoch)
        model.train();order=rng.permutation(len(tk));total=0.
        for i in range(0,len(tk),512):
            check(deadline);ix=order[i:i+512]
            optimizer.zero_grad(set_to_none=True)
            loss=(model(tx[ix])-ty[ix]).abs().mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=loss.item()*len(ix)*100
        val=float(abs(infer(model,vx)-vy).mean()) if vx is not None else None
        history.append(dict(epoch=epoch,learning_rate=lr_at(epoch),
                            train_mae=total/len(tk),validation_mae=val))
        if vx is None or val<best-.02:
            best=val if val is not None else total/len(tk)
            best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if vx is not None and epoch>=4 and stale>=3:break
    model.load_state_dict(best_state)
    a=dict(spec=s,prep=prep,sequence=sequence,state=best_state,epoch=best_epoch,
           inputs=tx.shape[-1],median=median,origin=origin,train_keys=tk,valid_keys=vk,
           max_label_end=y.loc[tk,'label_end'].max(),history=history,
           parameters=sum(p.numel() for p in model.parameters()),
           validation_prediction=infer(model,vx) if vx is not None else None,
           seconds=time.monotonic()-started)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');torch.save(a,temp);temp.replace(path)
    dump(path.with_suffix('.json'),{k:v for k,v in a.items() if k not in
        ['state','prep','train_keys','valid_keys','validation_prediction']})
    print('FIT',s['phase'],issue.date(),s['tag'],'epoch',best_epoch,
          'seconds',round(a['seconds'],1),flush=True)
    return str(path)

def batch(pool,specs,deadline):
    futures=[pool.submit(fit,s,deadline) for s in specs if not path_for(s).exists()]
    for future in as_completed(futures):future.result()
    return [torch.load(path_for(s),map_location='cpu',weights_only=False) for s in specs]

def evaluate(a):
    y=DATA[1]
    return dict(tag=a['spec']['tag'],spec=a['spec'],epoch=a['epoch'],
                parameters=a['parameters'],validation_mae=float(abs(
                    a['validation_prediction']-y.loc[a['valid_keys'],'magnitude_log_bps'].to_numpy()).mean()))

def rank(r):return (r['validation_mae'],r['parameters'],r['tag'])

def prediction(a,keys):
    model=Network(a['inputs'],a['spec']['layers'],a['spec']['hidden'],a['spec']['dense'],a['median'])
    model.load_state_dict(a['state'])
    return infer(model,q.seq(DATA[0],keys,a['prep'],a['sequence']))

def importance(issue,deadline):
    check(deadline)
    x,y,dev,ref,packs,checks=DATA
    start=issue-pd.DateOffset(months=3);stop=issue-pd.DateOffset(months=1)
    keys=y.index[(y.index>=start)&(y.index<stop)&(y.label_end<stop)]
    a=torch.load(q.OUT/issue.strftime('%Y%m%d')/'market_medium_inner.pt',
                 weights_only=False,map_location='cpu')
    assert a['issuance']==start and a['train_keys'].equals(y.index[y.label_end<start])
    assert y.loc[keys,'label_end'].max()<stop
    cols=a['prep']['columns'];tx=q.seq(x,keys,a['prep'],a['config'])
    model=q.Net(a['inputs'],a['config']['hidden'],a['median']);model.load_state_dict(a['state'])
    truth=y.loc[keys,'magnitude_log_bps'].to_numpy()
    base=abs(q.infer(model,tx)-truth)
    chunks=[np.arange(i,min(i+336,len(keys))) for i in range(0,len(keys),336)]
    perms=[np.concatenate([chunks[j] for j in np.random.default_rng(seed).permutation(len(chunks))])
           for seed in (48,149,250)]
    months=keys.strftime('%Y-%m');month_names=sorted(set(months));rows=[]
    for j,col in enumerate(cols):
        deltas=[]
        for perm in perms:
            check(deadline);changed=tx.clone()
            changed[:,:,j]=tx[perm,:,j]
            changed[:,:,j+len(cols)]=tx[perm,:,j+len(cols)]
            deltas.append(abs(q.infer(model,changed)-truth)-base)
        means=np.mean(deltas,axis=0)
        monthly={month:float(means[months==month].mean()) for month in month_names}
        rows.append(dict(feature=col,mae_increase_bps=float(means.mean()),
            permutation_std=float(np.std([v.mean() for v in deltas])),
            positive_months=sum(v>0 for v in monthly.values()),monthly=monthly))
    rows.sort(key=lambda r:(-r['mae_increase_bps'],-r['positive_months'],r['feature']))
    folder=OUT/'features'/issue.strftime('%Y%m%d');folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'importance.csv',index=False)
    dump(folder/'importance.json',dict(issue=str(issue),selection_start=str(start),
        selection_end=str(stop),max_label_end=str(y.loc[keys,'label_end'].max()),
        rows=len(keys),reference_model=str(q.OUT/issue.strftime('%Y%m%d')/'market_medium_inner.pt'),
        interpretation='Model-specific block-permutation MAE contribution; not causal importance. Correlated inputs can share importance.',
        ranked_features=rows))
    print('FEATURE RANKING',issue.date(),'top three',[r['feature'] for r in rows[:3]],flush=True)
    return [r['feature'] for r in rows]

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(target='absolute 48h endpoint log return in bps',primary='MAE only',
        budget_seconds=900,outer_starts=list(q.STARTS),
        phases='Finish feature selection for all windows, freeze it, then tune architectures',
        candidate_features='17 inputs from previous LSTM; no expansion of feature universe',
        feature_selection='Existing 17-feature inner LSTM, first two months of earlier quarter; labels mature strictly before architecture training origin; three 7-day-block permutations; value and missing-flag channels permuted together',
        count_grid=[2,4,8,17],count_control='17 features is a full-input control',
        count_model=dict(layers=1,hidden=8,dense=0),
        architecture_grid=ARCHS,architecture_validation='last month before outer issue; expanding training with label_end strictly before that month',
        max_epochs=MAX_EPOCHS,early_stopping='patience 3, minimum 4 epochs, >0.02 bps improvement',
        optimizer='AdamW; lr 0.0003; two-epoch warmup then cosine decay to 0.00003; decay 0.01 on matrices only',
        regularization='head dropout 0.2; inter-layer dropout 0.15 only for stacked LSTM; gradient norm clip 1; training-only imputation/scaling',
        seed=48,sequence_steps=24,sequence_stride_minutes=120,
        outer_refit='selected epoch count with identical deterministic LR schedule; same baseline outer training/test rows',
        scope='best among tested configurations, one seed; historical exposed evaluation; no test-driven retries',
        source_sha256=q.sha(__file__),helper_sha256=q.sha(HERE/'lstm_walk_forward.py'))
    if (OUT/'CONTRACT.json').exists():raise RuntimeError('Run directory already registered; no automatic extra budget or overwrite')
    dump(OUT/'CONTRACT.json',contract)
    (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=q.load();joblib.dump(data,OUT/'prepared.joblib',compress=3)
    manifest=json.loads((q.OUT/'MANIFEST.json').read_text())
    for path,digest in manifest['input_hashes'].items():assert q.sha(path)==digest
    manifest.update(prepared_sha256=q.sha(OUT/'prepared.joblib'),
                    source_sha256=contract['source_sha256'],helper_sha256=contract['helper_sha256'])
    dump(OUT/'MANIFEST.json',manifest)

def run():
    started=time.time();end=started+900;training_deadline=end-40
    try:
        prepare();initialize()
        x,y,dev,ref,packs,checks=DATA
        feature_choices=[];architecture_choices=[];all_trials=[];predictions=[]
        with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
            # Phase 1 is completed for every outer window before Phase 2 begins.
            for issue in q.STARTS:
                columns=importance(issue,training_deadline)
                specs=[spec(issue,columns[:k],(1,8,0)) for k in (2,4,8,17)]
                artifacts=batch(pool,specs,training_deadline)
                records=[evaluate(a) for a in artifacts];chosen=min(records,key=rank)
                feature_choices.append(dict(issue=str(issue),ranked_columns=columns,
                    count_trials=records,selected_columns=chosen['spec']['columns'],
                    selected_count=len(chosen['spec']['columns'])))
                all_trials += [dict(stage='feature_count',issue=str(issue),**r) for r in records]
                dump(OUT/'FEATURE_SELECTIONS.json',feature_choices)
                print('FEATURES FROZEN',issue.date(),len(chosen['spec']['columns']),
                      'validation MAE',chosen['validation_mae'],flush=True)
            print('PHASE 1 COMPLETE: FEATURE SELECTION FROZEN FOR ALL WINDOWS',flush=True)
            for feature_choice in feature_choices:
                issue=pd.Timestamp(feature_choice['issue']);columns=feature_choice['selected_columns']
                specs=[spec(issue,columns,arch) for arch in ARCHS]
                artifacts=batch(pool,specs,training_deadline)
                records=[evaluate(a) for a in artifacts];chosen=min(records,key=rank)
                all_trials += [dict(stage='architecture',issue=str(issue),**r) for r in records]
                selected=chosen['spec']
                outer=spec(issue,columns,(selected['layers'],selected['hidden'],selected['dense']),
                           phase='outer',epochs=chosen['epoch'])
                # Choice saved before outer fitting or test prediction.
                decision=dict(issue=str(issue),features=columns,trials=records,selected=chosen,outer_spec=outer)
                architecture_choices.append(decision)
                dump(OUT/'ARCHITECTURE_SELECTIONS.json',architecture_choices)
                a=batch(pool,[outer],training_deadline)[0]
                _,_,_,ot,te=q.windows(y,dev,issue)
                assert a['train_keys'].equals(ot)
                q.base_window(x,y,dev,ref,issue)
                pred=prediction(a,te)
                predictions.append(pd.DataFrame(dict(selected_lstm=pred,
                    baseline=ref.loc[te,'report_hybrid'],issue=str(issue)),index=te))
                print('OUTER WINDOW COMPLETE',issue.date(),selected['tag'],
                      'remaining seconds',round(end-time.time()),flush=True)
        p=pd.concat(predictions);assert p.index.equals(dev) and p.index.is_unique
        p['truth']=y.loc[dev,'magnitude_log_bps']
        previous=pd.read_csv(q.OUT/'predictions.csv',index_col=0,parse_dates=True)
        assert previous.index.equals(dev);p['previous_lstm']=previous.lstm
        p.to_csv(OUT/'predictions.csv',index_label='decision_time')
        dump(OUT/'TRIALS.json',all_trials)
        names=['baseline','previous_lstm','selected_lstm']
        scores=[dict(split=split,model=name,**q.b.metric(f[name],f.truth))
                for split,f in [('test',p),('last3months',p.loc['2024-11-01':])] for name in names]
        pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
        pd.DataFrame([dict(issue=i,model=n,**q.b.metric(f[n],f.truth))
                      for i,f in p.groupby('issue') for n in names]).to_csv(OUT/'window_scores.csv',index=False)
        audit(end)
        dump(OUT/'RUNTIME.json',dict(compute_seconds=time.time()-started,cap_seconds=900))
        print(pd.DataFrame(scores).to_string(index=False),flush=True)
    except BudgetExpired:
        dump(OUT/'STATUS.json',dict(status='TIME_BUDGET_REACHED',compute_seconds=time.time()-started,
             completed_fit_artifacts=len(list((OUT/'models').glob('*/*.pt'))),
             message='Stopped at requested cap. Incomplete comparison must not be presented as a full-period result.'))
        print('STOPPED AT TIME BUDGET',flush=True)

def audit(deadline=float('inf')):
    initialize();x,y,dev,ref,packs,checks=DATA
    p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert p.index.equals(dev) and p.index.is_unique
    np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(p.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
    count=0
    for path in (OUT/'models').glob('*/*.pt'):
        check(deadline)
        a=torch.load(path,map_location='cpu',weights_only=False);s=a['spec']
        issue=pd.Timestamp(s['issue']);origin=issue-pd.DateOffset(months=1) if s['phase']=='inner' else issue
        assert a['origin']==origin and a['train_keys'].equals(y.index[y.label_end<origin])
        assert a['max_label_end']<origin
        prep=q.prep(x,a['train_keys'],s['columns'])
        for key in ('median','scale'):pd.testing.assert_series_equal(prep[key],a['prep'][key])
        if s['phase']=='inner':
            assert y.loc[a['valid_keys'],'label_end'].max()<issue
            assert a['valid_keys'].min()>=origin
        count+=1
    selections=json.loads((OUT/'ARCHITECTURE_SELECTIONS.json').read_text())
    feature_selections=json.loads((OUT/'FEATURE_SELECTIONS.json').read_text())
    for s,fs in zip(selections,feature_selections):
        check(deadline)
        issue=pd.Timestamp(s['issue'])
        assert s['features']==fs['selected_columns']
        assert min(s['trials'],key=rank)['tag']==s['selected']['tag']
        importance_record=json.loads((OUT/'features'/issue.strftime('%Y%m%d')/'importance.json').read_text())
        assert pd.Timestamp(importance_record['max_label_end'])<issue-pd.DateOffset(months=1)
        a=torch.load(path_for(s['outer_spec']),weights_only=False,map_location='cpu')
        _,_,_,ot,te=q.windows(y,dev,issue)
        assert a['train_keys'].equals(ot) and a['epoch']==s['selected']['epoch']
        np.testing.assert_allclose(prediction(a,te),p.loc[te,'selected_lstm'],rtol=0,atol=1e-8)
        sample=te[::400]
        np.testing.assert_array_equal(q.seq(x,sample,a['prep'],a['sequence']).numpy(),
            q.seq(x.loc[:sample.max()],sample,a['prep'],a['sequence']).numpy())
    result=dict(status='PASS',fit_count=count,rows=len(p),strict_label_purge=True,
        feature_selection_before_architecture_validation=True,training_only_preprocessing=True,
        baseline_matched=True,checkpoint_replay=True,causal_prefix=True)
    dump(OUT/'AUDIT.json',result)
    scores=pd.read_csv(OUT/'scores.csv');test=scores[scores.split=='test'].set_index('model')
    dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=result,
        test_scores=test.to_dict('index'),beats_baseline=bool(test.loc['selected_lstm','mae']<test.loc['baseline','mae']),
        source_sha256=json.loads((OUT/'CONTRACT.json').read_text())['source_sha256']))
    print('AUDIT PASS',count,'fits',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit']);args=parser.parse_args()
    run() if args.action=='run' else audit()

