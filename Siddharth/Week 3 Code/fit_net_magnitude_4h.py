"""4h canonical absolute endpoint magnitude: user-run tree/neural feature search.

python Code/fit_net_magnitude_4h.py --check-inputs
python Code/fit_net_magnitude_4h.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import importlib.metadata
import numpy as np
import pandas as pd
import duckdb
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = 'net_magnitude_log_bps'
FORMULA = '10000 * abs(ln(P_end / P_start)); canonical absolute endpoint log return'
ISSUES = pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'], utc=True)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def engineered(r, origins):
    """All windows exclude return endpoints at/after origin; full coverage required."""
    v = {}; definitions = []
    def add(name, series, formula):
        v['NET4__'+name] = series
        definitions.append(dict(feature='NET4__'+name, formula=formula))
    for h in [4, 24, 48, 168, 336]:
        kw = dict(window=f'{h}h', closed='left', min_periods=h*12)
        roll = r.rolling(**kw)
        net = roll.sum(); absolute = r.abs().rolling(**kw).sum()
        square = r.pow(2).rolling(**kw).sum(); rv = np.sqrt(square)
        up = r.clip(lower=0).pow(2).rolling(**kw).sum()
        peak = r.abs().rolling(**kw).max()
        for name, series, formula in [
            ('rv',rv,'sqrt(sum(r^2))'), ('net',net,'sum(r)'),
            ('abs_net',net.abs(),'abs(sum(r))'), ('abs_sum',absolute,'sum(abs(r))'),
            ('efficiency',net.abs()/absolute.clip(lower=1e-12),'abs(sum(r))/sum(abs(r))'),
            ('up_variance_share',up/square.clip(lower=1e-12),'sum(max(r,0)^2)/sum(r^2)'),
            ('jump_share',peak.pow(2)/square.clip(lower=1e-12),'max(abs(r))^2/sum(r^2)'),
            ('rms',rv/np.sqrt(h*12),'sqrt(mean(r^2))'),
            ('positive_share',r.gt(0).astype(float).where(r.notna()).rolling(**kw).mean(),'mean(r>0)'),
        ]:
            add(f'{name}_{h}h',series,f'{formula}; prior {h}h; closed=left; {h*12} observations')
    for h in [4,24,48]:
        add(f'rv_ratio_{h}_168',v[f'NET4__rv_{h}h']/np.sqrt(h)/(v['NET4__rv_168h']/np.sqrt(168)).clip(lower=1e-9),
            f'(rv_{h}h/sqrt({h}))/(rv_168h/sqrt(168))')
    return pd.DataFrame(v).reindex(origins).replace([np.inf,-np.inf],np.nan), definitions


class GRURegressor:
    """Small MAE-trained GRU; torch imported only when this candidate is fitted."""
    def __init__(self):
        self.state = None

    def network(self, inputs):
        import torch
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru=torch.nn.GRU(inputs,16,batch_first=True)
                self.head=torch.nn.Linear(16,1)
            def forward(self,z):
                return self.head(self.gru(z)[0][:,-1]).squeeze(-1)
        return Net()

    def fit(self,x,y):
        import torch
        torch.manual_seed(48)
        self.inputs=x.shape[-1]; net=self.network(self.inputs)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
        xx=torch.tensor(x,dtype=torch.float32); yy=torch.tensor(y,dtype=torch.float32)
        net.train()
        for epoch in range(24):
            for start in range(0,len(xx),256):
                optimizer.zero_grad()
                loss=torch.nn.functional.l1_loss(net(xx[start:start+256]),yy[start:start+256])
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
        self.state={k:v.detach().clone() for k,v in net.state_dict().items()}
        return self

    def predict(self,x):
        import torch
        net=self.network(self.inputs);net.load_state_dict(self.state);net.eval()
        with torch.no_grad():
            return np.concatenate([net(torch.tensor(x[i:i+512],dtype=torch.float32)).numpy() for i in range(0,len(x),512)])


def load(args):
    # Existing read-only loaders retain the established source/availability masks.
    check(args.week2_code.is_dir(), 'Missing Week 2 code directory')
    sys.path.insert(0,str(args.week2_code))
    import fit_excursions_48h as prior
    # Reuse only the feature timeline, never the 48h outcome-eligibility mask.
    timeline, short, _, _, ref = prior.e2.old.load()
    x=timeline
    parts = sorted((args.canonical_root/'y/horizon_minutes=240/symbol=BTCUSDT').rglob('*.parquet'))
    check(bool(parts), 'No canonical 4h Y partitions')
    manifest = args.canonical_root/'OUTPUT_FILES.json'
    receipts = {row['path']: row['sha256'] for row in json.loads(manifest.read_text())}
    for path in parts:
        rel = path.relative_to(args.canonical_root).as_posix()
        check(rel in receipts and sha(path)==receipts[rel], f'Canonical Y checksum mismatch: {rel}')
    with duckdb.connect() as con:
        con.register('origins',pd.DataFrame({'decision_time':x.index}))
        raw = con.execute('''SELECT y.decision_time,y.net_magnitude_log_bps,y.direction,
            y.label_status,y.canonical_symbol,y.horizon_minutes,
            y.up_magnitude_log_bps,y.down_magnitude_log_bps
            FROM read_parquet(?,hive_partitioning=false) y
            JOIN origins k ON y.decision_time=k.decision_time''',[[str(p) for p in parts]]).fetchdf()
    raw['decision_time']=pd.to_datetime(raw.decision_time,utc=True).dt.as_unit('ns')
    check(raw.decision_time.is_unique,'Duplicate canonical Y keys')
    raw=raw.set_index('decision_time').reindex(x.index)
    valid=(raw.label_status.eq('VALID') & raw.canonical_symbol.eq('BTCUSDT')
           & pd.to_numeric(raw.horizon_minutes).eq(240)
           & np.isfinite(raw[[TARGET,'up_magnitude_log_bps','down_magnitude_log_bps']]).all(axis=1)
           & raw[[TARGET,'up_magnitude_log_bps','down_magnitude_log_bps']].ge(0).all(axis=1))
    crossing=(x.index<prior.TRANSITION)&(x.index+pd.Timedelta(hours=4)>=prior.TRANSITION)
    reason=pd.Series('included',index=x.index)
    reason.loc[~valid]='missing_or_invalid_label'
    reason.loc[valid & crossing]='source_binding_crossing'
    population=pd.DataFrame({'status':reason,'original_evaluation':x.index.isin(ref.index)})
    original_rows=len(x)
    x=x.loc[valid & ~crossing]; raw=raw.loc[x.index]
    check(len(x)>0,'No eligible 4h labels')
    dev=ref.index.intersection(x.index)
    check(raw.direction.isin([-1,0,1]).all() and raw.direction.eq(0).eq(raw[TARGET].eq(0)).all(), 'Invalid direction/magnitude relation')
    signed=raw.direction*raw[TARGET]
    check(((signed>=-raw.down_magnitude_log_bps-1e-7)&(signed<=raw.up_magnitude_log_bps+1e-7)).all(),'Endpoint outside excursion bounds')
    files=parts+[manifest]
    audit=dict(horizon_minutes=240,original_rows=original_rows,valid_rows=len(x),
               original_evaluation_rows=len(ref),evaluation_rows=len(dev),
               exclusions=reason.value_counts().to_dict(),
               label_policy='Canonical 240-minute VALID finite nonnegative labels; exclude source-binding crossing over 4h; no 48h eligibility filter.',
               access='Inherited exposed playground origin grid and masked features; not full canonical minute population.')
    returns_path=prior.e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=prior.e2.price.canonical.load_returns(returns_path)
    # Build the full sequence timeline BEFORE outcome eligibility filtering.
    # Missing snapshots must not reveal whether their future labels were valid.
    old_features=short.join(prior.e2.long_features(r,timeline.index))
    x_timeline=timeline
    new, definitions=engineered(r,x_timeline.index)
    cut=x.index[len(x)//2]; keys=x.index[x.index<=cut][-10:]
    pd.testing.assert_frame_equal(engineered(r,keys)[0],engineered(r[r.index<=cut],keys)[0])
    perturbed=r.copy();perturbed.loc[perturbed.index>=cut]=1e9
    pd.testing.assert_frame_equal(engineered(r,keys)[0],engineered(perturbed,keys)[0])
    design=x_timeline.join(old_features.loc[:,~old_features.columns.isin(x_timeline.columns)]).join(new)
    check(design.columns.is_unique and design.index.is_unique,'Duplicate feature/index')
    packs={'market':list(x),'prior_price':[c for c in old_features if c not in x],
           'new_price':list(new),'all':list(design)}
    packs['market_new']=list(x)+list(new)
    packs={k:v for k,v in packs.items() if v}
    y=pd.DataFrame({TARGET:raw[TARGET],'label_end':x.index+pd.Timedelta(hours=4)},index=x.index)
    files += [returns_path, args.canonical_root/'team_tools/build_dataset.py']
    # Hash every imported workspace source, including transitive loader dependencies.
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).suffix=='.py' and 'Desktop\\ML\\' in str(Path(p)):
            files.append(Path(p))
    files += list((prior.e2.old.data.DATA).glob('X_*.parquet'))
    audit.update(target=TARGET,formula=FORMULA,features=len(design.columns),packs={k:len(v) for k,v in packs.items()},
                 target_min=float(raw[TARGET].min()),target_max=float(raw[TARGET].max()),new_feature_causality='prefix and future perturbation passed')
    return design,y,dev,population,packs,definitions,list(set(files)),audit


def prepare(x, keys, cols):
    a=x.loc[keys,cols].replace([np.inf,-np.inf],np.nan)
    cols=[c for c in cols if a[c].notna().mean()>=.8 and a[c].nunique()>1]
    check(bool(cols),'No eligible features')
    a=a[cols]; med=a.median(); scale=(a.quantile(.75)-a.quantile(.25)).clip(lower=1e-6)
    return dict(cols=cols,median=med,scale=scale)


def transform(x, pp):
    a=x[pp['cols']].replace([np.inf,-np.inf],np.nan)
    return np.column_stack([((a.fillna(pp['median'])-pp['median'])/pp['scale']).clip(-8,8),a.isna().astype(float)])


def model_design(x,keys,pp,kind):
    if kind!='gru':
        return transform(x.reindex(keys),pp)
    snapshots=[]
    for h in [12,8,4,0]:
        snapshots.append(transform(x.reindex(keys-pd.Timedelta(hours=h)),pp))
    return np.stack(snapshots,axis=1)


def fit(x,y,cut,spec,packs,cap):
    keys=y.index[y.label_end<cut]; check(len(keys)>=100,'Insufficient mature training rows')
    pp=prepare(x,keys,packs[spec['pack']]); sample=keys[np.unique(np.linspace(0,len(keys)-1,min(cap,len(keys))).astype(int))]
    a=model_design(x,sample,pp,spec['model']); target=y.loc[sample,TARGET].to_numpy()
    center=float(y.loc[keys,TARGET].median()); scale=max(float(y.loc[keys,TARGET].std()),1.)
    if spec['model']=='hgb':
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=60,max_leaf_nodes=7,max_depth=3,min_samples_leaf=64,l2_regularization=5,early_stopping=False,random_state=48)
    elif spec['model']=='extra':
        model=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=32,max_features=.8,max_depth=10,n_jobs=1,random_state=48)
    elif spec['model']=='gru':
        model=GRURegressor()
    else:
        model=MLPRegressor(hidden_layer_sizes=(16,),activation='relu',alpha=1.,batch_size=min(256,len(sample)),learning_rate_init=.001,max_iter=40,early_stopping=False,shuffle=False,random_state=48)
    model.fit(a,(target-center)/scale)
    return dict(prep=pp,model=model,center=center,scale=scale,spec=spec,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())


def predict(a,x,keys=None):
    keys=x.index if keys is None else keys
    return np.maximum(0,a['model'].predict(model_design(x,keys,a['prep'],a['spec']['model']))*a['scale']+a['center'])


def metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    return dict(rows=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()))


def gate(errors,baseline):
    ma=np.array([np.mean(np.abs(e)) for e in errors]); ba=np.array([np.mean(np.abs(e)) for e in baseline])
    pooled=np.mean(np.abs(np.concatenate(errors))); ref=np.mean(np.abs(np.concatenate(baseline)))
    return bool(pooled<=.999*ref and (ma<=ba).sum()>=2 and (ma<=1.02*ba).all())


def run(x,y,dev,packs,out,report,models,cap,hgb_only=False):
    pure_model=models[0] if hgb_only else None
    check(not hgb_only or len(models)==1,'Pure mode requires exactly one model family')
    specs=[dict(pack=p,model=m) for p in packs for m in models]
    ledger=[]; picks=[]; frames=[]
    for issue in ISSUES:
        windows=[]
        for offset in [9,6,3]:
            cut=issue-pd.DateOffset(months=offset)
            va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&(y.label_end<issue)]
            check(len(va)>=100,'Too few inner validation rows')
            windows.append((cut,va))
        baseline=[np.full(len(va),y.loc[y.label_end<cut,TARGET].median())-y.loc[va,TARGET].to_numpy() for cut,va in windows]
        scores=[]; accepts=[]
        for i,spec in enumerate(specs):
            errors=[]
            for j,(cut,va) in enumerate(windows):
                a=fit(x,y,cut,spec,packs,cap); check(a['max_label_end']<va.min(),'Training purge failed')
                e=predict(a,x,va)-y.loc[va,TARGET].to_numpy();errors.append(e)
                ledger.append(dict(issue=issue,candidate=i,window=j,cut=cut,spec=spec,max_training_label_end=a['max_label_end'],**metrics(np.zeros(len(e)),e)))
            scores.append(float(np.abs(np.concatenate(errors)).mean()));accepts.append(gate(errors,baseline))
            print(issue.date(),i+1,'/',len(specs),spec,'validation MAE',round(scores[-1],4),flush=True)
            pd.DataFrame(ledger).to_csv(out/'validation_trials.csv',index=False)
        best=min(range(len(specs)),key=lambda i:(scores[i],i))
        accepted=[i for i in range(len(specs)) if accepts[i]]
        chosen=min(accepted,key=lambda i:(scores[i],i)) if accepted else None
        end=ISSUES[list(ISSUES).index(issue)+1] if issue!=ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
        te=dev[(dev>=issue)&(dev<end)]; check(len(te)>0,'Empty outer evaluation')
        tr=y.index[y.label_end<issue]
        f=pd.DataFrame({'actual':y.loc[te,TARGET],'issue':issue,'median':y.loc[tr,TARGET].median(),'mean':y.loc[tr,TARGET].mean()},index=te)
        fitted={}
        for i in set([best]+([] if chosen is None else [chosen])):
            a=fit(x,y,issue,specs[i],packs,cap);path=out/f'model_{issue:%Y%m}_{i}.joblib';joblib.dump(a,path,compress=3)
            p=predict(a,x,te);np.testing.assert_allclose(p,predict(joblib.load(path),x,te),atol=1e-8,rtol=0)
            fitted[i]=p
            picks.append(dict(issue=issue,candidate=i,spec=specs[i],features=a['prep']['cols'],validation_mae=scores[i],accepted=accepts[i],artifact=path.name,sha256=sha(path),training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end']))
        f['candidate']=fitted[best]; f['selected']=f['median'] if chosen is None else fitted[chosen]
        f['selected_candidate']=-1 if chosen is None else chosen
        if hgb_only:
            # Pure HGB policy: earlier validation chooses only its feature pack.
            # Preserve the guarded result for audit, but never substitute it.
            f['guarded_selected']=f['selected']
            f[pure_model]=fitted[best]
            f['selected']=f[pure_model]
            f['selected_candidate']=best
        frames.append(f);dump(out/'selections.json',picks)
    result=pd.concat(frames).sort_index();check(result.index.equals(dev.sort_values()),'Outer coverage mismatch')
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    rows=[]
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for model in (['median','mean',pure_model,'guarded_selected'] if hgb_only else ['median','mean','selected','candidate']):
            rows.append(dict(period=period,model=model,**metrics(g.actual,g[model])))
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False)
    policy=(f'Pure {pure_model} across all periods; earlier validation selects feature pack. No baseline fallback in primary forecast. Guarded result shown separately; gate acceptance is not implied.\n' if hgb_only else 'Diagnostic candidate is separate from validation-selected policy.\n')
    text='4h net magnitude (absolute endpoint log bps)\n'+FORMULA+'\n'+policy+'\n'+table.to_string(index=False)+'\n\nHistorical exposed evaluation; no independent confirmation.\n'
    (report/'SUMMARY.txt').write_text(text,encoding='utf-8');table.to_csv(report/'metrics.csv',index=False)
    dump(report/'feature_selection.json',picks)
    dump(out/'COMPLETE.json',dict(status='COMPLETE',target=TARGET,rows=len(result),prediction_sha256=sha(out/'predictions.csv'),replay_passed=True))
    print(text,'\nOutputs:',out,'\nReport:',report,flush=True)


def self_test():
    rng=np.random.default_rng(48)
    ri=pd.date_range('2020-01-01',periods=5000,freq='5min',tz='UTC');r=pd.Series(rng.normal(size=len(ri)),index=ri)
    keys=ri[-10:]; a,_=engineered(r,keys);changed=r.copy();changed.loc[keys[-1]:]=1e9
    pd.testing.assert_frame_equal(a,engineered(changed,keys)[0])
    np.testing.assert_allclose(a['NET4__rv_4h'].iloc[-1],np.sqrt(r[(r.index>=keys[-1]-pd.Timedelta(hours=4))&(r.index<keys[-1])].pow(2).sum()))
    idx=pd.date_range('2020-01-01',periods=900,freq='12h',tz='UTC');x=pd.DataFrame(rng.normal(size=(900,3)),index=idx,columns=list('abc'));x.iloc[::9,0]=np.nan
    y=pd.DataFrame({TARGET:100+10*x.b,'label_end':idx+pd.Timedelta(hours=4)},index=idx);cut=idx[700]
    with tempfile.TemporaryDirectory() as tmp:
        for model in ['hgb','extra','mlp','gru']:
            spec=dict(model=model,pack='all');a=fit(x,y,cut,spec,{'all':list(x)},1000)
            assert a['max_label_end']<cut
            xx=x.copy();xx.loc[cut:]=1e8; yy=y.copy();yy.loc[cut:,TARGET]=1e9
            b=fit(xx,yy,cut,spec,{'all':list(x)},1000)
            np.testing.assert_allclose(predict(a,x),predict(b,x))
            path=Path(tmp)/'model.joblib';joblib.dump(a,path);np.testing.assert_allclose(predict(a,x),predict(joblib.load(path),x))
    assert gate([np.ones(5)*.5]*3,[np.ones(5)]*3)
    assert not gate([np.ones(5)*2]*3,[np.ones(5)]*3)
    print('PASS: feature direct formula/future isolation, 4h purge, training-only preprocessing and target isolation, HGB/ExtraTrees/MLP fits, serialization replay, gates.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--week2-code',type=Path,default=ROOT.parent/'Week 2/Code')
    p.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--models',nargs='+',choices=['hgb','extra','mlp','gru'],default=['hgb','extra','mlp','gru'])
    p.add_argument('--hgb-only',action='store_true',help='HGB throughout; earlier validation chooses feature pack; no baseline fallback in primary forecast')
    p.add_argument('--pure-model',choices=['hgb','extra','mlp','gru'],help='Use this model family throughout all evaluation periods, without baseline fallback')
    p.add_argument('--row-cap',type=int,default=6000)
    p.add_argument('--check-inputs',action='store_true');p.add_argument('--self-test',action='store_true')
    args=p.parse_args();check(args.threads>0 and args.row_cap>=256,'Invalid thread/cap setting')
    if args.hgb_only:
        check(args.pure_model in [None,'hgb'],'Conflicting pure-model options')
        args.pure_model='hgb'
    if args.pure_model:
        args.models=[args.pure_model]
    if args.hgb_only:
        args.models=['hgb']
    if 'gru' in args.models:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError as exc:
            raise ImportError('GRU requires PyTorch. Install torch or use --models hgb extra mlp.') from exc
    with threadpool_limits(limits=args.threads):
        if args.self_test:self_test();return
        x,y,dev,population,packs,definitions,files,audit=load(args)
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));print('PASS: actual input checks; no model fitting.');return
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        name=('net_magnitude_4h_'+args.pure_model+'_only') if args.pure_model else 'net_magnitude_4h'
        out=HERE/'experiments'/(name+'_v1')/stamp;out.mkdir(parents=True)
        report=ROOT/'Report'/name/stamp;report.mkdir(parents=True)
        framework=ROOT.parents[1]/'ml_framework'
        files += [Path(__file__),ROOT/'PROJECT_CONTEXT.md']+[framework/v for v in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        dump(out/'CONTRACT.json',dict(target=TARGET,horizon_minutes=240,formula=FORMULA,arguments=vars(args),input_audit=audit,packs=packs,
            primary_policy=(f'{args.pure_model} every period, best pooled earlier-validation MAE feature pack; no baseline fallback. Gates reported separately.' if args.pure_model else 'Validation-gated model with median fallback.'),
            input_hashes={str(f):sha(f) for f in sorted(set(files))},seed=48,
            versions={v:importlib.metadata.version(v) for v in ['numpy','pandas','scikit-learn','duckdb','joblib']+(['torch'] if 'gru' in args.models else [])},
            selection='Three earlier 2-month windows offset 9/6/3 months; label_end strictly before fit cutoff; pooled MAE ranking; selected needs >=0.1% median gain, >=2/3 window wins/ties, no >2% window loss; otherwise median. Candidate separate.',
            models='HGB L1:60 iterations/7 leaves/depth3; ExtraTrees L2:48 trees/depth10; MLP L2:16 ReLU/40 epochs; GRU L1:16 hidden/24 epochs/AdamW lr.001 decay.03/gradient clip1/four snapshots12,8,4,0h. Full mature preprocessing and baselines; evenly spaced estimator row cap only.',
            limitations='Exposed historical 2022-2025 data; inherited source availability masks. No guaranteed gain, independent confirmation or exhaustive feature search. No resume.'))
        population.to_csv(out/'population_audit.csv',index_label='decision_time')
        pd.DataFrame(definitions).to_csv(out/'new_feature_formulas.csv',index=False)
        (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
        run(x,y,dev,packs,out,report,args.models,args.row_cap,bool(args.pure_model))


if __name__=='__main__':
    main()
