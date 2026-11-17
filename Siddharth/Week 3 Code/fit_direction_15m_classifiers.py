"""15m direction classification: direction features,trees/neural models,logloss+Brier."""
from pathlib import Path
import argparse,json,hashlib,sys,importlib.metadata
import numpy as np,pandas as pd,duckdb,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import direction_15m_features as features
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'Code'
ISSUES=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
KINDS=['logistic','hgb','random_forest','extra_trees','mlp_logloss','mlp_brier','gru_logloss','gru_brier']
WEIGHTS=[.25,.5,.75,1.]
TRANSITION=pd.Timestamp('2024-10-29T00:01Z')
def check(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def score(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    check(len(y)>0 and np.isfinite(p).all() and np.isin(y,[0,1]).all(),'Invalid probability/label')
    return dict(rows=len(y),log_loss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),brier=float(((p-y)**2).mean()),accuracy=float(((p>=.5)==y).mean()),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,mean_p_up=float(p.mean()),actual_up_rate=float(y.mean()))
def load(args):
    template=json.loads(args.direction_template.read_text(encoding='utf-8-sig'))['models'][0]
    inventory=list(template['preprocessing']['fields']);core=template['base_model']['config']['features']
    check(len(inventory)==384 and set(core)<=set(inventory),'Unexpected direction inventory')
    paths=[args.data_dir/'X_train.parquet',args.data_dir/'X_development.parquet']
    projection=','.join('"'+c+'"' for c in ['decision_time']+inventory)
    with duckdb.connect() as con:
        frames=[con.execute('SELECT '+projection+' FROM read_parquet(?)',[str(p)]).fetchdf() for p in paths]
    for f in frames:f['decision_time']=pd.to_datetime(f.decision_time,utc=True).dt.as_unit('ns')
    dev=pd.DatetimeIndex(frames[1].decision_time).sort_values()
    x=pd.concat(frames).set_index('decision_time').sort_index().replace([np.inf,-np.inf],np.nan)
    check(x.index.is_unique and not x.index.hasnans,'Duplicate/missing X keys')
    parts=sorted((args.canonical_root/'y/horizon_minutes=15/symbol=BTCUSDT').rglob('*.parquet'));check(bool(parts),'Missing canonical15m labels')
    receipts={r['path']:r['sha256'] for r in json.loads(args.canonical_manifest.read_text(encoding='utf-8-sig'))}
    for p in parts:check(sha(p)==receipts.get(p.relative_to(args.canonical_root).as_posix()),'Canonical checksum mismatch '+str(p))
    with duckdb.connect() as con:
        con.register('origins',pd.DataFrame({'decision_time':x.index}))
        raw=con.execute('SELECT y.decision_time,y.direction,y.net_magnitude_log_bps,y.label_status,y.canonical_symbol,y.horizon_minutes FROM read_parquet(?,hive_partitioning=false) y JOIN origins k ON y.decision_time=k.decision_time',[[str(p) for p in parts]]).fetchdf()
    raw['decision_time']=pd.to_datetime(raw.decision_time,utc=True).dt.as_unit('ns');check(raw.decision_time.is_unique,'Duplicate canonical labels')
    raw=raw.set_index('decision_time').reindex(x.index)
    valid=raw.label_status.eq('VALID')&raw.canonical_symbol.eq('BTCUSDT')&pd.to_numeric(raw.horizon_minutes).eq(15)&raw.direction.isin([-1,0,1])&np.isfinite(raw.net_magnitude_log_bps)&raw.net_magnitude_log_bps.ge(0)
    crossing=(x.index<TRANSITION)&(x.index+pd.Timedelta(minutes=15)>=TRANSITION)
    reason=pd.Series('included',index=x.index);reason.loc[~valid]='missing_invalid_label';reason.loc[valid&crossing]='source_binding_crossing'
    keys=x.index[valid&~crossing];check(len(keys)>0,'No eligible labels')
    check(raw.loc[keys,'direction'].eq(0).equals(raw.loc[keys,'net_magnitude_log_bps'].eq(0)),'Invalid zero relation')
    signed=raw.loc[keys,'direction']*raw.loc[keys,'net_magnitude_log_bps']
    y=pd.DataFrame(dict(y_up=signed.gt(0).astype(int),signed_return_log_bps=signed,label_end=keys+pd.Timedelta(minutes=15)),index=keys)
    fresh,definitions=features.build(x,inventory)
    cut=x.index[len(x)//2];k=x.index[x.index<=cut][-10:]
    xx=x.copy();xx.loc[xx.index>=cut]=1e9
    pd.testing.assert_frame_equal(fresh.loc[k],features.build(xx,inventory)[0].loc[k])
    pd.testing.assert_frame_equal(fresh.loc[k],features.build(x.loc[:cut],inventory)[0].loc[k])
    design=x.join(fresh)
    packs={'direction_core':list(core),'direction_original':inventory,'direction_new_state':list(core)+list(fresh)}
    audit=dict(horizon_minutes=15,target='canonical direction>0; zero is non-UP',original_rows=len(x),eligible_rows=len(y),evaluation_rows=len(dev.intersection(keys)),exclusions=reason.value_counts().to_dict(),packs={k:len(v) for k,v in packs.items()},inventory_families=pd.Series([c.split('__')[1] for c in inventory]).value_counts().to_dict(),feature_causality='prefix and future perturbation passed',limitations='Historical384 direction inventory selected previously for48h; reused for15m; exposed historical evaluation. Original availability masks inherited. No magnitude inputs used.')
    files=paths+parts+[args.canonical_manifest,args.direction_template,Path(__file__),Path(features.__file__),ROOT/'PROJECT_CONTEXT.md']
    return design,y,dev.intersection(keys),packs,definitions,reason,files,audit

def prepare(x,keys,cols):
    a=x.loc[keys,cols];keep=[c for c in cols if a[c].notna().mean()>=.8 and a[c].nunique()>1]
    check(bool(keep),'No eligible features');a=a[keep];med=a.median();iqr=(a.quantile(.75)-a.quantile(.25)).clip(lower=1e-6)
    return dict(cols=keep,median=med,iqr=iqr)
def design(x,keys,pp,kind):
    def transform(k):
        a=x.reindex(k)[pp['cols']];z=((a.fillna(pp['median'])-pp['median'])/pp['iqr']).clip(-8,8)
        return np.column_stack([z,a.isna().astype(float)]).astype('float32')
    return np.stack([transform(keys-pd.Timedelta(minutes=m)) for m in [90,60,30,0]],axis=1) if kind.startswith('gru') else transform(keys)
class NeuralClassifier:
    def __init__(self,kind,prior):self.kind=kind;self.prior=prior
    def network(self,inputs):
        import torch
        if self.kind.startswith('gru'):
            class Net(torch.nn.Module):
                def __init__(self):super().__init__();self.gru=torch.nn.GRU(inputs,8,batch_first=True);self.head=torch.nn.Linear(8,1)
                def forward(self,z):return self.head(self.gru(z)[0][:,-1]).squeeze(-1)
            net=Net();head=net.head
        else:net=torch.nn.Sequential(torch.nn.Linear(inputs,16),torch.nn.ReLU(),torch.nn.Linear(16,1),torch.nn.Flatten(start_dim=0));head=net[2]
        with torch.no_grad():head.weight.zero_();head.bias.fill_(float(np.log(self.prior/(1-self.prior))))
        return net
    def fit(self,x,y):
        import torch
        torch.manual_seed(48);torch.use_deterministic_algorithms(True);self.inputs=x.shape[-1];net=self.network(self.inputs)
        xx=torch.tensor(x);yy=torch.tensor(y,dtype=torch.float32);opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.1);g=torch.Generator().manual_seed(48)
        for epoch in range(16):
            for batch in torch.randperm(len(y),generator=g).split(256):
                opt.zero_grad(set_to_none=True);logits=net(xx[batch])
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yy[batch]) if self.kind.endswith('logloss') else ((torch.sigmoid(logits)-yy[batch])**2).mean()
                check(bool(torch.isfinite(loss)),'Nonfinite classifier loss');loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        self.state={k:v.detach().clone() for k,v in net.state_dict().items()};return self
    def predict_proba(self,x):
        import torch
        net=self.network(self.inputs);net.load_state_dict(self.state);net.eval()
        with torch.no_grad():p=np.concatenate([torch.sigmoid(net(torch.tensor(x[i:i+512]))).numpy() for i in range(0,len(x),512)])
        return np.column_stack([1-p,p])
def fit(x,y,cut,spec,packs,cap):
    keys=y.index[y.label_end<cut];check(len(keys)>=1000 and y.loc[keys,'y_up'].nunique()==2,'Insufficient mature binary training')
    pp=prepare(x,keys,packs[spec['pack']]);sample=keys[np.unique(np.linspace(0,len(keys)-1,min(cap,len(keys))).astype(int))]
    target=y.loc[sample,'y_up'].to_numpy();prior=float(np.clip(y.loc[keys,'y_up'].mean(),1e-7,1-1e-7));kind=spec['model']
    check(len(np.unique(target))==2,'Training sample lacks class')
    if kind=='logistic':model=LogisticRegression(C=.01,max_iter=300,random_state=48)
    elif kind=='hgb':model=HistGradientBoostingClassifier(loss='log_loss',max_iter=40,max_depth=3,max_leaf_nodes=7,min_samples_leaf=64,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=48)
    elif kind in ['random_forest','extra_trees']:
        cls=RandomForestClassifier if kind=='random_forest' else ExtraTreesClassifier
        model=cls(n_estimators=32,criterion='gini' if kind=='random_forest' else 'log_loss',max_depth=5,min_samples_leaf=32,max_features=.75,bootstrap=True,max_samples=min(1024,len(sample)),n_jobs=1,random_state=48)
    else:model=NeuralClassifier(kind,prior)
    model.fit(design(x,sample,pp,kind),target)
    return dict(model=model,prep=pp,spec=spec,prior=prior,training_rows=len(keys),estimator_rows=len(sample),max_label_end=y.loc[keys,'label_end'].max())
def predict(a,x,keys):return np.clip(a['model'].predict_proba(design(x,keys,a['prep'],a['spec']['model']))[:,1],1e-7,1-1e-7)
def assess(ys,ps,priors):
    metrics=[score(y,p) for y,p in zip(ys,ps)];baseline=[score(y,np.full(len(y),q)) for y,q in zip(ys,priors)]
    pooled=score(np.concatenate(ys),np.concatenate(ps));bs=score(np.concatenate(ys),np.concatenate([np.full(len(y),q) for y,q in zip(ys,priors)]))
    ratios=[pooled[k]/bs[k] for k in ['log_loss','brier']]
    accepted=all(pooled[k]<=.999*bs[k] and sum(m[k]<=b[k] for m,b in zip(metrics,baseline))>=3 and all(m[k]<=1.02*b[k] for m,b in zip(metrics,baseline)) for k in ['log_loss','brier'])
    return dict(score=sum(ratios)/2,accepted=bool(accepted),window_metrics=metrics,**pooled)
def choose(trials):
    key=lambda t:(t['score'],t['id'],t['weight']);best=min(trials,key=key);ok=[t for t in trials if t['accepted']]
    return best,min(ok,key=key) if ok else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=KINDS,default=KINDS)
    p.add_argument('--data-dir',type=Path,default=ROOT.parent/'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data')
    p.add_argument('--direction-template',type=Path,default=ROOT.parent/'Week 2/Code/neural_finetune_v45/neural_finetune_v45/final_pipeline.json')
    p.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    p.add_argument('--canonical-manifest',type=Path,default=ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json')
    p.add_argument('--row-cap',type=int,default=2000);p.add_argument('--threads',type=int,default=2);p.add_argument('--check-inputs',action='store_true')
    args=p.parse_args();check(args.row_cap>=1000 and args.threads>0,'Invalid cap/threads');check(len(set(args.models))==len(args.models),'Duplicate models')
    import torch;torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        x,y,dev,packs,defs,population,files,audit=load(args)
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));return
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');out=HERE/'experiments/direction_15m_classifiers_v1'/stamp;report=ROOT/'Report/direction_15m_classifiers_v1'/stamp;out.mkdir(parents=True);report.mkdir(parents=True)
        specs=[dict(model=m,pack=k) for m in args.models for k in packs]
        dump(out/'CONTRACT.json',dict(target='15m endpoint UP=direction>0;zero grouped with non-UP',arguments=vars(args),input_audit=audit,packs=packs,specs=specs,weights=WEIGHTS,selection='Rank average relative pooled logloss/Brier versus mature training prevalence;require.1%gain both,3/4windowwins/ties,no>2%loss;else prevalencefallback. Four earlier3monthwindows15/12/9/6months,strict15mmaturity.',neural='MLP16orGRU8,16epochs,AdamWlr.0005decay.1,zero finalweights/priorlogitbias;separate BCE/Brier objectives. No classweighting/oversampling.',new_features='Past-only seven direction-state sources and interactions;no magnitude model predictions or magnitude-derived feature packs.',input_hashes={str(f):sha(f) for f in files},versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','torch','duckdb']},limitations='Historically selected48h384featureinventory reused for15m,exposed evaluation;no guaranteed win,untouched test or resume. No probability calibration using outer outcomes.'))
        defs.to_csv(out/'new_direction_feature_formulas.csv',index=False);population.to_csv(out/'population_audit.csv',index_label='decision_time')
        frames=[];ledger=[];plans=[];ablations=[]
        for issue in ISSUES:
            windows=[]
            for offset in [15,12,9,6]:
                cut=issue-pd.DateOffset(months=offset);end=cut+pd.DateOffset(months=3)
                va=y.index[(y.index>=cut)&(y.index<end)&(y.label_end<end)&(y.label_end<issue)]
                check(len(va)>=100,'Too few validation labels');windows.append((cut,va))
            ys=[y.loc[va,'y_up'].to_numpy() for cut,va in windows];priors=[float(y.loc[y.label_end<cut,'y_up'].mean()) for cut,va in windows];trials=[]
            for i,spec in enumerate(specs):
                ps=[]
                for j,(cut,va) in enumerate(windows):
                    print(issue.date(),i+1,'/',len(specs),spec,'window',j+1,flush=True)
                    a=fit(x,y,cut,spec,packs,args.row_cap);check(a['max_label_end']<va.min(),'Purge failed');ps.append(predict(a,x,va))
                for w in WEIGHTS:
                    record=dict(issue=str(issue),id=i,spec=spec,weight=w,**assess(ys,[w*q+(1-w)*b for q,b in zip(ps,priors)],priors));trials.append(record);ledger.append(record)
                dump(out/'validation_trials.json',ledger)
            best,chosen=choose(trials);family_choices={m:choose([t for t in trials if t['spec']['model']==m])[1] for m in args.models}
            # Record feature-group comparison on validation only.
            for m in args.models:
                row=dict(issue=str(issue),model=m)
                for k in packs:
                    t=min([t for t in trials if t['spec']==dict(model=m,pack=k)],key=lambda t:t['score']);row[k+'_log_loss']=t['log_loss'];row[k+'_brier']=t['brier']
                ablations.append(row)
            plan=dict(issue=str(issue),candidate=best,selected=chosen,family_selected=family_choices,models=[]);plans.append(plan);dump(out/'selections.json',plans)
            end=ISSUES[list(ISSUES).index(issue)+1] if issue!=ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1);te=dev[(dev>=issue)&(dev<end)];check(len(te)>0,'Empty outer period')
            prior=float(y.loc[y.label_end<issue,'y_up'].mean());f=pd.DataFrame(dict(y_up=y.loc[te,'y_up'],signed_return_log_bps=y.loc[te,'signed_return_log_bps'],issue=issue,prevalence=prior,fair=.5),index=te)
            records=[best]+([] if chosen is None else [chosen])+[t for t in family_choices.values() if t is not None];cache={}
            for t in records:
                i=t['id']
                if i not in cache:
                    a=fit(x,y,issue,specs[i],packs,args.row_cap);path=out/f'model_{issue:%Y%m}_{i}.joblib';joblib.dump(a,path,compress=3);q=predict(a,x,te);np.testing.assert_allclose(q,predict(joblib.load(path),x,te),rtol=0,atol=1e-8);cache[i]=q
                    plan['models'].append(dict(artifact=path.name,sha256=sha(path),spec=specs[i],features=a['prep']['cols'],training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end']))
            f['candidate']=best['weight']*cache[best['id']]+(1-best['weight'])*prior
            f['p_up']=prior if chosen is None else chosen['weight']*cache[chosen['id']]+(1-chosen['weight'])*prior;f['p_down']=1-f.p_up
            f['predicted_direction']=np.where(f.p_up>=.5,1,-1);f['actual_direction']=np.sign(f.signed_return_log_bps).astype(int)
            for m,t in family_choices.items():f[m]=prior if t is None else t['weight']*cache[t['id']]+(1-t['weight'])*prior
            frames.append(f);dump(out/'selections.json',plans);pd.DataFrame(ablations).to_csv(out/'feature_pack_comparison.csv',index=False)
        result=pd.concat(frames).sort_index();check(result.index.equals(dev.sort_values()),'Outer coverage mismatch');result.to_csv(out/'predictions.csv',index_label='decision_time')
        rows=[]
        for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
            for m in ['prevalence','fair','p_up','candidate']+args.models:rows.append(dict(period=period,model=m,**score(g.y_up,g[m])))
        metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False);metrics.to_csv(report/'metrics.csv',index=False)
        # Fixed-bin calibration diagnostics,never used for fitting/selection.
        cal=result.assign(bin=pd.cut(result.p_up,bins=np.linspace(0,1,11),include_lowest=True)).groupby('bin',observed=True).agg(rows=('y_up','size'),mean_p_up=('p_up','mean'),actual_up_rate=('y_up','mean'));cal.to_csv(out/'calibration.csv')
        text='15m classification;lower logloss/Brier better. p_up is guarded policy;family columns may fall back to prevalence.\n'+metrics.to_string(index=False);(report/'SUMMARY.txt').write_text(text,encoding='utf-8');dump(report/'selections.json',plans)
        for path in [Path(__file__),Path(features.__file__)]: (out/path.name).write_bytes(path.read_bytes())
        dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=sha(out/'predictions.csv'),replay_passed=True));print(text,'\nOutputs:',out,'\nReport:',report,flush=True)
if __name__=='__main__':main()
