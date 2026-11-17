"""Fixed warm-start fine-tuning of frozen-style neural checkpoints."""
import sys,json,csv,copy,subprocess
from pathlib import Path
HOME=Path(__file__).resolve().parent;HERE=HOME.parent
sys.path.insert(0,str(HERE/'temporal_features_v35'))
import shared as s
import torch
np=s.np;pd=s.pd;b=s.b;h=s.h
PRIOR=3370;CAP=3394;RUN='direction_finetune_v45';MODES=['head','all'];SEEDS=[48,148,248]
def save(n,o):s.save(HOME/n,o)
def load(n):return s.load(HOME/n)
def verify():
    for p,d in load('registration.json')['protected_hashes'].items():assert s.digest(p)==d,p
    assert s.digest(__file__)==load('registration.json')['code_sha256']
def register():
    assert s.count()==PRIOR and not (HOME/'STARTED.json').exists()
    s.save(HERE/'budget_extension_v45.json',dict(version='TEAM_48H_PLAYGROUND_V1_MEMBER03_EXTENSION_35',registered_utc=s.n.now(),prior_fits=PRIOR,maximum_additional_fits=24,member_cumulative_fit_cap=CAP,team_fit_cap=CAP+60,previous_extension_sha256=s.digest(HERE/'budget_extension_v42.json'),authorization='User explicitly requested fine-tuning and neural/deep-learning experiments on the frozen baseline. Bounded exploratory warm-start experiment.'))
    (HERE/'extended_scorer_v45.py').write_text((HERE/'extended_scorer_v42.py').read_text().replace('budget_extension_v42.json','budget_extension_v45.json'))
    protected=[p for folder in [s.CURRENT,h.OLD,HERE/'direction_hybrid_v25',HERE/'direction_group_ablation_v27',HERE/'final_handoff_v41'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    protected += [HERE/'CURRENT_DIRECTION_BASELINE.json',HERE/'SELECTED_SUBMISSION.json']
    save('registration.json',dict(utc=s.n.now(),prior_fits=PRIOR,maximum_new_fits=24,
        recipe='Warm-start each matching chronological frozen-style seed checkpoint at32epochs. Two branches: output-layer-only and all-layer fine-tuning. Each16additional epochs, full-batch AdamW lr0.0003 weight_decay0.01,clip1. Dropout disabled during fine-tuning for stable small adjustments. Preprocessing and logistic base stay fixed.',
        objective='BCE(final blended probability,y)+Brier(final blended probability,y)+mean((new bounded logit correction-original correction)^2). Final probability remains75%logistic+25%corrected logistic. Bound0.15*tanh unchanged.',
        candidate='Fixed equal average of both branches and all three seeds. Individual branches and seeds diagnostic only; no post-outcome branch selection.',
        gate='Both pooled validation scores lower, both lower in>=2/3periods, and>=2/3seed-averaged-branch candidates lower on both pooled metrics. Then6final fits and one development score.',
        stopping='18outerfits plus6conditional final fits. No outcome-driven retries or setting changes.',
        limitation='Exploratory historical data, repeatedly researched. Unresolved upstream normalization and actual publication provenance. Not new causal certification or fresh confirmation.',
        protected_hashes={str(p):s.digest(p) for p in protected},code_sha256=s.digest(__file__)))
    (HOME/'models').mkdir();save('STARTED.json',dict(utc=s.n.now(),prior_fits=s.count()))
    (HOME/'environment-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
def log(tag,mode,seed,status,note=''):
    with (HERE/'experiments.csv').open('a',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow([RUN,s.n.now(),'direction','Warm-start neural fine-tuning',json.dumps(dict(mode=mode,seed=seed,extra_epochs=16)),tag,s.count()+(status=='STARTED'),status,'Fixed recipe; chronological purged evaluation',note])
def predict(X,m):
    previous=h.RUN;h.RUN=HOME/'models'
    try:return .75*b.predict(X,m['base_model'])+.25*h.predict_nn(X,m,48)
    finally:h.RUN=previous
def fit(X,y,ref,tag):
    models=[]
    for mode in MODES:
        for old in ref['models']:
            seed=old['seed'];assert s.count()<CAP
            assert old['training_rows']==len(X)
            expected=b.preprocess_fit(X,dict(features=list(X.columns),interaction=True));assert old['preprocessing']==expected
            item=old['checkpoints']['32'];path=Path(ref['folder'])/item['file'];assert s.digest(path)==item['sha256']
            log(tag,mode,seed,'STARTED')
            try:
                torch.manual_seed(seed);net=h.residual_net(old['input_dim']);net.load_state_dict(torch.load(path,weights_only=True));net.eval()
                initial={k:v.clone() for k,v in net.state_dict().items()}
                if mode=='head':
                    for param in net.parameters():param.requires_grad=False
                    for param in net[-1].parameters():param.requires_grad=True
                A=h.dl.design(X,old['preprocessing']);target=torch.tensor(y.to_numpy(),dtype=torch.float32).reshape(-1,1)
                base=torch.tensor(b.predict(X,old['base_model']),dtype=torch.float32).reshape(-1,1);offset=torch.logit(base.clamp(1e-7,1-1e-7))
                with torch.no_grad():original=.15*torch.tanh(net(A))
                opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);losses=[]
                for epoch in range(16):
                    opt.zero_grad();delta=.15*torch.tanh(net(A));prob=.75*base+.25*torch.sigmoid(offset+delta)
                    loss=torch.nn.functional.binary_cross_entropy(prob,target)+(prob-target).square().mean()+(delta-original).square().mean()
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
                if mode=='head':
                    for k,v in net.state_dict().items():
                        if not k.startswith('5.'):assert torch.equal(v,initial[k]),k
                name=f'{tag}_{mode}_s{seed}';dest=HOME/'models'/(name+'.pt');torch.save(net.state_dict(),dest)
                m=copy.deepcopy(old);m.update(name=name,epochs=48,mode=mode,source_checkpoint=str(path),source_sha256=item['sha256'],fine_tuning_losses=losses,checkpoints={'48':dict(file=dest.name,sha256=s.digest(dest))})
                save('models/'+name+'.json',m);models.append(m);log(tag,mode,seed,'COMPLETE');print(name+' complete',flush=True)
            except Exception as e:log(tag,mode,seed,'FAILED',repr(e));raise
    return models
def frame_predictions(X,models,reference):
    frame=pd.DataFrame(dict(current=s.predict_anchor(X,reference)),index=X.index)
    for m in models:frame[f'{m["mode"]}_s{m["seed"]}']=predict(X,m)
    for mode in MODES:frame[mode]=frame[[f'{mode}_s{k}' for k in SEEDS]].mean(axis=1)
    for seed in SEEDS:frame[f's{seed}']=frame[[f'{mode}_s{seed}' for mode in MODES]].mean(axis=1)
    frame['selected']=(frame['head']+frame['all'])/2
    return frame
def main():
    register();X,y=b.load_training();parts=[];replays=0
    for outer,(tr,va) in enumerate(b.training_folds(),1):
        assert y.loc[tr,'label_end'].max()<va.min();ref=s.anchor(f'o{outer}_outer')
        models=fit(X.loc[tr],y.loc[tr,'direction_up'],ref,f'o{outer}');frame=frame_predictions(X.loc[va],models,ref)
        frame['y']=y.loc[va,'direction_up'];frame['outer']=outer
        for m in models:
            np.testing.assert_allclose(predict(X.loc[va],m),frame[f'{m["mode"]}_s{m["seed"]}'],atol=0,rtol=0);replays+=1
        frame.to_parquet(HOME/f'outer{outer}.parquet');parts.append(frame)
    oof=pd.concat(parts);assert len(oof)==13104 and oof.index.is_unique;oof.to_parquet(HOME/'outer_oof.parquet')
    rows=s.summarize(oof,['current','selected','head','all']);seeds=s.summarize(oof,[f's{k}' for k in SEEDS])
    gate=bool(s.passes(rows[1]) and sum(r['delta_log_loss']<0 and r['delta_Brier']<0 for r in seeds)>=2)
    result=dict(status='COMPLETE',comparison=rows,seeds=seeds,passes_gate=gate,development_scored=False,both_development_metrics_improved=False)
    if gate:
        ref=s.anchor('final');models=fit(X,y.direction_up,ref,'final');xd=b.load_development_inputs();frame=frame_predictions(xd,models,ref)
        out=b.frozen_predictions(xd);out['p_up']=frame.selected;out.to_csv(HOME/'predictions.csv',index_label='decision_time',float_format='%.17g')
        save('final_pipeline.json',dict(models=models,recipe='Equal average of head/all branches and three seeds; each75%logistic+25%corrected probability.'))
        import extended_scorer_v45 as scorer
        meta=s.load(s.CURRENT/'metadata.json');meta.update(run_id=RUN,fit_count=s.count(),run_fit_count=s.count()-PRIOR,budget_extension='TEAM_48H_PLAYGROUND_V1_MEMBER03_EXTENSION_35',hypothesis='Warm-start conservative neural fine-tuning',selection_method=load('registration.json')['gate'],artifacts=dict(code=str(Path(__file__).relative_to(s.ROOT)),model=str((HOME/'final_pipeline.json').relative_to(s.ROOT)),environment=str((HOME/'environment-lock.txt').relative_to(s.ROOT))))
        save('metadata.json',meta);score=scorer.score(HOME/'predictions.csv',HOME/'score',HOME/'metadata.json')['candidate'];old=s.load(s.CURRENT/'score/RESULT.json')['candidate']
        result.update(development_scored=True,development_result=score,previous_frozen=old,both_development_metrics_improved=all(score[k]<old[k]-1e-12 for k in ['log_loss','Brier']))
        saved=pd.read_csv(HOME/'predictions.csv',index_col=0);np.testing.assert_allclose(frame.selected,saved.p_up,atol=1e-14,rtol=0)
        for m in models:np.testing.assert_allclose(predict(xd,m),frame[f'{m["mode"]}_s{m["seed"]}'],atol=0,rtol=0);replays+=1
    result.update(new_fits=s.count()-PRIOR,cumulative_fits=s.count());save('RESULT.json',result)
    periods=[dict(outer=int(i),model=c,**b.metrics(g.y,g[c])) for i,g in oof.groupby('outer') for c in ['current','selected','head','all']]
    pd.DataFrame(periods).to_csv(HOME/'PERIOD_RESULTS.csv',index=False)
    s.workbook(HOME/'FINE_TUNING_RESULTS.xlsx',{'Comparison':pd.DataFrame(rows),'Periods':pd.DataFrame(periods),'Seeds':pd.DataFrame(seeds)})
    verify();ledger=pd.read_csv(HERE/'experiments.csv');mine=ledger[ledger.run_id==RUN]
    assert int((mine.status=='STARTED').sum())==int((mine.status=='COMPLETE').sum())==result['new_fits']==18+6*int(gate)
    save('AUDIT.json',dict(status='PASS',replayed_models=replays,warm_start_hashes_checked=True,output_only_hidden_weights_unchanged=True,training_only_preprocessing_recomputed=True,chronological_purge_verified=True,ledger_reconciled=True,frozen_unchanged=True))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
