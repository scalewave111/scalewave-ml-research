"""User-run net300h adaptation of the focused HGB/GRU16 MAE refinement with previous net policy control."""
from pathlib import Path
import argparse,sys,json,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import fit_net_magnitude_300h as base
import net_300h_long_features as price_features
import net_300h_market_features as state_features

def load(args,models):
    _,y,dev,population,_,_,files,audit=base.load(args)
    import fit_excursions_48h as prior
    timeline,_,_,_,_=prior.e2.old.load()
    market,market_files=state_features.load_sources(prior.e2.old.data.DATA,timeline.index)
    files += market_files
    returns_path=prior.e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=prior.e2.price.canonical.load_returns(returns_path)
    price,pdict=price_features.build(r,market.index,300)
    state,sdict=state_features.build(market,price,market.index)
    full=price.join(state)
    cols=list(dict.fromkeys(c for pack in models.PACKS for c in models.pack_columns(full,pack)))
    full=full[cols]
    cut=market.index[len(market)//2];keys=market.index[market.index<=cut][-10:]
    changed=r.copy();changed.loc[changed.index>=cut]=1e9
    pp,_=price_features.build(changed,market.index,300)
    mm=market.copy();mm.loc[mm.index>=cut]=1e9
    ss,_=state_features.build(mm,pp,market.index)
    pd.testing.assert_frame_equal(full.loc[keys],pp.join(ss).loc[keys,cols])
    prefix=market.loc[market.index<=cut]
    pp,_=price_features.build(r.loc[r.index<=cut],prefix.index,300)
    ss,_=state_features.build(prefix,pp,prefix.index)
    pd.testing.assert_frame_equal(full.loc[keys],pp.join(ss).loc[keys,cols])
    # Exact sequences are constructed before any outcome eligibility filter.
    x=models.augment(full).loc[y.index].copy()
    seq=[dict(feature=f'SEQ{lag}__{c}',family='sequence',formula=f'Exact snapshot at t-{lag}h, no filling',sources=c) for lag in models.LAGS for c in models.pack_columns(full,'leverage')]
    dictionary=pd.concat([pdict,sdict,pd.DataFrame(seq)],ignore_index=True).fillna('')
    packs={p:models.pack_columns(x,p) for p in models.PACKS}
    audit.update(features=len(x.columns),base_pack_union_features=len(cols),market_source='Required columns projected from original masked X_train/X_development, aligned to full timeline',market_source_columns=state_features.SOURCES,packs={k:len(v) for k,v in packs.items()},gru_snapshots_hours=models.LAGS,feature_causality='prefix and future perturbation passed; sequences built before label filtering')
    files += [Path(price_features.__file__),Path(state_features.__file__)]
    return x,y,dev,population,packs,dictionary,files,audit

def metrics(actual,pred):
    m=base.metrics(actual,pred);m['median_absolute_error']=float(np.median(np.abs(np.asarray(pred)-np.asarray(actual))))
    return m

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--week2-code',type=Path,default=base.ROOT.parent/'Week 2/Code')
    parser.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    parser.add_argument('--canonical-manifest',type=Path,default=base.ROOT/'Datasets/canonical_receipts/OUTPUT_FILES.json')
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args();base.check(args.threads>0,'Positive threads required')
    sys.path.insert(0,str(args.week2_code))
    import net_300h_refinement as models
    import torch
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        x,y,dev,population,packs,dictionary,files,audit=load(args,models)
        reference=base.HERE/'experiments/net_magnitude_300h_downstyle_v1/20261116T111254635059Z'
        receipt=json.loads((reference/'COMPLETE.json').read_text())
        base.check(receipt['status']=='COMPLETE' and base.sha(reference/'predictions.csv')==receipt['prediction_sha256'],'Previous run integrity failure')
        previous=pd.read_csv(reference/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True)
        base.check(previous.index.is_unique and previous.index.equals(dev.sort_values()),'Previous evaluation timestamps mismatch')
        np.testing.assert_allclose(previous.actual,y.loc[previous.index,base.TARGET],atol=1e-8,rtol=0)
        files += [reference/'COMPLETE.json',reference/'predictions.csv',reference/'CONTRACT.json']
        folds={issue:models.windows(y.index,issue) for issue in base.ISSUES}
        if args.check_inputs:print(json.dumps(audit,indent=2,default=str));print('PASS: inputs and causal checks; no model fitting.');return
        out=base.HERE/'experiments/net_magnitude_300h_refined_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        report=base.ROOT/'Report/net_magnitude_300h_refined_v1'/out.name
        out.mkdir(parents=True);report.mkdir(parents=True)
        files += [Path(__file__),Path(base.__file__),base.ROOT/'PROJECT_CONTEXT.md']
        for m in list(sys.modules.values()):
            p=getattr(m,'__file__',None)
            if p and Path(p).suffix=='.py' and 'Desktop\\ML\\' in str(Path(p)):files.append(Path(p))
        framework=base.ROOT.parents[1]/'ml_framework'
        files += [framework/v for v in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        base.dump(out/'CONTRACT.json',dict(target=base.TARGET,formula=base.FORMULA,horizon_minutes=18000,arguments=vars(args),audit=audit,packs=packs,configs=models.CONFIG,weights=models.WEIGHTS,anchor_mean_weights=models.ANCHOR_WEIGHTS,gru_lags=models.LAGS,tree_cap=models.tree.CAP,neural_cap=models.CAP,epochs=models.EPOCHS,
            method='Focused net300h leverage-pack refinement; original HGB/GRU16 controls, depth1 HGB and decay.1 GRU16 alternatives, finer model and anchor weights. Joint quarterly selection. Four earlier3month windows at12/9/6/3months, strict300h maturity. Rank pooledMAE; .1%gain against both baselines,3/4wins/ties versus stronger window baseline,no>2%windowloss; otherwise median. Candidate diagnostic separate. RMSE/medianAE reported,not gated.',
            provenance='Read-only original Week2 model/search/preprocessing code; new local causal price/state features; canonical net300h labels. Previous net300h selected policy is a hash/alignment-verified comparison only, never a tuning input; prior UP/DOWN targets are not consumed. No guarantee of gain; exposed historical evaluation.',
            hashes={str(p):base.sha(p) for p in sorted(set(files))},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','torch','joblib']}))
        for p in set(files):
            if p.suffix=='.py':(out/p.name).write_bytes(p.read_bytes())
        dictionary.to_csv(out/'feature_dictionary.csv',index=False);population.to_csv(out/'population_audit.csv',index_label='decision_time')
        frames=[];selections=[]
        target=y[base.TARGET].to_numpy()
        print('Net300h refinement:64inner fits;leverage pack x4MAEconfigs x4windows x4issues. No runtime guarantee.',flush=True)
        for issue in base.ISSUES:
            folder=out/f'{issue:%Y%m}';folder.mkdir()
            plan=models.search(x,target,folds[issue],folder)
            base.dump(folder/'fit_boundaries.json',[dict(cutoff=str(issue-pd.DateOffset(months=offset)),max_training_label_end=str(y.iloc[tr].label_end.max()),validation_start=str(y.index[va].min()),max_validation_label_end=str(y.iloc[va].label_end.max())) for offset,(tr,va) in zip([12,9,6,3],folds[issue])])
            tr=y.index[y.label_end<issue];base.check(len(tr)>=1000,'Too few mature outer rows')
            end=base.ISSUES[list(base.ISSUES).index(issue)+1] if issue!=base.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
            te=dev[(dev>=issue)&(dev<end)];base.check(len(te)>0,'Empty evaluation')
            f=pd.DataFrame({'actual':y.loc[te,base.TARGET],'issue':issue,'mean':y.loc[tr,base.TARGET].mean(),'median':y.loc[tr,base.TARGET].median()},index=te)
            record=dict(issue=str(issue),plan=plan,models=[])
            for policy in ['selected','candidate']:
                spec=plan[policy];a=models.fit(x.loc[tr],y.loc[tr,base.TARGET].to_numpy(),spec)
                path=folder/(policy+'.joblib');joblib.dump(a,path,compress=3)
                prediction=models.predict(a,x.loc[te],spec.get('weight',1.))
                np.testing.assert_allclose(prediction,models.predict(joblib.load(path),x.loc[te],spec.get('weight',1.)),atol=1e-8,rtol=0)
                f[policy]=prediction
                record['models'].append(dict(policy=policy,artifact=str(path.relative_to(out)),sha256=base.sha(path),features=a['features'],spec=spec,training_rows=len(tr),max_training_label_end=str(y.loc[tr,'label_end'].max())))
            frames.append(f);selections.append(record);base.dump(out/'selections.json',selections)
        result=pd.concat(frames).sort_index();base.check(result.index.equals(dev.sort_values()),'Coverage mismatch')
        base.check(np.isfinite(result[['actual','mean','median','selected','candidate']]).all().all(),'Nonfinite final forecasts')
        for col in ['actual','mean','median']:np.testing.assert_allclose(result[col],previous[col],atol=1e-8,rtol=0)
        result['previous_selected']=previous['selected']
        result['previous_candidate']=previous['candidate']
        result.to_csv(out/'predictions.csv',index_label='decision_time')
        rows=[]
        for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
            for policy in ['mean','median','selected','candidate','previous_selected','previous_candidate']:rows.append(dict(period=period,model=policy,**metrics(g.actual,g[policy])))
        table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False);table.to_csv(report/'metrics.csv',index=False)
        text='300h net magnitude focused MAE refinement\n'+table.to_string(index=False)+'\nSelected may be median fallback; candidate diagnostic only. Historical exposed evaluation.\n'
        (report/'SUMMARY.txt').write_text(text);base.dump(report/'selections.json',selections)
        base.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),target=base.TARGET,prediction_sha256=base.sha(out/'predictions.csv'),replay_passed=True,previous_prediction_sha256=base.sha(reference/'predictions.csv')))
        print(text,'\nOutputs:',out,'\nReport:',report,flush=True)

if __name__=='__main__':main()
