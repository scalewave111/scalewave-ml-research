"""Independent 15-minute net magnitude model; no training on import.

python Code/fit_net_magnitude_15m.py --pure-model hgb
python Code/fit_net_magnitude_15m.py --check-inputs
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_net_magnitude_48h as core

ROOT=Path(__file__).resolve().parents[1]
TARGET=core.TARGET
HORIZON=pd.Timedelta(minutes=15)
TRANSITION=pd.Timestamp('2024-10-29T00:01:00Z')


def engineered(r,index):
    """57 causal short/long price features; windows exclude endpoint at origin."""
    values={}; definitions=[]
    def add(name,value,formula):
        key='NET15__'+name;values[key]=value
        definitions.append(dict(feature=key,formula=formula))
    for minutes in [15,30,60,240,1440,10080]:
        n=minutes//5;kw=dict(window=f'{minutes}min',closed='left',min_periods=n)
        roll=r.rolling(**kw);net=roll.sum();absolute=r.abs().rolling(**kw).sum()
        square=r.pow(2).rolling(**kw).sum();rv=np.sqrt(square)
        up=r.clip(lower=0).pow(2).rolling(**kw).sum();peak=r.abs().rolling(**kw).max()
        fields=[('rv',rv,'sqrt(sum(r^2))'),('net',net,'sum(r)'),('abs_net',net.abs(),'abs(sum(r))'),
            ('abs_sum',absolute,'sum(abs(r))'),('efficiency',net.abs()/absolute.clip(lower=1e-12),'abs(sum(r))/max(sum(abs(r)),1e-12)'),
            ('up_variance_share',up/square.clip(lower=1e-12),'sum(max(r,0)^2)/max(sum(r^2),1e-12)'),
            ('jump_share',peak.pow(2)/square.clip(lower=1e-12),'max(abs(r))^2/max(sum(r^2),1e-12)'),
            ('rms',rv/np.sqrt(n),f'sqrt(sum(r^2)/{n})'),
            ('positive_share',r.gt(0).astype(float).where(r.notna()).rolling(**kw).mean(),'mean(1[r>0])')]
        for name,value,formula in fields:
            add(f'{name}_{minutes}m',value,f'{formula}; prior {minutes} minutes; {n} finite 5m endpoints required; endpoint < origin')
    for short,long in [(15,60),(30,240),(60,1440)]:
        add(f'rv_ratio_{short}_{long}',(values[f'NET15__rv_{short}m']/np.sqrt(short))/(values[f'NET15__rv_{long}m']/np.sqrt(long)).clip(lower=1e-9),
            f'(RV_{short}m/sqrt({short}))/max(RV_{long}m/sqrt({long}),1e-9)')
    return pd.DataFrame(values).reindex(index).replace([np.inf,-np.inf],np.nan),definitions


def model_design(x,keys,pp,kind):
    if kind!='gru':return core.transform(x.reindex(keys),pp)
    # All snapshots lie on the original half-hour X grid; no nearest-time fill.
    return np.stack([core.transform(x.reindex(keys-pd.Timedelta(minutes=m)),pp) for m in [90,60,30,0]],axis=1)


def load(args):
    sys.path.insert(0,str(args.week2_code))
    import improve_excursion_features_15m as prior
    paths=[args.data_dir/n for n in ['X_train.parquet','X_development.parquet']]
    train,dev=[prior.base.read_x(path) for path in paths]
    core.check(len(train)==29615 and len(dev)==18481,'Original row counts changed')
    core.check(train.columns.equals(dev.columns),'X schemas differ')
    columns=[c for c in train if c.startswith('F4Z__') and c.endswith('__value')]
    core.check(len(columns)==522,'Expected 522 masked market value inputs')
    x=pd.concat([train[columns],dev[columns]]).sort_index().replace([np.inf,-np.inf],np.nan)
    x.index=x.index.as_unit('ns');dev.index=dev.index.as_unit('ns')
    core.check(x.index.is_unique and not x.index.hasnans,'Invalid original timeline')
    core.check((x.index.minute%30==0).all(),'Unexpected original X grid')
    files=sorted((args.canonical_root/'y/horizon_minutes=15/symbol=BTCUSDT').rglob('*.parquet'))
    core.check(bool(files),'No canonical 15-minute Y partitions')
    receipt=args.canonical_root/'OUTPUT_FILES.json'
    receipts={r['path']:r['sha256'] for r in json.loads(receipt.read_text())}
    for path in files:
        rel=path.relative_to(args.canonical_root).as_posix()
        core.check(rel in receipts and core.sha(path)==receipts[rel],f'Canonical checksum mismatch: {rel}')
    with duckdb.connect() as con:
        con.register('origins',pd.DataFrame({'decision_time':x.index}))
        raw=con.execute('''SELECT y.decision_time,y.label_status,y.canonical_symbol,y.horizon_minutes,
            y.net_magnitude_log_bps,y.direction,y.up_magnitude_log_bps,y.down_magnitude_log_bps
            FROM read_parquet(?,hive_partitioning=false) y JOIN origins k ON k.decision_time=y.decision_time''',[[str(p) for p in files]]).fetchdf()
    raw.decision_time=pd.to_datetime(raw.decision_time,utc=True).dt.as_unit('ns')
    core.check(raw.decision_time.is_unique,'Duplicate Y origins')
    raw=raw.set_index('decision_time').reindex(x.index)
    targetcols=[TARGET,'up_magnitude_log_bps','down_magnitude_log_bps']
    valid=(raw.label_status.eq('VALID') & raw.canonical_symbol.eq('BTCUSDT') & pd.to_numeric(raw.horizon_minutes).eq(15)
           & np.isfinite(raw[targetcols]).all(axis=1) & raw[targetcols].ge(0).all(axis=1)
           & raw.direction.isin([-1,0,1]) & raw.direction.eq(0).eq(raw[TARGET].eq(0)))
    signed=raw.direction*raw[TARGET]
    valid &= (signed>=-raw.down_magnitude_log_bps-1e-7)&(signed<=raw.up_magnitude_log_bps+1e-7)
    crossing=(x.index<TRANSITION)&(x.index+HORIZON>=TRANSITION)
    population=pd.DataFrame({'status':np.where(~valid,'missing_or_invalid_label',np.where(crossing,'source_binding_crossing','included'))},index=x.index)
    keep=valid & ~crossing;keys=x.index[keep]
    core.check(len(keys)>0,'No valid net target rows')
    y=pd.DataFrame({TARGET:raw.loc[keys,TARGET].astype(float),'label_end':keys+HORIZON},index=keys)
    evaluation=dev.index.intersection(keys)
    returns=args.returns
    r=prior.canonical.load_returns(returns)
    prior_price,_=prior.price_features(r,x.index)
    new,definitions=engineered(r,x.index)
    cut=x.index[len(x)//2];probe=x.index[x.index<=cut][-10:]
    pd.testing.assert_frame_equal(engineered(r,probe)[0],engineered(r.loc[:cut],probe)[0])
    changed=r.copy();changed.loc[changed.index>=cut]=1e9
    pd.testing.assert_frame_equal(engineered(r,probe)[0],engineered(changed,probe)[0])
    # No target filtering of the sequence timeline.
    design=x.join(prior_price).join(new)
    core.check(design.columns.is_unique,'Duplicate design columns')
    packs={'market':list(x),'prior_price':list(prior_price),'new_price':list(new),
           'all':list(design),'market_new':list(x)+list(new)}
    audit=dict(target=TARGET,horizon_minutes=15,formula=core.FORMULA,original_rows=len(x),eligible_rows=len(y),evaluation_rows=len(evaluation),
        exclusions=population.status.value_counts().to_dict(),feature_counts={p:len(c) for p,c in packs.items()},
        evaluation_first=str(evaluation.min()),evaluation_last=str(evaluation.max()),origin_grid='original 30-minute grid, not full canonical minute population',
        gru_snapshot_minutes=[90,60,30,0],causality='new-feature prefix and future-perturbation checks passed',
        target_min=float(y[TARGET].min()),target_max=float(y[TARGET].max()))
    sources=paths+files+[receipt,returns,Path(__file__),Path(core.__file__),Path(prior.__file__),Path(prior.base.__file__),Path(prior.canonical.__file__),args.canonical_root/'team_tools/build_dataset.py']
    return design,y,evaluation,population,packs,definitions,sources,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pure-model',choices=['hgb','extra','mlp','gru'],default='hgb')
    p.add_argument('--week2-code',type=Path,default=ROOT.parent/'Week 2/Code')
    p.add_argument('--data-dir',type=Path,default=ROOT.parent/'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data')
    p.add_argument('--returns',type=Path,default=ROOT.parent/'Week 1/Dataset/canonical_returns_5m.parquet')
    p.add_argument('--canonical-root',type=Path,default=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1'))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--row-cap',type=int,default=6000)
    p.add_argument('--check-inputs',action='store_true')
    args=p.parse_args();core.check(args.threads>0 and args.row_cap>=256,'Invalid threads/row cap')
    # Reuse the tested family models/validation engine, with horizon-specific sequences.
    core.model_design=model_design
    if args.pure_model=='gru' and not args.check_inputs:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError as exc:raise ImportError('GRU requires torch.') from exc
    with threadpool_limits(limits=args.threads):
        x,y,dev,population,packs,definitions,sources,audit=load(args)
        if args.check_inputs:print(json.dumps(audit,indent=2));print('Input checks only; no model fitting.');return
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');name=f'net_magnitude_15m_{args.pure_model}_only'
        out=ROOT/'Code/experiments'/(name+'_v1')/stamp;report=ROOT/'Report'/name/stamp
        out.mkdir(parents=True);report.mkdir(parents=True)
        framework=ROOT.parents[1]/'ml_framework'
        sources += [ROOT/'PROJECT_CONTEXT.md']+[framework/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        versions=['numpy','pandas','scikit-learn','duckdb','joblib']+(['torch'] if args.pure_model=='gru' else [])
        core.dump(out/'CONTRACT.json',dict(target=TARGET,horizon_minutes=15,formula=core.FORMULA,arguments=vars(args),input_audit=audit,packs=packs,
            input_hashes={str(f):core.sha(f) for f in sorted(set(sources))},versions={v:importlib.metadata.version(v) for v in versions},seed=48,
            policy='One model family in all periods; feature pack selected by pooled MAE in three earlier two-month windows offset9/6/3months. Training label_end strictly before cutoff. No baseline fallback in primary forecast.',
            preprocessing='Training-only >=80%coverage and variation; median/IQR; clip8; missing flags. Full mature baselines/target scaling; evenly spaced estimator cap.',
            models='Same families/hyperparameters as 48h: HGB60/L1/depth3/7leaves; ExtraTrees48/L2/depth10; MLP16/40epochs/L2; GRU16/24epochs/L1/AdamWlr.001decay.03/clip1. GRU snapshots90/60/30/0minutes.',
            limitation='Historical exposed evaluation; original half-hour X grid; inherited F4Z availability masks; group feature search, not exhaustive individual selection; no guaranteed gain; no resume.'))
        population.to_csv(out/'population_audit.csv',index_label='decision_time')
        pd.DataFrame(definitions).to_csv(out/'new_feature_formulas.csv',index=False)
        (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
        for src in [Path(__file__),Path(core.__file__)]: (out/src.name).write_bytes(src.read_bytes())
        print('Output:',out,flush=True)
        core.run(x,y,dev,packs,out,report,[args.pure_model],args.row_cap,True,horizon_label='15m')


if __name__=='__main__':main()
