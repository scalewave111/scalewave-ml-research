"""Portable frozen BTC48h development playground. No model fitting on import."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_training():
    return pd.read_parquet(ROOT/'data/X_train.parquet'), pd.read_parquet(ROOT/'data/y_train.parquet')

def load_development_inputs():
    return pd.read_parquet(ROOT/'data/X_development.parquet')

def training_folds():
    """Expanding folds; use only labels matured strictly before validation begins."""
    _, y = load_training()
    spec = json.loads((ROOT/'CONTRACT.json').read_text())
    for start, end in spec['inner_folds']:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        tr = y.index[y.label_end < start]
        va = y.index[(y.index >= start) & (y.index < end)]
        if not len(tr) or not len(va):
            raise ValueError('Empty fold')
        yield tr, va

def frozen_predictions(X):
    cards = json.loads((ROOT/'benchmarks/frozen_models.json').read_text())
    out = pd.DataFrame(index=X.index)
    for task in ['magnitude','direction']:
        c=cards[task]; cols=[np.ones(len(X))]
        for f in c['features']:
            v=X[f].replace([np.inf,-np.inf],np.nan)
            a=c['preprocessing'][f]
            cols.extend([(v.fillna(a['median']).to_numpy()-a['center'])/a['scale'],v.isna().to_numpy(float)])
        z=np.column_stack(cols)@np.asarray(c['coefficients'])
        if task=='magnitude': out['magnitude_log_bps']=np.maximum(z,0)
        else:
            # Stable logistic without overflow.
            q=np.exp(-np.abs(z));out['p_up']=np.where(z>=0,1/(1+q),q/(1+q))
    return out

def losses(y,p):
    prob=np.clip(p.p_up.to_numpy(),1e-15,1-1e-15)
    truth=y.direction_up.to_numpy()
    return pd.DataFrame({'MAE':np.abs(y.magnitude_log_bps.to_numpy()-p.magnitude_log_bps.to_numpy()),
        'log_loss':-(truth*np.log(prob)+(1-truth)*np.log1p(-prob)),
        'Brier':(truth-p.p_up.to_numpy())**2},index=y.index)

def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Hash mismatch: '+name)
    X,y=load_training(); dev=load_development_inputs()
    assert len(X)==29615 and len(dev)==18481 and X.index.equals(y.index)
    assert X.index.is_unique and dev.index.is_unique and list(X.columns)==list(dev.columns)
    assert y.label_end.max()<dev.index.min()
    saved=pd.read_parquet(ROOT/'benchmarks/development_predictions.parquet')
    replay=frozen_predictions(dev)
    assert replay.index.equals(saved.index)
    assert np.allclose(replay,saved[['magnitude_log_bps','p_up']],atol=1e-12,rtol=0)
    for tr,va in training_folds():assert y.loc[tr,'label_end'].max()<va.min()
    truth=pd.read_parquet(ROOT/'data/y_development.parquet')
    actual=losses(truth,replay).mean().to_dict()
    expected=json.loads((ROOT/'benchmarks/scores.json').read_text())['frozen_pair']
    for key in actual:assert abs(actual[key]-expected[key])<1e-12
    return {'status':'PASS','training_rows':len(X),'development_rows':len(dev),'input_fields':len(X.columns),'frozen_pair':actual,'new_fits':0}

def score(path,outdir,metadata):
    y=pd.read_parquet(ROOT/'data/y_development.parquet')
    p=pd.read_csv(path)
    required=['decision_time','magnitude_log_bps','p_up']
    if list(p.columns)!=required:raise ValueError('CSV columns must be exactly '+str(required))
    p['decision_time']=pd.to_datetime(p.decision_time,utc=True,errors='raise');p=p.set_index('decision_time')
    if not p.index.is_unique or not p.index.equals(y.index):raise ValueError('Submit every development key exactly once, in canonical order; no missing/extra rows')
    if not np.isfinite(p.to_numpy(float)).all():raise ValueError('Nonfinite predictions')
    if (p.magnitude_log_bps<0).any() or ((p.p_up<0)|(p.p_up>1)).any():raise ValueError('Invalid prediction range')
    meta=json.loads(Path(metadata).read_text())
    for k in ['member','run_id','hypothesis','features','fit_count','selection_method','training_cutoff','development_used_for_fitting','artifacts']:
        if k not in meta:raise ValueError('Missing metadata: '+k)
    if meta['development_used_for_fitting'] is not False:raise ValueError('Development-label fitting is not eligible')
    if not isinstance(meta['fit_count'],int) or meta['fit_count']<0:raise ValueError('Invalid fit count')
    if pd.Timestamp(meta['training_cutoff'])>pd.Timestamp('2024-01-01T00:00:00Z'):raise ValueError('Training cutoff outside contract')
    spec=json.loads((ROOT/'CONTRACT.json').read_text())
    if meta['fit_count']>spec['initial_fit_budget_per_member']:raise ValueError('Budget exceeded; register a new contract version first')
    c=losses(y,p);ref=pd.read_parquet(ROOT/'benchmarks/development_predictions.parquet')
    b=losses(y,ref);m0=losses(y,ref.rename(columns={'magnitude_log_bps':'pair_mag','p_up':'pair_p','M0_magnitude':'magnitude_log_bps','M0_p_up':'p_up'}))
    overall=c.mean();delta=c-b;monthly=c.groupby(c.index.strftime('%Y-%m')).mean()
    monthly_ref=b.groupby(b.index.strftime('%Y-%m')).mean();monthly_m0=m0.groupby(m0.index.strftime('%Y-%m')).mean()
    intervals=[]
    for days in [7,14]:
        group=((c.index-c.index.min()).total_seconds()//(days*86400)).astype(int)
        sizes=pd.Series(1,index=c.index).groupby(group).sum()
        sums=delta.groupby(group).sum().reindex(sizes.index)
        draws=np.random.default_rng(48+days).integers(0,len(sizes),(2000,len(sizes)))
        for col in c.columns:
            vals=sums[col].to_numpy()[draws].sum(1)/sizes.to_numpy()[draws].sum(1)
            lo,hi=np.quantile(vals,[.025,.975]);intervals.append({'metric':col,'block_days':days,'low':float(lo),'high':float(hi)})
    # Ignore round-trip floating-point noise; this is not a practical-effect gate.
    eps=1e-12
    meanbeat=overall<b.mean()-eps
    mag=bool(meanbeat.MAE and overall.MAE<m0.MAE.mean()-eps)
    direction=bool(meanbeat.log_loss and meanbeat.Brier and overall.log_loss<m0.log_loss.mean()-eps and overall.Brier<m0.Brier.mean()-eps)
    result={'run_id':meta['run_id'],'rows':len(p),'candidate':overall.to_dict(),'frozen_pair':b.mean().to_dict(),'M0':m0.mean().to_dict(),
        'delta_vs_pair':delta.mean().to_dict(),'magnitude_average_win':mag,'direction_average_win':direction,
        'months_better_pair':(monthly<monthly_ref).sum().to_dict(),'nominal_paired_block_intervals':intervals,
        'interpretation':'Development comparison only. Average win is not promotion. Intervals do not correct team-wide or prior adaptive selection. Metadata is self-declared; code/lineage review required.'}
    out=Path(outdir)
    if out.exists():raise ValueError('Output directory already exists; use a new run ID')
    out.mkdir(parents=True);(out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'submission.json').write_text(json.dumps(meta,indent=2))
    monthly.add_prefix('candidate_').join(monthly_ref.add_prefix('pair_')).join(monthly_m0.add_prefix('M0_')).to_csv(out/'monthly.csv')
    delta.to_parquet(out/'paired_loss_differences.parquet');p.to_csv(out/'predictions.csv',float_format='%.17g')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest='command',required=True)
    sub.add_parser('verify');r=sub.add_parser('replay');r.add_argument('--output',required=True)
    q=sub.add_parser('score');q.add_argument('--predictions',required=True);q.add_argument('--metadata',required=True);q.add_argument('--output',required=True)
    args=a.parse_args()
    if args.command=='verify':print(json.dumps(verify(),indent=2))
    elif args.command=='replay':
        target=Path(args.output)
        if target.exists():raise ValueError('Refusing to overwrite predictions')
        target.parent.mkdir(parents=True,exist_ok=True);frozen_predictions(load_development_inputs()).to_csv(target,index_label='decision_time',float_format='%.17g');print(target)
    else:print(json.dumps(score(args.predictions,args.output,args.metadata),indent=2))
