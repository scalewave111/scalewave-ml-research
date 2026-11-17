"""Read-only audit of constant liquidation activity and missing 168h volatility.
Save in Week 2/Code. No torch, fitting, imputation, data changes, or model selection.
Dependencies: numpy pandas joblib; duckdb for optional canonical-return verification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import joblib

LIQ='F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value'
VOL='PRICE__realized_vol_168h'
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
EXPECTED_SHA='ed122ee8f1582b669957f8b477e449134e30e020f83094f1e0dda28a0af64036'


def check(ok,msg):
    if not ok: raise ValueError(msg)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def missing_runs(s):
    # Consecutive observed dataset rows, not an assumption of continuous time coverage.
    mask=s.isna();groups=mask.ne(mask.shift()).cumsum()
    rows=[]
    for _,g in s[mask].groupby(groups[mask]):
        rows.append(dict(first_observed_missing=str(g.index[0]),last_observed_missing=str(g.index[-1]),
                         missing_rows=len(g),elapsed_hours=(g.index[-1]-g.index[0]).total_seconds()/3600))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    ap.add_argument('--skip-canonical',action='store_true',help='Skip optional raw-return check')
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    out=args.out or ws/'Code/experiments/feature_input_audit_v1'
    check(not out.exists(),f'Preserving {out}; use --out with a new folder')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid dataset index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected test population')
    check(LIQ in x and VOL in x,'Required feature missing')
    raw=x[[LIQ,VOL]].astype(float);data=raw.replace([np.inf,-np.inf],np.nan)
    side=pd.Series(np.where(y.signed_return_log_bps>0,'UP',np.where(y.signed_return_log_bps<0,'DOWN','FLAT')),index=y.index)
    stats=[];values=[]
    for issue in STARTS:
        end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(nanoseconds=1)
        populations={'train':y.index[y.label_end<issue],'test':dev[(dev>=issue)&(dev<end)]}
        for stage,idx in populations.items():
            for direction in ['ALL','UP','DOWN']:
                keys=idx if direction=='ALL' else idx[side.loc[idx].eq(direction)]
                for col in [LIQ,VOL]:
                    s=data.loc[keys,col];v=s.dropna();counts=v.value_counts()
                    stats.append(dict(issue=str(issue),stage=stage,side=direction,feature=col,rows=len(s),
                        original_nan_rows=int(raw.loc[keys,col].isna().sum()),
                        original_infinite_rows=int(np.isinf(raw.loc[keys,col]).sum()),
                        missing_fraction=float(s.isna().mean()),unique_finite_values=int(v.nunique()),
                        minimum=v.min(),maximum=v.max(),median=v.median(),std=v.std(ddof=0),
                        most_common_value=counts.index[0] if len(counts) else np.nan,
                        most_common_fraction_of_observed=float(counts.iloc[0]/len(v)) if len(v) else np.nan))
                    if col==LIQ:
                        for val,count in counts.head(10).items():
                            values.append(dict(issue=str(issue),stage=stage,side=direction,value=val,rows=count))
    daily=[]
    for day,idxs in pd.Series(data.index,index=data.index).groupby(data.index.floor('D')):
        keys=pd.DatetimeIndex(idxs)
        daily.append(dict(day=str(day),rows=len(keys),liq_unique=data.loc[keys,LIQ].nunique(),
                          liq_min=data.loc[keys,LIQ].min(),liq_max=data.loc[keys,LIQ].max(),
                          liq_missing=int(data.loc[keys,LIQ].isna().sum()),vol_missing=int(data.loc[keys,VOL].isna().sum())))
    missing=pd.DataFrame({'side':side,'in_test':y.index.isin(dev),'vol_missing':data[VOL].isna()},index=y.index)
    missing=missing[missing.vol_missing].drop(columns='vol_missing')
    # Read source text only; never import or execute candidate generator modules.
    hits=[];errors=[]
    needles=[LIQ,VOL,'historical_active_rate','min_periods=h*12','min_periods = h * 12']
    for root in [w1/'Code',ws/'Code']:
        if not root.exists(): continue
        for path in root.rglob('*.py'):
            if any(p in {'experiments','__pycache__','.venv','venv'} for p in path.relative_to(root).parts): continue
            if path.name==Path(__file__).name: continue
            try:
                lines=path.read_text(encoding='utf-8-sig').splitlines()
                matched=set()
                for i,line in enumerate(lines):
                    if any(n in line for n in needles): matched.update(range(max(0,i-2),min(len(lines),i+3)))
                for i in sorted(matched): hits.append(f'{path}:{i+1}: {lines[i]}')
            except (OSError,UnicodeError) as e: errors.append(f'{path}: {e}')
    canonical={'status':'SKIPPED','reason':'Requested --skip-canonical'}
    detail=None;gaps=None
    rawpath=w1/'Dataset/canonical_returns_5m.parquet'
    if not args.skip_canonical:
        if not rawpath.exists(): canonical={'status':'UNAVAILABLE','reason':str(rawpath)}
        else:
            try:
                import duckdb
                check(digest(rawpath)==EXPECTED_SHA,'Canonical-return file hash differs from previously verified source')
                con=duckdb.connect()
                try:
                    q=con.execute("SELECT decision_time,signed_return_log_bps,label_status,horizon_minutes FROM read_parquet(?) WHERE canonical_symbol='BTCUSDT' AND EXTRACT(minute FROM decision_time)::INTEGER % 5=0 ORDER BY decision_time",[str(rawpath)]).fetchdf()
                finally: con.close()
                check(q.horizon_minutes.eq(5).all(),'Unexpected return horizon')
                idx=pd.DatetimeIndex(pd.to_datetime(q.decision_time,utc=True))+pd.Timedelta(minutes=5)
                check(idx.is_unique and idx.is_monotonic_increasing,'Invalid return timestamps')
                grid=pd.date_range(idx.min(),idx.max(),freq='5min')
                r=pd.Series(q.signed_return_log_bps.to_numpy(float),index=idx)
                invalid=q.label_status.ne('VALID').to_numpy()|~np.isfinite(r.to_numpy())
                r.loc[idx[invalid]]=np.nan;r=r.reindex(grid)
                count=r.rolling('168h',closed='left',min_periods=0).count().reindex(data.index)
                rebuilt=np.sqrt(r.pow(2).rolling('168h',closed='left',min_periods=2016).sum()).reindex(data.index)
                detail=pd.DataFrame({'saved_vol':data[VOL],'rebuilt_vol':rebuilt,'valid_returns_in_past_168h':count,
                    'saved_missing':data[VOL].isna(),'rebuilt_missing':rebuilt.isna(),'side':side,'in_test':y.index.isin(dev)})
                both=detail.saved_vol.notna()&detail.rebuilt_vol.notna()
                error=float((detail.loc[both,'saved_vol']-detail.loc[both,'rebuilt_vol']).abs().max()) if both.any() else None
                mismatch=int(detail.saved_missing.ne(detail.rebuilt_missing).sum())
                canonical=dict(status='CHECKED',sha256=EXPECTED_SHA,return_rows=len(r),missing_returns=int(r.isna().sum()),
                    inserted_missing_grid_rows=len(grid)-len(idx),missing_mask_mismatches=mismatch,
                    paired_finite_rows=int(both.sum()),max_absolute_vol_difference=error,
                    matches_saved=(mismatch==0 and error is not None and error<=1e-8),
                    formula='sqrt(sum of squared log-bps returns), preceding 168h, closed=left, 2016 valid observations required')
                gaps=pd.DataFrame({'return_endpoint':r.index[r.isna()]})
            except Exception as e:
                canonical={'status':'FAILED','reason':f'{type(e).__name__}: {e}'}
    stats=pd.DataFrame(stats)
    focus=stats[(stats.stage=='test')&(stats.side.isin(['ALL','UP']))]
    summary=('READ-ONLY FEATURE INPUT AUDIT\nNo fitting, repairs, imputation, or model selection.\n'
        'Unique values and missing counts include ALL and direction-specific populations.\n'
        'A constant active-rate feature can be saturated by definition; this alone does not prove a bug.\n'
        'Strict 168h windows require 2016 valid 5m returns. One invalid return can affect subsequent windows for up to 168h.\n'
        'Canonical reconstruction is only confirmed when matches_saved is true. No missing values are filled.\n'
        'Source matches are excerpts for review, not proof of upstream provenance.\n'
        'Missing runs describe consecutive dataset rows; elapsed time does not establish continuous data coverage.\n\n'
        +focus.to_string(index=False)+'\n\nCANONICAL RETURN CHECK\n'+json.dumps(canonical,indent=2)
        +'\n\nSource excerpt lines: '+str(len(hits))+'; source read errors: '+str(len(errors)))
    out.mkdir(parents=True)
    stats.to_csv(out/'feature_value_stats.csv',index=False)
    pd.DataFrame(values).to_csv(out/'liquidation_top_values.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily_feature_quality.csv',index=False)
    missing.to_csv(out/'volatility_missing_timestamps.csv',index_label='decision_time')
    pd.DataFrame(missing_runs(data[VOL])).to_csv(out/'volatility_missing_runs.csv',index=False)
    if detail is not None: detail.to_csv(out/'volatility_reconstruction.csv',index_label='decision_time')
    if gaps is not None: gaps.to_csv(out/'invalid_return_endpoints.csv',index=False)
    (out/'source_excerpts.txt').write_text('\n'.join(hits+errors),encoding='utf-8')
    (out/'canonical_check.json').write_text(json.dumps(canonical,indent=2),encoding='utf-8')
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__': main()
