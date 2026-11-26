"""Independent all-entry accounting. Concurrent positions are permitted."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/SAK_GROSS_V01_COMPLETE'

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if hasattr(v,'item'):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def write(name,value):
    (ROOT/name).write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')

def stats(frame):
    score=frame.signed_score_bps.to_numpy();r=frame.actual_bps.to_numpy()
    side=np.sign(score);gross=side*r;n=len(frame)
    cumulative=np.r_[0.,np.cumsum(gross)]
    return {'trades':n,'longs':int((side>0).sum()),'shorts':int((side<0).sum()),
            'wins':int((gross>0).sum()),'losses':int((gross<0).sum()),'breakeven':int((gross==0).sum()),
            'win_rate':float((gross>0).mean()) if n else None,'wins_at_least_10bps':int((gross>=10).sum()),
            'avg_gross_bps':float(gross.mean()) if n else None,'sum_gross_bps':float(gross.sum()),
            'max_realised_drawdown_bps_equal_entry_notional':float((np.maximum.accumulate(cumulative)-cumulative).max()),
            'always_long_same_entries_avg_bps':float(r.mean()) if n else None,
            'long_avg_gross_bps':float(gross[side>0].mean()) if (side>0).any() else None,
            'short_avg_gross_bps':float(gross[side<0].mean()) if (side<0).any() else None,
            **{f'avg_net_assuming_{cost}bps_round_trip':float(gross.mean()-cost) if n else None for cost in [2,4,8]}}

def bootstrap(trades):
    dates=pd.date_range('2023-04-01','2024-12-31',freq='D',tz='UTC')
    daily=pd.DataFrame({'n':1,'wins':trades.gross_bps.gt(0).astype(int),'gross':trades.gross_bps},index=trades.index).groupby(trades.index.floor('D')).sum().reindex(dates,fill_value=0)
    a=daily.to_numpy(float);n=len(a);rng=np.random.default_rng(20261125);samples=[]
    for _ in range(2000):
        starts=rng.integers(0,n-14+1,size=math.ceil(n/14))
        rows=(starts[:,None]+np.arange(14)).ravel()[:n]
        count,wins,pnl=a[rows].sum(axis=0)
        if count:samples.append([wins/count,pnl/count])
    bounds=np.quantile(samples,[.025,.975],axis=0)
    return {'method':'2000 moving-block replicates, 14 calendar days, full 2023Q2-2024Q4 calendar, seed 20261125',
            'accuracy_ci95':bounds[:,0].tolist(),'gross_bps_ci95':bounds[:,1].tolist(),
            'active_days':int((daily.n>0).sum()),'top5_day_share_of_net_signed_gross_sum':float(daily.gross.nlargest(5).sum()/daily.gross.sum()),
            'limitation':'No correction for historical research/model selection; gross costs not included'},daily

def main():
    frames=[pd.read_parquet(p) for p in sorted((SOURCE/'evidence/replay').glob('20*.parquet'))]
    data=pd.concat(frames).sort_index()
    assert data.index.is_unique and len(data)==923007
    cols=['p_up','p_down','predicted_magnitude_bps','signed_score_bps','actual_bps','gross_bps']
    assert np.isfinite(data[cols].to_numpy()).all()
    assert ((data.p_up>=0)&(data.p_up<=1)).all() and (data.predicted_magnitude_bps>=0).all()
    assert np.array_equal(data.eligible.to_numpy(),data.signed_score_bps.abs().ge(10).to_numpy())
    np.testing.assert_allclose(data.signed_score_bps,(2*data.p_up-1)*data.predicted_magnitude_bps,rtol=0,atol=0)
    np.testing.assert_allclose(data.gross_bps,np.sign(data.signed_score_bps)*data.actual_bps,rtol=0,atol=0)
    assert (data.entry_timestamp_utc==data.index+pd.Timedelta(minutes=1)).all()
    assert (data.exit_timestamp_utc==data.index+pd.Timedelta(minutes=16)).all()
    selected=data.loc[data.eligible].copy()
    selected.to_parquet(ROOT/'ALL_ENTRY_LEDGER.parquet')
    summary=stats(selected)
    times=selected.index.as_unit('ns').asi8;hold=15*60*10**9
    concurrent=np.arange(1,len(times)+1)-np.searchsorted(times+hold,times,side='right')
    summary['peak_simultaneous_positions']=int(concurrent.max())
    summary['mean_active_positions_at_signal_arrival']=float(concurrent.mean())
    summary['all_entries_allowed']=True
    summary['unweighted_sum_is_not_account_return']=True
    keep=[];until=None
    for i,t in enumerate(times):
        if until is None or t>=until:keep.append(i);until=t+hold
    single=selected.iloc[keep].copy();single.to_parquet(ROOT/'SINGLE_POSITION_DIAGNOSTIC.parquet')
    rows=[]
    for month in pd.period_range('2023-04','2024-12',freq='M'):
        start=month.start_time.tz_localize('UTC');stop=(month+1).start_time.tz_localize('UTC')
        f=selected[(selected.index>=start)&(selected.index<stop)]
        all_month=data[(data.index>=start)&(data.index<stop)]
        record={'month':str(month),'forecast_rows':len(all_month),**stats(f)}
        record['active_days']=len(f.index.floor('D').unique())
        record['positive_score_actual_up']=int(((f.signed_score_bps>0)&(f.actual_bps>0)).sum())
        record['positive_score_actual_down']=int(((f.signed_score_bps>0)&(f.actual_bps<0)).sum())
        record['negative_score_actual_down']=int(((f.signed_score_bps<0)&(f.actual_bps<0)).sum())
        record['negative_score_actual_up']=int(((f.signed_score_bps<0)&(f.actual_bps>0)).sum())
        rows.append(record)
    monthly=pd.DataFrame(rows)
    assert monthly.trades.sum()==1145 and monthly.wins.sum()==755
    assert monthly.longs.sum()+monthly.shorts.sum()==1145
    monthly.to_csv(ROOT/'MONTHLY_TRADES.csv',index=False)
    write('MONTHLY_TRADES.json',rows)
    quarterly=[]
    for q in pd.period_range('2023Q2','2024Q4',freq='Q'):
        f=selected[(selected.index>=q.start_time.tz_localize('UTC'))&(selected.index<(q+1).start_time.tz_localize('UTC'))]
        quarterly.append({'quarter':str(q),**stats(f)})
    pd.DataFrame(quarterly).to_csv(ROOT/'QUARTERLY_TRADES.csv',index=False)
    boot,daily=bootstrap(selected);singleboot,_=bootstrap(single)
    daily.to_csv(ROOT/'DAILY_TRADES.csv',index_label='day')
    p=np.clip(data.p_up.to_numpy(),1e-12,1-1e-12);up=data.actual_bps.to_numpy()>0
    losses={'direction_log_loss':float(-np.log(np.where(up,p,1-p)).mean()),'direction_accuracy_all_origins':float(((p>.5)==up).mean()),
            'magnitude_mae_bps':float(np.abs(data.predicted_magnitude_bps-np.abs(data.actual_bps)).mean())}
    reference=json.loads((SOURCE/'evidence/statistical_checks.json').read_text())
    boot_error=float(np.max(np.abs(np.array([boot['accuracy_ci95'],boot['gross_bps_ci95']])-np.array([reference['accuracy_95_interval'],reference['avg_gross_bps_95_interval']]))))
    boot_matches=boot_error<1e-10
    input_path=ROOT/'independent_inputs/research_week2_v2/aligned_panel.parquet'
    label_check={'status':'PENDING_INPUT_RECONSTRUCTION'}
    if input_path.exists():
        target=pd.read_parquet(input_path).r.reindex(data.index)
        error=np.abs(target.to_numpy()-data.actual_bps.to_numpy())
        label_check={'rows':len(data),'valid':bool(np.isfinite(target).all()),'max_abs_bps_error':float(error.max()),'exact_values':bool(np.array_equal(target,data.actual_bps))}
        assert label_check['valid'] and error.max()<1e-10
    result={'all_entries_primary':summary,'single_position_diagnostic_only':stats(single),'all_entry_months':rows,
            'forecast_losses':losses,'provided_bootstrap_reproduced':boot_matches,'primary_block_intervals':boot,
            'bootstrap_max_numeric_error':boot_error,
            'single_position_block_intervals':singleboot,'native_target_check':label_check,
            'economic_interpretation':'All entries may be executed subject to a declared sizing/exposure plan. Up to 15 simultaneous unit positions; results are gross reference-VWAP returns, not fills or account ROI.'}
    write('LEDGER_AUDIT.json',result)
    print(monthly[['month','trades','longs','shorts','wins','losses','win_rate','avg_gross_bps','sum_gross_bps']].to_string(index=False),flush=True)
    print(json.dumps(clean({k:v for k,v in result.items() if k!='all_entry_months'}),indent=2),flush=True)

if __name__=='__main__':main()
