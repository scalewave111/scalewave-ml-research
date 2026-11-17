"""Exploratory 48h UP/DOWN conditional magnitude baseline."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
OUT=ROOT/'Code/experiments/conditional_48h_magnitude_v1'
CUTOFF=pd.Timestamp('2024-01-01',tz='UTC')

def ordinary_bps(log_bps):
    return 10000*np.expm1(np.asarray(log_bps,dtype=float)/10000)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    x,y,dev,ref=joblib.load(SRC)
    assert y.index.equals(x.index) and dev.equals(ref.index)
    assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    train=y.index[(y.index<CUTOFF)&(y.label_end<CUTOFF)]
    assert len(train)>10000 and len(dev)==18481
    target=pd.Series(ordinary_bps(y.signed_return_log_bps),index=y.index)
    isup=y.signed_return_log_bps>0
    rows=[]
    for side,mask in [('UP',isup.loc[train]),('DOWN',~isup.loc[train])]:
        values=target.loc[train].to_numpy()[mask.to_numpy()]
        rows.append(dict(side=side,training_rows=len(values),training_probability=len(values)/len(train),
            mean_signed_bps=float(values.mean()),mean_absolute_bps=float(abs(values).mean()),
            median_signed_bps=float(np.median(values))))
    pd.DataFrame(rows).to_csv(OUT/'training_conditional_sizes.csv',index=False)
    train_up=rows[0]['training_probability'];up_mean=rows[0]['mean_signed_bps'];down_mean=rows[1]['mean_signed_bps']
    forecasts=pd.DataFrame(index=dev)
    forecasts['actual_signed_bps']=target.loc[dev]
    forecasts['actual_direction']=np.where(forecasts.actual_signed_bps>0,'UP','DOWN')
    forecasts['training_direction_rate_up']=train_up
    forecasts['train_conditional_expected_signed_bps']=train_up*up_mean+(1-train_up)*down_mean
    forecasts['up_conditional_size_bps']=up_mean
    forecasts['down_conditional_size_bps']=down_mean
    forecasts.to_csv(OUT/'diagnostic_predictions.csv',index_label='decision_time')
    windows=[]
    for start in pd.date_range('2024-01-01','2025-01-01',freq='QS',tz='UTC'):
        end=start+pd.DateOffset(months=3)
        idx=dev[(dev>=start)&(dev<end)]
        z=forecasts.loc[idx]
        windows.append(dict(period=start.strftime('%Y-%m'),rows=len(z),up_rate=float((z.actual_signed_bps>0).mean()),
            mean_actual_signed_bps=float(z.actual_signed_bps.mean()),baseline_expected_signed_bps=float(z.train_conditional_expected_signed_bps.iloc[0]),
            signed_mae_vs_constant_bps=float(abs(z.actual_signed_bps-z.train_conditional_expected_signed_bps).mean()),
            mean_actual_up_bps=float(z.loc[z.actual_signed_bps>0,'actual_signed_bps'].mean()),
            mean_actual_down_bps=float(z.loc[z.actual_signed_bps<0,'actual_signed_bps'].mean())))
    pd.DataFrame(windows).to_csv(OUT/'period_diagnostics.csv',index=False)
    summary=dict(status='EXPLORATORY_DIAGNOSTIC_COMPLETE',target='48h signed endpoint return; simple bps conversion per observation',
        training='decision_time before 2024-01-01 and label_end before 2024-01-01',evaluation='saved 2024-01-01 through 2025-01-22 observations; historically exposed',
        training_rows=len(train),evaluation_rows=len(dev),direction_source='Realized historical outcomes used only to estimate training UP/DOWN frequencies and conditional sizes; no v45 timestamp probabilities were available or joined.',
        method='Training-period unconditional UP/DOWN empirical means; evaluation uses constant probability-weighted signed-bps expectation. Period outcomes grouped by realized side are diagnostic only and not available at forecast time.',
        limitation='Exploratory constant baseline, not a fitted conditional magnitude model or v45-combined forecast. Historical evaluation is exposed; not fresh confirmation.',
        training_conditional_sizes=rows,evaluation_periods=windows)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'SUMMARY.txt').write_text(pd.DataFrame(rows).to_string(index=False)+'\n\n'+pd.DataFrame(windows).to_string(index=False)+'\n',encoding='utf-8')
    print((OUT/'SUMMARY.txt').read_text())
if __name__=='__main__':main()
