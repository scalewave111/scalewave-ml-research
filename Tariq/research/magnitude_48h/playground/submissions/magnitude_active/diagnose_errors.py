"""Descriptive error analysis; no fitting or model selection from outer losses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CUR=HERE.parent/'magnitude_current/run_001'
c=json.loads((CUR/'config.json').read_text());d=pd.read_parquet(CUR/'predictions.parquet')
features={'volatility_4h':'F4Z__volatility__realized_variance_level_rz__4h__value','flow_1h':'F4Z__flow__flow_activity_percentile__1h__value'}
x=pd.concat([pd.read_parquet(ROOT/p['X'],columns=list(features.values())) for p in c['data_parts']]).sort_index().replace([np.inf,-np.inf],np.nan)
y=pd.concat([pd.read_parquet(ROOT/p['y'],columns=[c['label_end_column'],c['target_column']]) for p in c['data_parts']]).sort_index()
np.testing.assert_array_equal(d.actual,y.loc[d.index,c['target_column']])
rows=[];thresholds=[]
def add(group,period,frame):
 for model in ['pair','forest','prediction']:
  error=frame[model]-frame.actual
  rows.append({'group':group,'period':str(period),'model':model,'rows':len(frame),'MAE':float(error.abs().mean()),'mean_error':float(error.mean()),'median_error':float(error.median()),'underprediction_fraction':float((error<0).mean()),'delta_MAE_vs_pair':float((error.abs()-(frame.pair-frame.actual).abs()).mean())})
add('overall','all',d)
for fold,g in d.groupby('fold'):add('fold',fold,g)
for month,g in d.groupby(d.index.strftime('%Y-%m')):add('month',month,g)
for name,feature in features.items():
 categories=pd.Series(index=d.index,dtype=object)
 for fold,(start,end) in enumerate(c['validation_windows'],1):
  train=y.index[y[c['label_end_column']]<pd.Timestamp(start)];valid=d.index[d.fold==fold]
  q=x.loc[train,feature].quantile([1/3,2/3]).to_numpy();assert np.isfinite(q).all()
  v=x.loc[valid,feature];categories.loc[valid]=np.where(v.isna(),'missing',np.where(v<=q[0],'low',np.where(v<=q[1],'middle','high')))
  thresholds.append({'condition':name,'fold':fold,'training_lower_tertile':float(q[0]),'training_upper_tertile':float(q[1]),'training_rows':len(train)})
  for category,g in d.loc[valid].groupby(categories.loc[valid]):add(name+'_by_fold',str(fold)+'/'+category,g)
 for category,g in d.groupby(categories):add(name,category,g)
out=HERE/'diagnostics';out.mkdir(exist_ok=True)
for name in ['error_metrics.csv','thresholds.json']:
 if (out/name).exists():(out/name).read_text()
pd.DataFrame(rows).to_csv(out/'error_metrics.csv',index=False)
(out/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
print(pd.DataFrame(rows).query("group in ['overall','volatility_4h','flow_1h']").to_string(index=False))
