from pathlib import Path
import sys,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
sys.path.insert(0,str(PACKAGE/'submissions/member_02'))
import rf_walk_forward as old
xtr,ytr,xd,yd=old.data()
x=pd.concat([xtr,xd]).sort_index(); y=pd.concat([ytr,yd]).sort_index()
last=xd.index.max().normalize().replace(day=1)
months=pd.date_range(last-pd.DateOffset(months=2),last,freq='MS')
mapping=json.loads((ROOT/'Report/ic_rf_sources/magnitude_feature_mapping.json').read_text())
cols=[r['column'] for r in mapping if r['mapped']]
records=[]
for month in months:
    train=y.index[(y.index>=month-pd.DateOffset(months=3))&(y.label_end<month)]
    test=xd.index[(xd.index>=month)&(xd.index<month+pd.DateOffset(months=1))]
    admitted=x.loc[train].notna().sum().ge(1000)&x.loc[train].nunique().ge(2)
    records.append(dict(month=str(month),train_rows=len(train),test_rows=len(test),admitted=int(admitted.sum()),report_admitted=[c for c in cols if admitted[c]]))
print(json.dumps(records,indent=2))
out=ROOT/'Report/ic_rf_sources'
(out/'nested_preflight.json').write_text(json.dumps(records,indent=2))
print('Target:',y.columns.tolist())
print('Raw families:',x.columns.str.split('__').str[1].value_counts().to_dict())
