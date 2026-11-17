import json,hashlib
import numpy as np,pandas as pd,joblib
import tree_magnitude_filter as m
import canonical_return_features as c
from threadpoolctl import threadpool_limits
def main():
 raw,oy,dev,report=m.old.load();r=c.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet');y=c.make_labels(r,raw.index,72);x=raw.loc[y.index]
 out=m.OUT/'72h';issues=[]
 for sel in json.loads((out/'selections.json').read_text()):
  issue=pd.Timestamp(sel['issue']);folder=out/issue.strftime('%Y%m%d');frame=pd.concat([x,m.engineered(x,sel['engineering_parents'])],axis=1);s=m.FoldSearch(frame,y,issue,72,folder)
  def replay(kind,cols):
   key=(kind,tuple(sorted(cols)));slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16]
   pp=[]
   for i,(_,tr,va) in enumerate(s.folds):
    a=joblib.load(folder/f'{slug}_inner{i}.joblib')
    # One prediction worker fixes RF accumulation ordering for diagnosis.
    if a['model'].__class__.__name__=='RandomForestRegressor':a['model'].n_jobs=1
    pp.append(pd.Series(m.predict(a,frame.loc[va]),index=va))
   p=pd.concat(pp);return dict(p=p,metrics=m.metrics(p,s.truth))
  for j,rec in enumerate(json.loads((folder/'contributions.json').read_text())):
   if rec['stage']=='drop_ablation':continue
   b=replay(rec['kind'],rec['reference_columns']);a=replay(rec['kind'],rec['reference_columns']+[rec['feature']]);g=s.gate(a,b)
   if g['monthly_mae_wins']!=rec['monthly_mae_wins'] or g['pass_gate']!=rec['pass_gate']:
    changes=[]
    for month in sorted(set(s.truth.index.strftime('%Y-%m'))):
     ix=s.truth.index.strftime('%Y-%m')==month;changes.append(float(abs(b['p'][ix]-s.truth[ix]).mean()-abs(a['p'][ix]-s.truth[ix]).mean()))
    issues.append(dict(issue=str(issue),record=j,feature=rec['feature'],saved_gate=rec['pass_gate'],replayed_gate=g['pass_gate'],saved_wins=rec['monthly_mae_wins'],replayed_wins=g['monthly_mae_wins'],monthly_mae_gains=changes,overall_mae_gain=g['mae_gain'],ic_gain=g['ic_gain']))
 m.dump(out/'AUDIT_NUMERICAL_DIAGNOSTIC.json',issues);print(json.dumps(issues,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
