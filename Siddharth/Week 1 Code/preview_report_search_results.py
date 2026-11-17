"""Score already-frozen primary fits; never fit or select a model."""
from pathlib import Path
import json,time
import joblib
import numpy as np
import pandas as pd
import torch
import search_report_features as run


def main():
    waiting=[];sources=[]
    for method in ['rf','mlp']:
        for issue in run.q.STARTS:
            folder=run.OUT/method/f'{issue:%Y%m%d}'
            selection=folder/'SELECTION.json'
            if not selection.exists():waiting.append(str(folder.relative_to(run.OUT)));continue
            s=json.loads(selection.read_text())
            best=min(s['finalists'],key=lambda r:(r['mae'],len(r['cols']),r['id']))
            assert best==s['selected'] and s['selected_before_outer_predictions']
            path=folder/'full'/best['id']/'outer/bundle.joblib'
            if not path.exists() or time.time()-path.stat().st_mtime<5:
                waiting.append(str(folder.relative_to(run.OUT)));continue
            sources.append((method,issue,s,path))
    ready=[method for method in ['rf','mlp'] if sum(name==method for name,_,_,_ in sources)==4]
    if not ready:
        print(json.dumps(dict(status='PRIMARY_FITS_NOT_ALL_READY',waiting=waiting)));return
    x,y,dev,ref,features=joblib.load(run.OUT/'prepared.joblib')
    result=pd.DataFrame(dict(truth=ref.truth,baseline=ref.report_hybrid),index=dev)
    old=pd.read_csv(run.HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True)
    result['yesterday_best_rf']=old.vol_ratio
    for method in ready:
        pieces=[]
        for name,issue,s,path in sources:
            if name!=method:continue
            _,_,_,keys,te=run.q.windows(y,dev,issue)
            a=joblib.load(path)
            assert a['keys'].equals(keys) and a['cutoff']==issue and a['cols']==s['selected']['cols']
            assert y.loc[keys,'label_end'].max()<issue
            pieces.append(pd.Series(run.predict(a,x,te),index=te))
        pred=pd.concat(pieces);assert pred.index.equals(dev);result[method]=pred
    scores=[dict(model=c,**run.metrics(result[c],result.truth)) for c in result.columns if c!='truth']
    assert len(result)==18481 and np.isfinite(result.to_numpy()).all()
    pd.DataFrame(scores).to_csv(run.OUT/'PRIMARY_PREVIEW_scores.csv',index=False)
    result.to_csv(run.OUT/'PRIMARY_PREVIEW_predictions.csv',index_label='decision_time')
    run.dump(run.OUT/'PRIMARY_PREVIEW.json',dict(status='AVAILABLE_PRIMARY_FITS_COMPLETE_DIAGNOSTICS_AND_FINAL_AUDIT_PENDING',models_ready=ready,waiting=waiting,
        frozen_selections_verified=True,rows=len(result),scores=scores,
        outer_model_hashes={str(path):run.q.sha(path) for _,_,_,path in sources},
        warning='Provisional saved-prediction scores. Do not treat as final audited results. No selection or fitting performed.'))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(1)
    with run.threadpool_limits(limits=1):main()
