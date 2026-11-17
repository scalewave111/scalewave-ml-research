"""Fixed retrospective diagnostics and prior-quarter-only calibration. No base-model fits."""
from pathlib import Path
import json, hashlib, importlib.metadata, shutil
import numpy as np
import pandas as pd
import joblib
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'experiments/baseline_single_feature_v6'
OUT=HERE/'experiments/vol_ratio_calibration_diagnostics_v1'
NAMES=['baseline','vol_ratio','half_offset','full_offset','scale_only']

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')

def parameters(v):
    # L1-optimal offset and positive multiplicative factor, estimated ONLY on prior data.
    p=v.vol_ratio.to_numpy(); y=v.truth.to_numpy()
    offset=float(np.median(y-p))
    valid=p>1e-12
    ratio=y[valid]/p[valid]; weights=p[valid]
    order=np.argsort(ratio); ratio=ratio[order]; weights=weights[order]
    scale=float(ratio[np.searchsorted(np.cumsum(weights),weights.sum()/2)]) if len(ratio) else 1.
    return offset,max(scale,1e-6)

def run():
    OUT.mkdir(exist_ok=False)
    contract=dict(primary='half_offset',comparison=NAMES,
        calibration='For each outer window: median(truth - candidate prediction) on complete purged prior validation quarter; primary adds half this offset. Secondary full_offset adds full median residual; scale_only multiplies by prediction-weighted median of truth/prediction. Clip predictions at zero.',
        selection='No selection among calibration methods. All three fixed before computation; no test-driven tuning.',
        groups='Per-model low/medium/high cutoffs from prior validation prediction tertiles; frozen for following test window. Ties remain together.',
        extremes='Actual magnitude >500 log bps and >prior validation truth P90/P95. Outcome-conditioned diagnostic, not an implementable event filter.',
        target='BTCUSDT absolute 48-hour endpoint log-return magnitude in bps',
        features='Original liquidation pair plus 4h/24h volatility ratio; original models and weights unchanged.',
        limitation='Historical exposed evaluation; validation was also used to choose original blend weights. Calibration transfer from inner fit to refitted outer model is evaluated, not assumed. No new holdout.',
        uncertainty='5000 paired 7-day block bootstrap replicates, seed 20261104; descriptive and unadjusted for previous searches.')
    save('CONTRACT.json',contract)
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    inputs=[SOURCE/n for n in ['validation_predictions.csv','predictions.csv','prepared.joblib','SELECTIONS.json','AUDIT.json']]
    hashes={str(p):q.sha(p) for p in inputs}
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    save('MANIFEST.json',dict(inputs=hashes,instructions={str(p):q.sha(p) for p in docs},code=q.sha(__file__),versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']}))
    v=pd.read_csv(SOURCE/'validation_predictions.csv',index_col=0,parse_dates=True)
    p=pd.read_csv(SOURCE/'predictions.csv',index_col=0,parse_dates=True)
    x,y,dev,ref=joblib.load(SOURCE/'prepared.joblib')
    assert p.index.equals(dev) and p.index.is_unique and len(p)==18481
    assert json.loads((SOURCE/'AUDIT.json').read_text())['status']=='PASS'
    np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
    np.testing.assert_allclose(p.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
    parts=[]; decisions=[]; windows=[]; group_rows=[]; extreme_rows=[]
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        vv=v.loc[pd.to_datetime(v.issue,utc=True)==issue].copy()
        pp=p.loc[te,['baseline','vol_ratio','truth','issue']].copy()
        assert vv.index.equals(vk) and y.loc[vk,'label_end'].max()<issue
        assert y.loc[ik,'label_end'].max()<inner
        np.testing.assert_allclose(vv.truth,y.loc[vk,'magnitude_log_bps'],rtol=0,atol=1e-8)
        offset,scale=parameters(vv)
        d=dict(issue=str(issue),offset=offset,scale=scale,calibration_rows=len(vv),last_calibration_label_end=str(y.loc[vk,'label_end'].max()),group_cutoffs={},extreme_cutoffs={})
        for df in [vv,pp]:
            df['half_offset']=np.maximum(0,df.vol_ratio+.5*offset)
            df['full_offset']=np.maximum(0,df.vol_ratio+offset)
            df['scale_only']=df.vol_ratio*scale
        for name in NAMES:
            windows.append(dict(issue=str(issue),model=name,**q.b.metric(pp[name],pp.truth)))
            cuts=np.quantile(vv[name],[1/3,2/3]);d['group_cutoffs'][name]=cuts.tolist()
            pp[name+'_group']=np.searchsorted(cuts,pp[name],side='left')
            for g,label in enumerate(['low','medium','high']):
                z=pp.loc[pp[name+'_group']==g]
                if len(z):group_rows.append(dict(issue=str(issue),model=name,group=label,rows=len(z),mean_prediction=z[name].mean(),mean_actual=z.truth.mean(),median_actual=z.truth.median(),mae=abs(z[name]-z.truth).mean(),event_rate=(z.truth>500).mean()))
        for tag,threshold in [('above_500',500.),('above_prior_p90',float(vv.truth.quantile(.9))),('above_prior_p95',float(vv.truth.quantile(.95)))]:
            d['extreme_cutoffs'][tag]=threshold;pp[tag]=pp.truth>threshold
            for name in NAMES:
                z=pp.loc[pp[tag]]
                if len(z):extreme_rows.append(dict(issue=str(issue),subset=tag,threshold=threshold,model=name,rows=len(z),mae=abs(z[name]-z.truth).mean(),mean_actual=z.truth.mean(),mean_prediction=z[name].mean(),bias=(z[name]-z.truth).mean()))
        decisions.append(d);parts.append(pp)
    # Parameters are frozen before aggregate test scoring. No adaptive retries.
    save('CALIBRATION_PARAMETERS.json',decisions)
    out=pd.concat(parts);assert out.index.equals(dev)
    out.to_csv(OUT/'predictions.csv',index_label='decision_time')
    scores=pd.DataFrame([dict(model=n,**q.b.metric(out[n],out.truth)) for n in NAMES])
    scores.to_csv(OUT/'scores.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'window_scores.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'group_window_scores.csv',index=False)
    pd.DataFrame(extreme_rows).to_csv(OUT/'extreme_window_scores.csv',index=False)
    groups=[];extremes=[]
    for n in NAMES:
        for g,label in enumerate(['low','medium','high']):
            z=out.loc[out[n+'_group']==g]
            if len(z):groups.append(dict(model=n,group=label,rows=len(z),mean_prediction=z[n].mean(),mean_actual=z.truth.mean(),median_actual=z.truth.median(),mae=abs(z[n]-z.truth).mean(),event_rate=(z.truth>500).mean()))
        for tag in ['above_500','above_prior_p90','above_prior_p95']:
            z=out.loc[out[tag]]
            extremes.append(dict(model=n,subset=tag,rows=len(z),mae=abs(z[n]-z.truth).mean(),mean_prediction=z[n].mean(),mean_actual=z.truth.mean(),bias=(z[n]-z.truth).mean()))
    groups=pd.DataFrame(groups);extremes=pd.DataFrame(extremes)
    groups.to_csv(OUT/'prediction_groups.csv',index=False);extremes.to_csv(OUT/'extreme_scores.csv',index=False)
    uncertainty=[]
    block=(out.index.asi8-out.index.asi8.min())//pd.Timedelta(days=7).value
    for n in NAMES[1:]:
        for reference in ['baseline','vol_ratio']:
            if n==reference:continue
            delta=abs(out[n]-out.truth)-abs(out[reference]-out.truth)
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count'])
            ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1)
            lo,hi=np.quantile(boot,[.025,.975])
            uncertainty.append(dict(model=n,reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi,blocks=len(b)))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    replay=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    for d in decisions:
        z=replay.loc[pd.to_datetime(replay.issue,utc=True)==pd.Timestamp(d['issue'])]
        np.testing.assert_allclose(z.half_offset,np.maximum(0,z.vol_ratio+.5*d['offset']),rtol=0,atol=1e-8)
        np.testing.assert_allclose(z.full_offset,np.maximum(0,z.vol_ratio+d['offset']),rtol=0,atol=1e-8)
        np.testing.assert_allclose(z.scale_only,z.vol_ratio*d['scale'],rtol=0,atol=1e-8)
    assert all(q.sha(Path(path))==digest for path,digest in hashes.items())
    save('AUDIT.json',dict(status='PASS',strict_label_purges=True,original_test_rows=True,baseline_reproduced=True,calibration_replayed=True,input_files_unchanged=True,base_model_training=False,rows=len(out)))
    report='Historical diagnostics and calibration (executed on real saved predictions)\n\n'+scores.to_string(index=False)+'\n\nPrediction groups (cutoffs from prior validation):\n'+groups.query("model in ['baseline','vol_ratio']").to_string(index=False)+'\n\nExtreme moves:\n'+extremes.to_string(index=False)+'\n\nCalibration parameters:\n'+pd.DataFrame(decisions)[['issue','offset','scale']].to_string(index=False)+'\n\n'+json.dumps(contract,indent=2)
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8')
    save('COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',scores=scores.to_dict('records'),primary='half_offset',fresh_holdout=False))
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Volatility ratio: diagnostics and calibration\nHistorical walk-forward evaluation; no fresh holdout. The primary test is half-offset calibration. Group cutoffs and calibration parameters use only each earlier purged validation quarter.']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot = Path('experiments/vol_ratio_calibration_diagnostics_v1')\nfor name in ['scores', 'window_scores', 'prediction_groups', 'extreme_scores', 'uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name + '.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    (HERE/'BTC_48h_Vol_Ratio_Calibration.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(scores.to_string(index=False),flush=True)
    print(groups.query("model=='vol_ratio'").to_string(index=False),flush=True)
    print(extremes.query("subset=='above_500'").to_string(index=False),flush=True)
    print(pd.DataFrame(decisions)[['issue','offset','scale']].to_string(index=False),flush=True)

if __name__=='__main__':run()
