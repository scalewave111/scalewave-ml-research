"""Inspect timing evidence and stress frozen incumbent with older snapshots; no fits."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODEL = ROOT/'submissions/magnitude_current/run_001'
c = json.loads((MODEL/'config.json').read_text())
features = list(dict.fromkeys(c['anchor_features']+list(c['additions'].values())))
X = pd.concat([pd.read_parquet(ROOT/p['X'],columns=features) for p in c['data_parts']]).sort_index()
y = pd.concat([pd.read_parquet(ROOT/p['y']) for p in c['data_parts']]).sort_index()
saved = pd.read_parquet(MODEL/'predictions.parquet')
assert X.index.equals(y.index) and X.index.is_unique
assert (y.label_end-y.index == pd.Timedelta(hours=48)).all()
schema_evidence = {}
for part in c['data_parts']:
    f = ROOT/part['X']; schema = pq.read_schema(f)
    schema_evidence[part['X']] = {'columns':len(schema.names),
        'timestamp_fields':[str(field) for field in schema if 'timestamp' in str(field.type)],
        'availability_fields':[n for n in schema.names if any(s in n.lower() for s in ['available_after','received_at','published_at','computed_at','event_time'])],
        'metadata_keys':[k.decode() for k in (schema.metadata or {})]}

def design(x,pre):
    columns=[]
    for f,a in pre.items():
        v=x[f]; columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
    return np.column_stack(columns)

folds=[]; all_predictions={delay:[] for delay in [0,30,60,120]}; lookup_missing={delay:0 for delay in all_predictions}
for fold,(start,end) in enumerate(c['validation_windows'],1):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    idx=y.index[(y.index>=start)&(y.index<end)]
    training=y.index[y.label_end<start]
    assert y.loc[training,'label_end'].max()<idx.min()
    card=json.loads((MODEL/f'fold_{fold}_selected.json').read_text())
    pair=json.loads((ROOT/c['outer_pair_profile']/f'fold_{fold}_pair.json').read_text())
    models=joblib.load(MODEL/f'fold_{fold}_selected.joblib')
    models=models if isinstance(models,list) else [models]
    for f,a in card['preprocessing'].items():
        v=X.loc[training,f]; filled=v.fillna(v.median())
        assert abs(v.median()-a['median'])<1e-12
        assert abs(filled.mean()-a['center'])<1e-12
        assert abs(filled.std(ddof=0)-a['scale'])<1e-12
    for delay in all_predictions:
        requested=idx-pd.Timedelta(minutes=delay)
        lookup_missing[delay]+=int((~requested.isin(X.index)).sum())
        x=X.reindex(requested).copy();x.index=idx
        a=design(x,card['preprocessing'])
        forest=np.mean([np.maximum(np.mean([tree.predict(a) for tree in model.estimators_],axis=0),0) for model in models],axis=0)
        anchor=np.maximum(pair['intercept']+design(x,pair['preprocessing'])@pair['coefficients'],0)
        w=card['selection']['weight']; prediction=(1-w)*anchor+w*forest
        all_predictions[delay].append(pd.Series(prediction,index=idx))
        folds.append({'delay_minutes':delay,'fold':fold,'rows':len(idx),'MAE':float(np.abs(y.loc[idx,c['target_column']]-prediction).mean())})
rows=[]
for delay,parts in all_predictions.items():
    prediction=pd.concat(parts)
    assert prediction.index.equals(saved.index)
    if delay==0:np.testing.assert_allclose(prediction,saved.prediction,rtol=0,atol=1e-10)
    rows.append({'delay_minutes':delay,'rows':len(prediction),'MAE':float(np.abs(saved.actual-prediction).mean()),'unavailable_snapshot_rows':lookup_missing[delay]})
report={'actual_market_to_model_delay':'Not measurable from supplied prepared files: no per-feature event/publication/receipt/computation timestamps.',
    'schema_evidence':schema_evidence,'current_features':features,'training_preprocessing_verified':True,'label_purges_verified':True,
    'original_predictions_replayed':True,'sensitivity':rows,
    'limitations':'Hypothetical older-input stress test, not measured latency or a leakage test. Exact timestamp lookup; absent snapshots use saved training imputation. Models, targets, all evaluation rows and preprocessing fixed; no refitting. Thirty-minute sampling cannot test seconds or one-minute delays.',
    'new_fits':0}
(OUT/'magnitude_timing_checks.json').write_text(json.dumps(report,indent=2))
pd.DataFrame(rows).to_csv(OUT/'magnitude_delay_sensitivity.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'magnitude_delay_by_fold.csv',index=False)
print(json.dumps(report,indent=2))
