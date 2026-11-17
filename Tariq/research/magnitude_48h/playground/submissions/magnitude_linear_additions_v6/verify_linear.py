"""Replay all candidate and reused RF artifacts independently; no training."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'run_001'
c=json.loads((OUT/'config.json').read_text())
receipt=json.loads((HERE/'preflight/verification.json').read_text())
for path,sha in receipt['hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
features=list(dict.fromkeys(c['anchor_features']+list(c['additions'].values())))
X=pd.concat([pd.read_parquet(ROOT/p['X'],columns=features) for p in c['data_parts']]).sort_index().replace([np.inf,-np.inf],np.nan)
y=pd.concat([pd.read_parquet(ROOT/p['y'],columns=[c['label_end_column'],c['target_column']]) for p in c['data_parts']]).sort_index()
assert X.index.equals(y.index) and X.index.is_unique
assert (y[c['label_end_column']]-y.index==pd.Timedelta(minutes=c['horizon_minutes'])).all()
pred=pd.read_parquet(OUT/'predictions.parquet')
base=pd.read_parquet(ROOT/c['benchmark_predictions'])
assert pred.index.equals(base.index) and pred.index.is_unique
np.testing.assert_array_equal(pred.actual,y.loc[pred.index,c['target_column']])
np.testing.assert_array_equal(pred.incumbent_hybrid,base[c['benchmark_column']])
np.testing.assert_array_equal(pred.refitted_pair,base.refitted_pair)

def matrix(train,valid,metadata):
    columns=[]
    for feature,a in metadata.items():
        values=X.loc[train,feature]
        assert values.count()>=c['minimum_valid_training'] and values.nunique()>=2
        assert a['median']==values.median()
        filled=values.fillna(a['median'])
        assert abs(a['center']-filled.mean())<1e-12
        assert abs(a['scale']-(float(filled.std(ddof=0)) or 1.0))<1e-12
        v=X.loc[valid,feature]
        columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
    return np.column_stack(columns)

largest_error=0.0
for fold,(start,end) in enumerate(c['validation_windows'],1):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    inner_start=start-pd.DateOffset(months=c['inner_validation_months'])
    ot=y.index[y[c['label_end_column']]<start]
    ov=y.index[(y.index>=start)&(y.index<end)]
    it=y.index[y[c['label_end_column']]<inner_start]
    iv=ot[ot>=inner_start]
    assert y.loc[it,c['label_end_column']].max()<iv.min()
    assert y.loc[ot,c['label_end_column']].max()<ov.min()
    assert y.loc[iv,c['label_end_column']].max()<ov.min()
    forests={}
    for role,train,valid in [('inner',it,iv),('outer',ot,ov)]:
        card=json.loads((ROOT/c['anchor_profile']/f'fold_{fold}_{role}.json').read_text())
        assert list(card['preprocessing'])==c['anchor_features']
        A=matrix(train,valid,card['preprocessing'])
        model=joblib.load(ROOT/c['anchor_profile']/f'fold_{fold}_{role}_forest.joblib')
        for key,value in c['forest'].items():
            assert model.get_params()[key]==value
        forests[role]=np.maximum(np.mean([tree.predict(A) for tree in model.estimators_],axis=0),0)
        np.testing.assert_allclose(forests[role],np.maximum(model.predict(A),0),rtol=0,atol=1e-10)
    cached=pd.read_parquet(ROOT/c['anchor_profile']/f'fold_{fold}_inner_predictions.parquet')
    assert cached.index.equals(iv)
    np.testing.assert_array_equal(cached.actual,y.loc[iv,c['target_column']])
    np.testing.assert_allclose(forests['inner'],cached.forest,rtol=0,atol=1e-10)
    np.testing.assert_allclose(forests['outer'],base.loc[ov,'forest_only'],rtol=0,atol=1e-10)
    for name,feature in c['additions'].items():
        linear={};cards={}
        for role,train,valid in [('inner',it,iv),('outer',ot,ov)]:
            card=json.loads((OUT/f'fold_{fold}_{name}_{role}.json').read_text())
            cards[role]=card
            assert card['features']==c['anchor_features']+[feature]
            assert list(card['preprocessing'])==card['features']
            assert card['training_rows']==len(train)
            A=matrix(train,valid,card['preprocessing'])
            model=joblib.load(OUT/f'fold_{fold}_{name}_{role}.joblib')
            for key,value in c['linear'].items():
                assert model.get_params()[key]==value
            linear[role]=np.maximum(card['intercept']+A@card['coefficients'],0)
            np.testing.assert_allclose(linear[role],np.maximum(model.predict(A),0),rtol=0,atol=1e-10)
        inner=pd.read_parquet(OUT/f'fold_{fold}_{name}_inner_predictions.parquet')
        assert inner.index.equals(iv)
        np.testing.assert_array_equal(inner.actual,y.loc[iv,c['target_column']])
        np.testing.assert_allclose(inner.linear,linear['inner'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(inner.forest,forests['inner'],rtol=0,atol=1e-10)
        scores=[(float(np.abs(inner.actual-((1-w)*linear['inner']+w*forests['inner'])).mean()),w) for w in c['forest_weights']]
        for (mae,w),saved in zip(scores,cards['inner']['scores']):
            assert w==saved['weight'] and abs(mae-saved['MAE'])<1e-10
        chosen=min(scores)[1]
        assert chosen==cards['inner']['chosen_weight']==cards['outer']['weight']
        replay=(1-chosen)*linear['outer']+chosen*forests['outer']
        error=float(np.max(np.abs(replay-pred.loc[ov,name])))
        largest_error=max(largest_error,error)
        assert error<1e-10
        frame=pd.read_parquet(OUT/f'fold_{fold}_{name}_predictions.parquet')
        assert frame.index.equals(ov) and (frame.weight==chosen).all()
        np.testing.assert_allclose(frame.prediction,replay,rtol=0,atol=1e-10)

report=json.loads((OUT/'RESULT.json').read_text())
for name,mae in report['pooled_MAE'].items():
    assert abs(np.abs(pred.actual-pred[name]).mean()-mae)<1e-10
    rank=float(spearmanr(pred.actual,pred[name]).statistic)
    independent=float(np.corrcoef(pred.actual.rank(method='average'),pred[name].rank(method='average'))[0,1])
    assert abs(rank-independent)<1e-12
    assert abs(rank-report['pooled_Spearman_IC'][name])<1e-12
    assert abs(float(pearsonr(pred.actual,pred[name]).statistic)-report['pooled_Pearson_IC'][name])<1e-12
assert abs(report['pooled_MAE']['incumbent_hybrid']-c['benchmark_mae'])<1e-10
ledger=json.loads((OUT/'fit_ledger.json').read_text())
assert len(ledger)==c['new_fit_budget'] and all(r['status']=='completed' for r in ledger)
result={'status':'PASS','rows':len(pred),'horizon_minutes':c['horizon_minutes'],
    'max_prediction_replay_error':largest_error,'new_verification_fits':0,
    'checks':['source hashes','identical target and scoring rows','strict inner and outer label purges',
    'training-only transformations','linear parameters and coefficient replay','unchanged RF individual-tree replay',
    'earlier-inner-MAE blend selection','independent pooled MAE and rank IC recomputation'],
    'fresh_data_accessed':False}
with (OUT/'verification.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
