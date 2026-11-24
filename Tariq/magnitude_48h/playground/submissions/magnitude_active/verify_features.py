"""Independent artifact replay and training-only selection checks; no fits."""
from pathlib import Path
import json
import hashlib
from scipy.stats import spearmanr
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'run_001'
config=json.loads((OUT/'config.json').read_text())
receipt=json.loads((HERE/'preflight/verification.json').read_text())
for path,sha in receipt['hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
features=list(dict.fromkeys(config['anchor_features']+list(config['additions'].values())))
X=pd.concat([pd.read_parquet(ROOT/p['X'],columns=features) for p in config['data_parts']]).sort_index().replace([np.inf,-np.inf],np.nan)
y=pd.concat([pd.read_parquet(ROOT/p['y'],columns=[config['label_end_column'],config['target_column']])
             for p in config['data_parts']]).sort_index()
assert (y[config['label_end_column']]-y.index == pd.Timedelta(minutes=config['horizon_minutes'])).all()
pred=pd.read_parquet(OUT/'predictions.parquet')
base=pd.read_parquet(ROOT/config['benchmark_predictions'])
assert pred.index.equals(base.index) and pred.index.is_unique
np.testing.assert_array_equal(pred.actual,y.loc[pred.index,config['target_column']])
np.testing.assert_array_equal(pred.incumbent_hybrid,base[config['benchmark_column']])
np.testing.assert_array_equal(pred.refitted_pair,base.refitted_pair)


def matrix(train,valid,metadata):
    columns=[]
    for feature,a in metadata.items():
        values=X.loc[train,feature]
        assert values.count()>=config['minimum_valid_training'] and values.nunique()>=2
        assert a['median']==values.median()
        filled=values.fillna(a['median'])
        assert abs(a['center']-filled.mean())<1e-12
        assert abs(a['scale']-filled.std(ddof=0))<1e-12
        v=X.loc[valid,feature]
        columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
    return np.column_stack(columns)


largest_error=0.0
for fold,(start,end) in enumerate(config['validation_windows'],1):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    inner_start=start-pd.DateOffset(months=config['inner_validation_months'])
    ot=y.index[y[config['label_end_column']]<start]
    ov=y.index[(y.index>=start)&(y.index<end)]
    it=y.index[y[config['label_end_column']]<inner_start]
    iv=ot[ot>=inner_start]
    assert y.loc[it,config['label_end_column']].max()<iv.min()
    assert y.loc[ot,config['label_end_column']].max()<ov.min()
    assert y.loc[iv,config['label_end_column']].max()<ov.min()
    inner_pair=json.loads((ROOT/config['anchor_profile']/f'fold_{fold}_inner.json').read_text())
    pair=np.maximum(inner_pair['intercept']+matrix(it,iv,inner_pair['preprocessing'])@inner_pair['coefficients'],0)
    for name,feature in config['additions'].items():
        card=json.loads((OUT/f'fold_{fold}_{name}_inner.json').read_text())
        assert card['features']==config['anchor_features']+[feature]
        assert card['training_rows']==len(it)
        A=matrix(it,iv,card['preprocessing'])
        model=joblib.load(OUT/f'fold_{fold}_{name}_inner.joblib')
        for key,value in config['forest'].items():
            assert model.get_params()[key]==value
        forest=np.maximum(model.predict(A),0)
        inner=pd.read_parquet(OUT/f'fold_{fold}_{name}_inner_predictions.parquet')
        assert inner.index.equals(iv)
        np.testing.assert_array_equal(inner.actual,y.loc[iv,config['target_column']])
        np.testing.assert_allclose(inner.pair,pair,rtol=0,atol=1e-10)
        np.testing.assert_allclose(inner.forest,forest,rtol=0,atol=1e-10)
        scores=[(float(np.abs(inner.actual-((1-w)*pair+w*forest)).mean()),w) for w in config['forest_weights']]
        chosen=min(scores)[1]
        assert chosen==card['chosen_weight']
        outer=json.loads((OUT/f'fold_{fold}_{name}_outer.json').read_text())
        assert outer['training_rows']==len(ot) and outer['weight']==chosen
        B=matrix(ot,ov,outer['preprocessing'])
        model=joblib.load(OUT/f'fold_{fold}_{name}_outer.joblib')
        for key,value in config['forest'].items():
            assert model.get_params()[key]==value
        forest=np.maximum(model.predict(B),0)
        tree_mean=np.mean([tree.predict(B) for tree in model.estimators_],axis=0)
        np.testing.assert_allclose(forest,tree_mean,rtol=0,atol=1e-10)
        replay=(1-chosen)*base.loc[ov,'refitted_pair'].to_numpy()+chosen*forest
        error=float(np.max(np.abs(replay-pred.loc[ov,name])))
        largest_error=max(largest_error,error)
        assert error<1e-10
report=json.loads((OUT/'RESULT.json').read_text())
for name,mae in report['pooled_MAE'].items():
    assert abs(np.abs(pred.actual-pred[name]).mean()-mae)<1e-10
for name,ic in report['pooled_Spearman_IC'].items():
    assert abs(float(spearmanr(pred.actual,pred[name]).statistic)-ic)<1e-12
    assert abs(float(np.corrcoef(pred.actual.rank(),pred[name].rank())[0,1])-ic)<1e-12
ledger=json.loads((OUT/'fit_ledger.json').read_text())
assert len(ledger)==config['new_fit_budget'] and all(r['status']=='completed' for r in ledger)
receipt={'status':'PASS','horizon_minutes':config['horizon_minutes'],'rows':len(pred),
    'max_prediction_replay_error':largest_error,'new_verification_fits':0,
    'checks':['target horizon and exact row alignment','inner/outer label-end purges',
              'fold-specific feature admission and preprocessing','fixed forest parameters',
              'training-only blend selection','all saved inner/outer forest replays',
              'independent individual-tree averaging','independent MAE recomputation'],
    'fresh_data_accessed':False}
with (OUT/'verification.json').open('x') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
