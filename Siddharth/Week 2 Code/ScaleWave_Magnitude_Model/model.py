"""Portable historical BTCUSDT 48h hybrid replay. No training on import."""
from pathlib import Path
import argparse, hashlib, json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HOME=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def transform(X,prep):
    columns=[]
    for f,a in prep.items():
        v=X[f].replace([np.inf,-np.inf],np.nan)
        columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'],v.isna().astype(float)])
    return np.column_stack(columns)

def predict(X, home=HOME):
    c=read(home/'config.json')
    features=c['anchor_features']+list(c['additions'].values())
    missing=set(features)-set(X.columns)
    if missing:raise ValueError(f'Missing feature columns: {sorted(missing)}')
    if not isinstance(X.index,pd.DatetimeIndex) or X.index.tz is None or not X.index.is_unique:
        raise ValueError('Require unique timezone-aware DatetimeIndex of forecast origins.')
    X=X.copy();X.index=X.index.tz_convert('UTC');X=X.sort_index()
    result=pd.DataFrame(index=X.index,columns=['fold','pair','forest','forest_weight','prediction'],dtype=float)
    for fold,(start,end) in enumerate(c['validation_windows'],1):
        idx=X.index[(X.index>=pd.Timestamp(start))&(X.index<pd.Timestamp(end))]
        if not len(idx):continue
        card=read(home/f'models/fold_{fold}_selected.json');sel=card['selection']
        pair=read(home/f'models/fold_{fold}_pair.json')
        anchor=np.maximum(pair['intercept']+transform(X.loc[idx],pair['preprocessing'])@pair['coefficients'],0)
        A=transform(X.loc[idx],card['preprocessing'])
        models=joblib.load(home/f'models/fold_{fold}_selected.joblib')
        assert len(models)==len(c['seeds'])==3
        predictions=[]
        for model,seed in zip(models,c['seeds']):
            assert model.random_state==seed and len(model.estimators_)==128
            # Individual-tree replay matches the original verification path.
            predictions.append(np.maximum(np.mean([t.predict(A) for t in model.estimators_],axis=0),0))
        forest=np.mean(predictions,axis=0);w=sel['weight']
        result.loc[idx,:]=np.column_stack([np.full(len(idx),fold),anchor,forest,np.full(len(idx),w),(1-w)*anchor+w*forest])
    if result.isna().any().any():raise ValueError('Origins outside the four historical evaluation windows. These are historical snapshots, not a current live deployment.')
    return result

def verify(home=HOME):
    for name,expected in read(home/'manifest.json').items():
        assert hashlib.sha256((home/name).read_bytes()).hexdigest()==expected,name
    X=pd.read_parquet(home/'replay_inputs.parquet');expected=pd.read_parquet(home/'expected.parquet')
    actual=predict(X,home);assert actual.index.equals(expected.index)
    for col in ['pair','forest','prediction','fold']:
        np.testing.assert_allclose(actual[col],expected[col],atol=1e-10,rtol=0)
    mae=float((expected.actual-actual.prediction).abs().mean())
    ic=float(spearmanr(expected.actual,actual.prediction).statistic)
    benchmark=read(home/'evidence/RESULT.json')['metrics']['prediction']
    assert abs(mae-benchmark['MAE'])<1e-10 and abs(ic-benchmark['Spearman_IC'])<1e-12
    print(json.dumps({'status':'PASS','rows':len(actual),'MAE':mae,'Spearman_IC':ic,'training_fits':0}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.verify:verify()
    elif args.input and args.output:
        if args.output.exists():raise FileExistsError('Choose a new output path; refusing to overwrite.')
        predict(pd.read_parquet(args.input)).to_parquet(args.output)
    else:parser.error('Use --verify, or --input features.parquet --output predictions.parquet')
