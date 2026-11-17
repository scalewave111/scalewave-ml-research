"""Recompute MAE and magnitude IC from saved forecasts; zero model fits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, pearsonr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def metrics(actual,predicted):
    actual=np.asarray(actual,dtype=float)
    predicted=np.asarray(predicted,dtype=float)
    assert actual.shape==predicted.shape and actual.ndim==1
    assert np.isfinite(actual).all() and np.isfinite(predicted).all()
    if len(actual)<2 or np.ptp(actual)==0 or np.ptp(predicted)==0:
        rank_ic=linear_ic=None
    else:
        rank_ic=float(spearmanr(predicted,actual).statistic)
        linear_ic=float(pearsonr(predicted,actual).statistic)
        # Independent rank construction verifies tie handling and coefficient.
        ranks_x=pd.Series(predicted).rank(method='average').to_numpy()
        ranks_y=pd.Series(actual).rank(method='average').to_numpy()
        assert abs(rank_ic-np.corrcoef(ranks_x,ranks_y)[0,1])<1e-12
        assert abs(linear_ic-np.corrcoef(predicted,actual)[0,1])<1e-12
    return {'rows':len(actual),'MAE':float(np.abs(actual-predicted).mean()),
            'Spearman_IC':rank_ic,'Pearson_IC':linear_ic}


if __name__=='__main__':
    registry=json.loads((ROOT/'submissions/magnitude_benchmark_registry.json').read_text())
    current=registry['current'];source=ROOT/current['prediction_file']
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==current['prediction_sha256']
    pred=pd.read_parquet(source)
    y=pd.read_parquet(ROOT/'data/y_development.parquet',columns=['magnitude_log_bps','label_end'])
    assert pred.index.equals(y.index) and pred.index.is_unique and len(pred)==registry['rows']
    np.testing.assert_array_equal(pred.actual,y.magnitude_log_bps)
    assert (y.label_end-y.index==pd.Timedelta(minutes=registry['horizon_minutes'])).all()
    names={'frozen_MP0004':'frozen_MP0004','walkforward_pair':'refitted_pair',
           'current_hybrid':current['prediction_column']}
    overall={name:metrics(pred.actual,pred[column]) for name,column in names.items()}
    assert abs(overall['current_hybrid']['MAE']-current['MAE'])<1e-12
    monthly=[];folds=[]
    for name,column in names.items():
        for month,frame in pred.groupby(pred.index.strftime('%Y-%m')):
            monthly.append({'model':name,'month':month,**metrics(frame.actual,frame[column])})
        for fold,frame in pred.groupby('fold'):
            folds.append({'model':name,'fold':int(fold),**metrics(frame.actual,frame[column])})
    out=HERE/'results';out.mkdir(exist_ok=False)
    pd.DataFrame.from_dict(overall,orient='index').rename_axis('model').to_csv(out/'overall_metrics.csv')
    pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    result={'status':'VERIFIED','definition':{
        'primary_IC':'Spearman rank correlation between predicted and actual absolute 48h endpoint log returns',
        'secondary_IC':'Pearson correlation between the same raw forecast and target values',
        'primary_definition_status':'Working convention stated to user; confirm if team requires Pearson instead',
        'population':'Pooled time-series correlation across exactly the original 18481 BTC development origins; not a cross-asset IC or mean of monthly/fold ICs',
        'ties':'Average ranks','missing':'Fail on nonfinite values; no row drops',
        'constant_forecasts':'IC undefined (null), never silently zero',
        'p_values':'Not reported: overlapping 48h labels violate independent-observation inference'},
        'horizon_minutes':registry['horizon_minutes'],'start':str(pred.index.min()),'end':str(pred.index.max()),
        'models':overall,'new_model_fits':0,'fresh_test_accessed':False,
        'source_prediction_file':current['prediction_file'],'prediction_sha256':sha,
        'source_y_sha256':hashlib.sha256((ROOT/'data/y_development.parquet').read_bytes()).hexdigest(),
        'software':{'scipy':scipy.__version__,'pandas':pd.__version__,'numpy':np.__version__},
        'verification':['exact saved forecast hash','exact target values and timestamps','independent MAE replay',
                        'Spearman checked as Pearson correlation of average ranks','Pearson checked against numpy correlation'],
        'promotion_rule':'Require strictly lower same-row MAE AND strictly higher pooled Spearman IC than incumbent; retain incumbent if either fails. Pearson retained as diagnostic unless team specifies it as primary.'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'models':overall,'definition':result['definition'],'new_fits':0},indent=2))
