"""Synthetic orchestration tests; never train on project data."""
import contextlib, io, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import run_window_comparison as r

class FakeTree:
    def predict(self,A):return np.full(len(A),5.)

class FakeForest(FakeTree):
    calls=0
    fail_at=None
    def __init__(self,**params):self.estimators_=[FakeTree() for _ in range(params['n_estimators'])]
    def fit(self,A,y):
        type(self).calls+=1
        if type(self).calls==type(self).fail_at:raise KeyboardInterrupt()
        return self

def run_case(winner):
    with tempfile.TemporaryDirectory(dir=r.HERE,prefix='window_synthetic_') as directory:
        root=Path(directory);current=root/'current';current.mkdir();out=root/'trial'
        index=pd.date_range('2023-01-01',periods=2400,freq='h',tz='UTC')
        X=pd.DataFrame({'f':np.sin(np.arange(len(index)))},index=index)
        y=pd.DataFrame({'actual':10+np.arange(len(index))%3,'label_end':index+pd.Timedelta(hours=48)},index=index)
        folds=[];frames=[];windows=[]
        for fold in range(1,5):
            ov=index[1800+fold*50:1820+fold*50];ot=index[y.label_end<ov.min()]
            folds.append((fold,ot,ov,ot,ov));windows.append([str(ov.min()),str(ov.max()+pd.Timedelta(hours=1))])
            frames.append(pd.DataFrame({'actual':y.loc[ov,'actual'],'prediction':0. if winner else y.loc[ov,'actual'],'pair':8.,'fold':fold},index=ov))
            r.write(current/f'fold_{fold}_selected.json',{'selection':{'weight':.5 if fold<4 else 0.,'depth':5,'leaf':64,'max_samples':16384}})
        base=pd.concat(frames);reg={'current':r.metric(base.actual,base.prediction)}
        c={'validation_windows':windows,'label_end_column':'label_end','target_column':'actual',
           'forest':{'n_estimators':128},'seeds':[48,17,91]}
        FakeForest.calls=0;FakeForest.fail_at=2
        with patch.multiple(r,CURRENT=current,OUT=out,HERE=root,RandomForestRegressor=FakeForest), \
             patch.object(r,'preflight',return_value=(c,reg,X,y,base)), \
             patch.object(r,'inputs',return_value={}),patch.object(r,'splits',return_value=folds), \
             patch.object(r.sys,'argv',['runner']),contextlib.redirect_stdout(io.StringIO()):
            try:r.main()
            except KeyboardInterrupt:pass
            assert len(list(out.glob('*.joblib')))==1
            FakeForest.fail_at=None;r.main()
            assert FakeForest.calls==28  # One interrupted fit retried; first fit reused.
            state=r.read(out/'state.json');assert len(state['completed_fits'])==27
            assert (state['leader'] is not None)==winner
            assert len(list(out.glob('*.joblib')))==(9 if winner else 0)
            assert len(list(out.glob('*.parquet')))==(1 if winner else 0)
            r.main();assert FakeForest.calls==28  # Completed rerun does not train.
            assert (out/'WINDOW_RESULTS.txt').exists()

if __name__=='__main__':
    idx=pd.date_range('2023-01-01','2024-01-01',freq='D',tz='UTC')
    assert r.window(idx,pd.Timestamp('2024-01-01',tz='UTC'),6).min()==pd.Timestamp('2023-07-01',tz='UTC')
    run_case(True);run_case(False)
    print('PASS: window boundary, interrupted-fit resume, completed-fit reuse, replay, winning retention, losing cleanup, completed rerun. Synthetic data only.')
