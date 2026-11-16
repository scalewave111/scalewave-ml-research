"""Synthetic checks only; no market data or real-data fitting."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from long_horizon_features import build
import check_excursion_search_120h_synthetic as search_tests
import discover_long_features_120h as runner


def main():
    rng=np.random.default_rng(48)
    idx=pd.date_range('2022-01-01',periods=22000,freq='5min',tz='UTC')
    r=pd.Series(rng.normal(size=len(idx)),index=idx)
    cut=idx[21000].floor('h')
    origins=pd.DatetimeIndex([cut,cut+pd.Timedelta(minutes=30)])
    f,d=build(r,origins,120)
    assert len(f.columns)==104 and d.feature.is_unique
    assert f.notna().all().all()
    # Both half-hour samples use the same last completed hourly bin.
    np.testing.assert_array_equal(f.iloc[0],f.iloc[1])
    raw=r[(r.index>=cut-pd.Timedelta(hours=120))&(r.index<cut)]
    hourly=raw.to_numpy().reshape(120,12).sum(axis=1)
    level=np.r_[0.,np.cumsum(hourly)]
    np.testing.assert_allclose(f.iloc[0]['LONG__net__120h'],raw.sum())
    np.testing.assert_allclose(f.iloc[0]['LONG__vol_per_sqrt_hour__120h'],np.sqrt(np.mean(hourly**2)))
    np.testing.assert_allclose(f.iloc[0]['LONG__trend_efficiency__120h'],hourly.sum()/np.abs(hourly).sum())
    np.testing.assert_allclose(f.iloc[0]['LONG__range__120h'],np.ptp(level))
    np.testing.assert_allclose(f.iloc[0]['LONG__range_position__120h'],(level[-1]-level.min())/np.ptp(level))
    np.testing.assert_allclose(f.iloc[0]['LONG__up_variance_share__120h'],np.square(np.maximum(hourly,0)).sum()/np.square(hourly).sum())
    pd.testing.assert_frame_equal(f,build(r[r.index<=origins[-1]],origins,120)[0])
    changed=r.copy(); changed.loc[changed.index>=cut]=999999.
    pd.testing.assert_frame_equal(f,build(changed,origins,120)[0])
    missing=r.copy();missing.loc[cut-pd.Timedelta(minutes=5)]=np.nan
    assert build(missing,origins,120)[0].isna().all().all()
    assert build(r,pd.DatetimeIndex([idx[0]-pd.Timedelta(hours=1)]),120)[0].isna().all().all()
    zero,_=build(r*0,origins,120)
    assert not np.isinf(zero.to_numpy()).any()
    search_tests.main()
    # Exercise save/score/complete path with synthetic data and deterministic choice.
    dates=pd.date_range('2022-04-22','2025-01-22',freq='6h',tz='UTC')
    x=pd.DataFrame({'LONG__signal__120h':rng.uniform(size=len(dates))},index=dates)
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    y=pd.DataFrame({t:10+20*x.iloc[:,0] for t in runner.b.TARGETS},index=dates)
    dev=dates[dates>=runner.b.ISSUES[0]]
    rows=[]
    for t in runner.b.TARGETS:
        for stamp in dev:
            issue=runner.b.ISSUES[runner.b.ISSUES<=stamp][-1]
            rows.append(dict(decision_time=stamp,issue=issue,target=t,actual=y.loc[stamp,t],selected=20.))
    old=pd.DataFrame(rows)
    dictionary=pd.DataFrame([dict(feature=x.columns[0],formula='synthetic')])
    def fake_load(args):
        return x,y,dev,old,dictionary,[]
    def choose(*args):
        return dict(model='ridge',features=list(x.columns),validation_mae=1.)
    with tempfile.TemporaryDirectory() as temp:
        with patch.object(runner,'load',fake_load),patch.object(runner.b,'search',choose),patch.object(sys,'argv',['runner','--output-dir',temp]):
            runner.main()
        folders=list(Path(temp).iterdir());assert len(folders)==1
        done=json.loads((folders[0]/'COMPLETE.json').read_text())
        assert done['development_origins']==len(dev) and done['replay_passed']
        assert done['predictions_sha256']==runner.b.sha(folders[0]/'predictions.csv')
    print('PASS: 104 new features, direct formulas, strict endpoint exclusion, prefix/future isolation, missingness, shared search checks, synthetic pipeline and saved-model replay.')


if __name__=='__main__':
    main()
