"""Synthetic checks: formulas, missing windows, causality, selection isolation, replay."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import discover_long_features_200h as s


def main():
    rng=np.random.default_rng(48)
    idx=pd.date_range('2022-01-01',periods=21000,freq='5min',tz='UTC')
    r=pd.Series(rng.normal(size=len(idx)),index=idx)
    market=pd.DataFrame({f'F4Z__{family}__{measure}__4h__value':rng.random(len(idx[::6]))
                        for family,measure in s.features.MARKET.items()},index=idx[::6])
    cut=idx[20000]
    keys=pd.DatetimeIndex([cut])
    a,defs=s.features.make_features(r,keys,market)
    assert len(a.columns)==116 and len(defs)==116
    raw=r[(r.index>=cut-pd.Timedelta(hours=200))&(r.index<cut)].to_numpy()
    path=np.r_[0,np.cumsum(raw)]
    np.testing.assert_allclose(a['LONGPRICE__net__200h'].iloc[0],raw.sum())
    np.testing.assert_allclose(a['LONGPRICE__rv__200h'].iloc[0],np.sqrt((raw**2).sum()))
    np.testing.assert_allclose(a['LONGPRICE__drawdown__200h'].iloc[0],path.max()-path[-1],atol=1e-9)
    np.testing.assert_allclose(a['LONGPRICE__range__200h'].iloc[0],np.ptp(path),atol=1e-9)
    np.testing.assert_allclose(a['LONGPRICE__trend_efficiency__200h'].iloc[0],raw.sum()/abs(raw).sum())
    col='F4Z__flow__flow_pressure_percentile__4h__value'
    window=market.loc[(market.index>=cut-pd.Timedelta(hours=168))&(market.index<cut),col]
    np.testing.assert_allclose(a['LONGMARKET__flow_mean__168h'].iloc[0],window.mean())
    changed=r.copy(); changed.loc[changed.index>=cut]=99999
    m=market.copy();m.loc[m.index>=cut]=99999
    pd.testing.assert_frame_equal(a,s.features.make_features(changed,keys,m)[0])
    pd.testing.assert_frame_equal(a,s.features.make_features(r.loc[:cut],keys,market.loc[:cut])[0])
    missing=r.copy();missing.loc[idx[19999]]=np.nan
    assert s.features.make_features(missing,keys)[0].isna().all().all()
    flat=s.features.make_features(r*0,keys)[0]
    assert np.isfinite(flat.to_numpy()).all() and (flat==0).all().all()
    index=pd.date_range('2021-01-01','2025-01-22',freq='6h',tz='UTC')
    issue=s.base.ISSUES[-1]
    folds=s.base.windows(index,issue)
    for tr,va in folds:
        assert (index[tr]+pd.Timedelta(hours=200)<index[va].min()).all()
    x=pd.DataFrame({'signal':rng.uniform(0,10,len(index))},index=index)
    y=10+20*x.signal.to_numpy()+rng.normal(size=len(index))
    args=SimpleNamespace(shortlist=1,max_features=1,screen_rows=2000,min_gain=.001)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        root=Path(temp)
        (root/'a').mkdir();(root/'b').mkdir()
        choice=s.base.search(x,y,folds,args,root/'a')
        gate=s.baseline_gate(choice,y,folds)
        assert gate['accepted']
        y2=y.copy();y2[index>=issue]=1e8
        x2=x.copy();x2.loc[index>=issue]=1e8
        other=s.base.search(x2,y2,folds,args,root/'b')
        assert choice==other and gate==s.baseline_gate(other,y2,folds)
        assert not s.baseline_gate(dict(features=[],validation_window_mae=[0,0,0]),y,folds)['accepted']
        bad=dict(features=['signal'],validation_window_mae=[1e8]*3)
        assert not s.baseline_gate(bad,y,folds)['accepted']
        tr=np.flatnonzero(index+pd.Timedelta(hours=200)<issue)
        model=s.base.fit(x.iloc[tr].to_numpy(),y[tr],choice['model'])
        joblib.dump(model,root/'model.joblib')
        np.testing.assert_allclose(s.base.predict(model,x.to_numpy()),s.base.predict(joblib.load(root/'model.joblib'),x.to_numpy()))
    print('PASS:116 features, direct formulas, missingness, flat paths, future isolation, 200h purges, selection and replay')


if __name__=='__main__':
    main()
