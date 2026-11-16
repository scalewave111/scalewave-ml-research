"""Synthetic checks only; no real-data model training."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import refine_long_features_300h as s


def main():
    rng = np.random.default_rng(48)
    ri = pd.date_range('2022-01-01', periods=18000, freq='5min', tz='UTC')
    r = pd.Series(rng.normal(size=len(ri)), index=ri)
    cut = ri[-2]
    keys = pd.DatetimeIndex([cut])
    f, groups = s.long_features.features(r, keys)
    assert f.shape == (1, 54)
    for h in s.long_features.HOURS:
        raw = r[(r.index >= cut-pd.Timedelta(hours=h)) & (r.index < cut)].to_numpy()
        path = np.r_[0, np.cumsum(raw)]
        variance = np.square(raw).sum()
        expected = dict(net=raw.sum(), rv=np.sqrt(variance),
            trend_to_vol=raw.sum()/np.sqrt(variance), efficiency=raw.sum()/np.abs(raw).sum(),
            range_position=(path[-1]-path.min())/(path.max()-path.min()),
            drawdown_from_high=path.max()-path[-1], rebound_from_low=path[-1]-path.min(),
            range_to_vol=(path.max()-path.min())/np.sqrt(variance),
            up_rv=np.sqrt(np.square(np.maximum(raw, 0)).sum()),
            down_rv=np.sqrt(np.square(np.minimum(raw, 0)).sum()),
            downside_share=np.square(np.minimum(raw, 0)).sum()/variance,
            jump_concentration=np.square(raw).max()/variance)
        for name, v in expected.items():
            np.testing.assert_allclose(f[f'LONG300__{name}__{h}h'].iloc[0], v, atol=1e-9)
    for a, b in [(168,720), (300,720), (720,1440)]:
        np.testing.assert_allclose(f[f'LONG300__vol_rate_ratio__{a}h_{b}h'],
            f[f'LONG300__rv__{a}h']/np.sqrt(a)/(f[f'LONG300__rv__{b}h']/np.sqrt(b)))
        np.testing.assert_allclose(f[f'LONG300__trend_rate_change__{a}h_{b}h'],
            f[f'LONG300__net__{a}h']/a-f[f'LONG300__net__{b}h']/b)
    changed = r.copy()
    changed.loc[changed.index >= cut] = 99999.
    pd.testing.assert_frame_equal(f, s.long_features.features(changed, keys)[0])
    pd.testing.assert_frame_equal(f, s.long_features.features(r.loc[:cut], keys)[0])
    missing = r.copy()
    missing.iloc[-3] = np.nan
    assert s.long_features.features(missing, keys)[0].isna().all().all()
    assert s.long_features.features(r, ri[:10])[0].isna().all().all()
    assert np.isfinite(s.long_features.features(r*0, keys)[0]).all().all()
    try:
        s.long_features.features(r.drop(ri[3]), keys)
        raise AssertionError('Gap was accepted')
    except ValueError:
        pass
    index = pd.date_range('2021-01-01', '2025-01-20', freq='3h', tz='UTC')
    for issue in s.base.ISSUES:
        tr = s.training_rows(index, issue)
        assert (index[tr]+pd.Timedelta(hours=300) < issue).all()
        for tr, va in s.base.windows(index, issue):
            assert (index[tr]+pd.Timedelta(hours=300) < index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=300) < issue).all()
    baseline = np.array([[10.,100.]]*3)
    assert s.admissible(baseline*.9, baseline, baseline, [1,1,1])
    assert not s.admissible(baseline, baseline, baseline, [1,1,1])
    assert not s.admissible(np.array([[1.,1.],[11.,1.],[11.,1.]]), baseline, baseline, [1,1,1])
    x = pd.DataFrame(dict(noise=rng.normal(size=len(index)), signal=rng.uniform(0,10,len(index)), missing=np.nan), index=index)
    y = 10+20*x.signal.to_numpy()
    issue = s.base.ISSUES[-1]
    spec = dict(features=['noise'], model='tree')
    groups = dict(trend_path=['signal','missing'], volatility_regime=['noise'])
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
        root = Path(temp)
        for name in ['a','b','zero']:
            (root/name).mkdir()
        chosen = s.run_cycle(x,y,issue,spec,groups,root/'a')
        assert chosen is not None and 'signal' in chosen['features'] and 'missing' not in chosen['features']
        changed, yy = x.copy(), y.copy()
        changed.loc[index >= issue] = 99999.
        yy[index >= issue] = 99999.
        assert chosen == s.run_cycle(changed,yy,issue,spec,groups,root/'b')
        assert s.run_cycle(x,np.zeros(len(y)),issue,spec,groups,root/'zero') is None
        tr = s.training_rows(index,issue)
        model = s.fit_candidate(x.iloc[tr][chosen['features']].to_numpy(),y[tr],'original')
        bundle = dict(candidate=model,selection=chosen,median=float(np.median(y[tr])))
        sample = x.tail(20).copy()
        sample.iloc[0] = np.nan
        expected = s.predict_bundle(bundle,sample)
        joblib.dump(bundle,root/'model.joblib')
        np.testing.assert_allclose(s.predict_bundle(joblib.load(root/'model.joblib'),sample),expected)
        bundle['candidate'] = None
        np.testing.assert_array_equal(s.predict_bundle(bundle,sample),np.full(len(sample),bundle['median']))
    print('PASS: 54 formulas, causality, missing/gap/zero handling, 300h maturity, admission, selection, outer isolation, fallback and serialization')


if __name__ == '__main__':
    main()
