"""Synthetic tests only. No market data or real-data training."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import tune_excursions_120h as s


def main():
    rng = np.random.default_rng(48)
    index = pd.date_range('2021-01-01', '2025-01-22', freq='6h', tz='UTC')
    for issue in s.base.ISSUES:
        for days in [0, 365]:
            tr = s.training_rows(index, issue, days)
            assert (index[tr]+pd.Timedelta(hours=120) < issue).all()
            if days:
                assert (index[tr] >= issue-pd.Timedelta(days=days)).all()
        for tr, va in s.base.windows(index, issue):
            assert (index[tr]+pd.Timedelta(hours=120) < index[va].min()).all()
    baseline = np.array([[10., 200.]]*3)
    assert s.admissible(np.array([[9., 199.]]*3), baseline, baseline, [1, 1, 1])
    assert not s.admissible(np.array([[9., 201.]]*3), baseline, baseline, [1, 1, 1])
    assert not s.admissible(baseline, baseline, baseline, [1, 1, 1])
    assert not s.admissible(np.array([[1., 100.], [11., 100.], [11., 100.]]), baseline, baseline, [1, 1, 1])
    ri = pd.date_range('2023-01-01', periods=5000, freq='5min', tz='UTC')
    r = pd.Series(rng.normal(size=len(ri)), index=ri)
    cut = ri[4000]
    frame, groups = s.extended_price(r, pd.DatetimeIndex([cut]))
    assert len(frame.columns) == 59 and len(groups['price_120h']) == 8
    assert s.prior.HOURS == [1, 4, 12, 24, 48, 168]
    raw = r[(r.index >= cut-pd.Timedelta(hours=120)) & (r.index < cut)]
    np.testing.assert_allclose(frame['NEWPRICE__rv__120h'].iloc[0], np.sqrt(np.square(raw).sum()))
    pd.testing.assert_frame_equal(frame, s.extended_price(r[r.index <= cut], pd.DatetimeIndex([cut]))[0])
    altered = r.copy()
    altered.loc[altered.index >= cut] = 99999.
    pd.testing.assert_frame_equal(frame, s.extended_price(altered, pd.DatetimeIndex([cut]))[0])
    missing = r.copy()
    missing.loc[ri[3999]] = np.nan
    assert s.extended_price(missing, pd.DatetimeIndex([cut]))[0].isna().all().all()
    x = pd.DataFrame({'F4Z__noise': rng.normal(size=len(index)),
                      'signal': rng.uniform(0, 10, len(index))}, index=index)
    y = 10+20*x.signal.to_numpy()+rng.normal(size=len(index))
    issue = s.base.ISSUES[-1]
    spec = dict(features=['F4Z__noise'], model='tree')
    s.CONFIGS = {'test': dict(max_iter=12, max_leaf_nodes=7, max_depth=3,
                            min_samples_leaf=32, l2_regularization=1.)}
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
        root = Path(temp)
        for name in ['a', 'b', 'zero']:
            (root/name).mkdir()
        chosen = s.run_cycle(x, y, issue, spec, {'price_120h': ['signal']}, root/'a')
        assert chosen is not None and 'signal' in chosen['features']
        # Alter only observations outside every inner validation window.
        changed, yy = x.copy(), y.copy()
        outer = index >= issue
        changed.loc[outer] = 99999.
        yy[outer] = 99999.
        repeated = s.run_cycle(changed, yy, issue, spec, {'price_120h': ['signal']}, root/'b')
        assert chosen == repeated, 'Outer outcomes changed earlier selection'
        assert s.run_cycle(x, np.zeros(len(y)), issue, spec, {'price_120h': ['signal']}, root/'zero') is None
        tr = s.training_rows(index, issue)
        control = s.base.fit(x.iloc[tr][spec['features']].to_numpy(), y[tr], 'tree')
        model = s.fit_candidate(x.iloc[tr][chosen['features']].to_numpy(), y[tr], 'test')
        bundle = dict(control=control, control_features=spec['features'], candidate=model, selection=chosen)
        joblib.dump(bundle, root/'model.joblib')
        np.testing.assert_array_equal(s.predict_bundle(bundle, x), s.predict_bundle(joblib.load(root/'model.joblib'), x))
        bundle['candidate'] = None
        np.testing.assert_array_equal(s.predict_bundle(bundle, x), s.base.predict(control, x[spec['features']].to_numpy()))
        a = np.array([[1., np.nan], [2., 3.], [3., 4.]]*50)
        m = s.fit_candidate(a, np.arange(len(a), dtype=float), 'test')
        assert np.isfinite(s.predict_candidate(m, a)).all()
    print('PASS: 120h/365-day purges, MAE/RMSE gate, 120h price formulas/causality/missingness, signal selection, outer isolation, fallback, NaN fitting and serialized blend replay.')


if __name__ == '__main__':
    main()

