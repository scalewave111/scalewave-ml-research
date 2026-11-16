"""Synthetic-only checks for the 72h median-anchored experiment."""
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_neural_excursions_72h as s


def main():
    s.CONFIGS = {'test': dict(hidden=[16, 8], epochs=30, weight_decay=.01)}
    rng = np.random.default_rng(48)
    index = pd.date_range('2021-01-01', '2025-01-20', freq='3h', tz='UTC')
    for issue in s.base.ISSUES:
        for days in [0, 180, 365]:
            tr = s.training_rows(index, issue, days)
            assert (index[tr]+pd.Timedelta(hours=72) < issue).all()
            if days:
                assert (index[tr] >= issue-pd.Timedelta(days=days)).all()
    baseline = np.array([[10., 100.]]*3)
    assert s.admissible(baseline*.9, baseline, baseline, [1, 1, 1])
    assert not s.admissible(baseline, baseline, baseline, [1, 1, 1])
    assert not s.admissible(np.array([[1., 1.], [11., 1.], [11., 1.]]), baseline, baseline, [1, 1, 1])
    ri = pd.date_range('2023-01-01', periods=10000, freq='5min', tz='UTC')
    r = pd.Series(rng.normal(size=len(ri)), index=ri)
    cut = ri[9500]
    keys = pd.DatetimeIndex([cut])
    frame, groups = s.extended_price(r, keys)
    assert len(frame.columns) == 75
    for h in [72, 168, 336, 720]:
        raw = r[(r.index >= cut-pd.Timedelta(hours=h)) & (r.index < cut)]
        np.testing.assert_allclose(frame[f'NEWPRICE__rv__{h}h'].iloc[0], np.sqrt(np.square(raw).sum()))
    changed = r.copy()
    changed.loc[changed.index >= cut] = 99999.
    pd.testing.assert_frame_equal(frame, s.extended_price(changed, keys)[0])
    pd.testing.assert_frame_equal(frame, s.extended_price(r[r.index <= cut], keys)[0])
    x = pd.DataFrame({'noise': rng.normal(size=len(index)), 'signal': rng.uniform(0, 10, len(index))}, index=index)
    y = 10+20*x.signal.to_numpy()
    issue = s.base.ISSUES[-1]
    spec = dict(features=['noise'], model='tree')
    groups = {f'price_{h}h': ['signal'] for h in [72, 168, 336, 720]}
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
        root = Path(temp)
        for name in ['a', 'b', 'zero']:
            (root/name).mkdir()
        chosen = s.run_cycle(x, y, issue, spec, groups, root/'a')
        assert chosen is not None and 'signal' in chosen['features']
        changed, yy = x.copy(), y.copy()
        changed.loc[index >= issue] = 99999.
        yy[index >= issue] = 99999.
        assert chosen == s.run_cycle(changed, yy, issue, spec, groups, root/'b')
        assert s.run_cycle(x, np.zeros(len(y)), issue, spec, groups, root/'zero') is None
        tr = s.training_rows(index, issue, chosen['days'])
        model = s.fit_candidate(x.iloc[tr][chosen['features']].to_numpy(), y[tr], 'test')
        assert model['config']['hidden'] == [16, 8]
        assert model['scale'] == max(float(np.median(y[tr])), 1.)
        control = s.base.fit(x.iloc[tr][['noise']].to_numpy(), y[tr], 'tree')
        bundle = dict(control=control, control_features=['noise'], candidate=model, selection=chosen, median=20.)
        sample = x.tail(20).copy()
        sample.iloc[0] = np.nan
        expected = chosen['weight']*s.predict_candidate(model, sample[chosen['features']].to_numpy())+(1-chosen['weight'])*20.
        joblib.dump(bundle, root/'model.joblib')
        np.testing.assert_allclose(s.predict_bundle(joblib.load(root/'model.joblib'), sample), expected)
        bundle['candidate'] = None
        np.testing.assert_array_equal(s.predict_bundle(bundle, sample), np.full(len(sample), 20.))
    print('PASS: strict maturity, recency, long-price formulas/causality, gate, signal, outer isolation, zero fallback, missingness and serialized replay')


if __name__ == '__main__':
    main()
