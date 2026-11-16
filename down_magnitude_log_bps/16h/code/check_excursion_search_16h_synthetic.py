"""Synthetic checks only; never opens market datasets."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import discover_excursion_features_16h as s


def main():
    rng = np.random.default_rng(48)
    index = pd.date_range('2022-04-22 00:30', '2025-01-22', freq='30min', tz='UTC')
    for issue in s.ISSUES:
        for tr, va in s.windows(index, issue):
            assert (index[tr]+pd.Timedelta(hours=16) < index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=16) < issue).all()
    n = 6000
    x = pd.DataFrame({'signal': rng.uniform(0, 10, n), 'noise': rng.normal(size=n), 'constant': 1.})
    y = 10 + 8*x.signal.to_numpy() + rng.normal(size=n)
    folds = [(np.arange(k), np.arange(k, k+500)) for k in [2000, 3000, 4000]]
    args = SimpleNamespace(screen_rows=1500, shortlist=2, max_features=2, min_gain=.001)
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
        p = Path(temp)
        (p/'a').mkdir()
        (p/'b').mkdir()
        first = s.search(x, y, folds, args, p/'a')
        assert 'signal' in first['features'] and 'constant' not in first['features']
        altered_x, altered_y = x.copy(), y.copy()
        altered_x.loc[4500:, :] = 999999
        altered_y[4500:] = 999999
        second = s.search(altered_x, altered_y, folds, args, p/'b')
        assert first == second, 'Outer observations affected selection'
        (p/'zero').mkdir()
        zero = s.search(x, np.zeros(n), folds, args, p/'zero')
        assert zero['features'] == [] and zero['model'] == 'median'
        a = np.array([[1., np.nan], [2., 3.], [3., 4.]])
        prep = s.transform_fit(a)
        assert np.isfinite(s.transform(a, prep)).all()
        assert s.metric(np.ones(3), np.ones(3))['ic'] is None
        for kind in ['ridge', 'tree', 'screen']:
            m = s.fit(x[['signal']].to_numpy()[:2000], y[:2000], kind)
            s.joblib.dump(m, p/'model.joblib')
            np.testing.assert_array_equal(s.predict(m, x[['signal']].to_numpy()[4500:]),
                s.predict(s.joblib.load(p/'model.joblib'), x[['signal']].to_numpy()[4500:]))
    print('PASS: 16h purges; signal selection; outer-data isolation; zero-target fallback; missing values; model replay.')


if __name__ == '__main__':
    main()

