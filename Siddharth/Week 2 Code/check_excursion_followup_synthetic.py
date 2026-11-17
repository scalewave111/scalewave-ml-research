"""Synthetic-only checks for the follow-up price-feature experiment."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import improve_excursion_features_4h as s


def main():
    rng = np.random.default_rng(48)
    index = pd.date_range('2023-01-01', periods=5000, freq='5min', tz='UTC')
    r = pd.Series(rng.normal(size=len(index)), index=index)
    origin = index[4000]
    features, groups = s.price_features(r, pd.DatetimeIndex([origin]))
    assert len(features.columns) == 51
    for h in s.HOURS:
        raw = r[(r.index >= origin-pd.Timedelta(hours=h)) & (r.index < origin)]
        assert len(raw) == h*12
        path = np.r_[0., np.cumsum(raw.to_numpy())]
        expected = dict(rv=np.sqrt(np.sum(raw**2)), up_rv=np.sqrt(np.sum(raw.clip(lower=0)**2)),
            down_rv=np.sqrt(np.sum(raw.clip(upper=0)**2)), net=raw.sum(),
            past_range=path.max()-path.min(), distance_from_high=path.max()-path[-1],
            distance_from_low=path[-1]-path.min(), max_abs_return=raw.abs().max())
        for name, value in expected.items():
            np.testing.assert_allclose(features.iloc[0][f'NEWPRICE__{name}__{h}h'], value, rtol=0, atol=1e-9)
    s.causal_check(r, index[2500:4500:10])
    missing = r.copy()
    missing.loc[index[3999]] = np.nan
    bad, _ = s.price_features(missing, pd.DatetimeIndex([origin]))
    assert bad.isna().all().all(), 'Missing return window was silently filled'
    assert s.gate(np.array([9., 9., 11.]), np.array([10., 10., 10.]), [1, 1, 1], .001)
    assert not s.gate(np.array([1., 11., 11.]), np.array([10., 10., 10.]), [1, 1, 1], .001)
    assert not s.gate(np.array([10., 10., 10.]), np.array([10., 10., 10.]), [1, 1, 1], .001)
    n = 6000
    x = pd.DataFrame({'anchor': rng.uniform(size=n), 'new_signal': rng.uniform(size=n), 'noise': rng.normal(size=n)})
    y = 20 + 100*x.new_signal.to_numpy() + rng.normal(size=n)
    folds = [(np.arange(k), np.arange(k, k+500)) for k in [2000, 3000, 4000]]
    groups = {'signal': ['new_signal'], 'noise': ['noise']}
    anchor = dict(features=['anchor'], model='tree')
    args = SimpleNamespace(max_blocks=2, min_gain=.001)
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
        p = Path(temp)
        for name in ['a', 'b', 'zero']:
            (p/name).mkdir()
        first = s.search(x, y, folds, anchor, groups, args, p/'a')
        assert first['selected_variant'] != 'original'
        assert 'new_signal' in first['variants'][first['selected_variant']]['features']
        changed, yy = x.copy(), y.copy()
        changed.loc[4500:, :] = 99999
        yy[4500:] = 99999
        second = s.search(changed, yy, folds, anchor, groups, args, p/'b')
        assert first == second, 'Outer data affected selection'
        zero = s.search(x, np.zeros(n), folds, anchor, groups, args, p/'zero')
        assert zero['selected_variant'] == 'original', 'Fallback failed'
    print('PASS: direct rolling formulas, missing-window masking, prefix/future causality, gain gate, new signal selection, outer isolation, fallback.')


if __name__ == '__main__':
    main()
