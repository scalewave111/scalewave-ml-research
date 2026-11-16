"""Synthetic rolling maturity, causality and selector checks only."""
import numpy as np
import pandas as pd
from compare_rolling_medians_72h import causal_medians, choose


def main():
    rng = np.random.default_rng(72)
    index = pd.date_range('2021-01-01', periods=1200, freq='12h', tz='UTC')
    labels = pd.Series(rng.uniform(0, 1000, len(index)), index=index)
    labels = labels.drop(index[[20, 80, 100, 600]])
    queries = pd.DatetimeIndex([index[900], index[900]+pd.Timedelta(nanoseconds=1), index[1100]+pd.Timedelta(hours=2)])
    result, counts = causal_medians(labels, queries, days=[30, 90, 180, 365], minimum=2)
    for time in queries:
        mature = labels[labels.index+pd.Timedelta(hours=72) < time]
        np.testing.assert_allclose(result.loc[time, 'online_expanding'], mature.median())
        for days in [30, 90, 180, 365]:
            recent = mature[mature.index+pd.Timedelta(hours=72) >= time-pd.Timedelta(days=days)]
            np.testing.assert_allclose(result.loc[time, f'rolling_{days}d'], recent.median())
            assert counts.loc[time, f'rolling_{days}d'] == len(recent)
    cut = queries[0]
    changed = labels.copy()
    changed.loc[changed.index+pd.Timedelta(hours=72) >= cut] = 999999.
    prefix = labels[labels.index+pd.Timedelta(hours=72) < cut]
    for source in [changed, prefix]:
        replay, _ = causal_medians(source, pd.DatetimeIndex([cut]), minimum=2)
        np.testing.assert_allclose(replay.loc[cut], result.loc[cut])
    early, _ = causal_medians(labels, pd.DatetimeIndex([index[0]]), minimum=2)
    assert early.isna().all().all()
    truths = [np.array([10., 10.])]*3
    controls = [np.array([0., 0.])]*3
    selected, _ = choose(truths, controls, {'good': [np.array([9., 9.])]*3})
    assert selected == 'good'
    assert choose(truths, controls, {'equal': controls})[0] == 'quarterly_median'
    # Large pooled gain from only one window must not pass the consistency gate.
    truths = [np.array([100.]), np.array([1.]), np.array([1.])]
    controls = [np.array([0.])]*3
    assert choose(truths, controls, {'one': [np.array([100.]), np.array([3.]), np.array([3.])]})[0] == 'quarterly_median'
    print('PASS: exact maturity boundary, rolling bounds/counts, irregular keys, future isolation, prefix replay, missing history, selection and fallback')


if __name__ == '__main__':
    main()
