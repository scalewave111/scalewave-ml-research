"""Causal long-window price features; no target data or learned cutoffs."""
import numpy as np
import pandas as pd

HOURS = [168, 300, 720, 1440]


def features(r, origins):
    """r contains endpoint-indexed five-minute log-bps returns.

    Every input endpoint is strictly before origin. Complete windows required.
    Path levels are relative log prices; gaps invalidate the whole window.
    """
    if not r.index.is_unique or not r.index.is_monotonic_increasing:
        raise ValueError('Return endpoints must be unique and sorted')
    if len(r) > 1 and not (np.diff(r.index.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError('Return grid must be complete at five-minute spacing')
    if not origins.isin(r.index).all():
        raise ValueError('Origins must be on the return endpoint grid')
    level = r.fillna(0).cumsum().shift(1)
    values, groups = {}, {'trend_path': [], 'volatility_regime': []}
    rv, net = {}, {}
    for h in HOURS:
        n = 12*h
        opts = dict(window=n, min_periods=n)
        past = r.shift(1)
        valid = past.rolling(**opts).count().eq(n)
        net[h] = past.rolling(**opts).sum()
        variance = past.pow(2).rolling(**opts).sum()
        rv[h] = np.sqrt(variance)
        down = past.clip(upper=0).pow(2).rolling(**opts).sum()
        high = level.rolling(n+1, min_periods=n+1).max()
        low = level.rolling(n+1, min_periods=n+1).min()
        width = high-low
        path_length = past.abs().rolling(**opts).sum()
        stats = {
            'trend_path': dict(net=net[h], trend_to_vol=net[h]/rv[h].clip(lower=1e-6),
                efficiency=net[h]/path_length.clip(lower=1e-6),
                range_position=(level-low)/width.clip(lower=1e-6),
                drawdown_from_high=high-level, rebound_from_low=level-low,
                range_to_vol=width/rv[h].clip(lower=1e-6)),
            'volatility_regime': dict(rv=rv[h], up_rv=np.sqrt((variance-down).clip(lower=0)),
                down_rv=np.sqrt(down), downside_share=down/variance.clip(lower=1e-12),
                jump_concentration=past.pow(2).rolling(**opts).max()/variance.clip(lower=1e-12))}
        for group, measures in stats.items():
            for name, value in measures.items():
                col = f'LONG300__{name}__{h}h'
                values[col] = value.where(valid).reindex(origins)
                groups[group].append(col)
    for short, long in [(168, 720), (300, 720), (720, 1440)]:
        col = f'LONG300__vol_rate_ratio__{short}h_{long}h'
        values[col] = (rv[short]/np.sqrt(short)/(rv[long]/np.sqrt(long)).clip(lower=1e-6)).reindex(origins)
        groups['volatility_regime'].append(col)
        col = f'LONG300__trend_rate_change__{short}h_{long}h'
        values[col] = (net[short]/short-net[long]/long).reindex(origins)
        groups['trend_path'].append(col)
    return pd.DataFrame(values, index=origins).replace([np.inf, -np.inf], np.nan), groups
